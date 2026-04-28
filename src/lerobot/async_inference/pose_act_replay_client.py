from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import pickle  # nosec
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import grpc
import torch
from PIL import Image

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation, get_logger
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

POSE7D_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper_width")


@dataclass
class LoadedObservation:
    raw_observation: dict[str, object]
    preview_image: torch.Tensor
    task: str | None
    episode_idx: int
    frame_idx: int


class ObservationSource(Protocol):
    def get_observation(self, episode_idx: int, frame_idx: int) -> dict[str, object]:
        ...


class ActionChunkConsumer(Protocol):
    def consume(self, actions: list[TimedAction], metadata: dict[str, object] | None = None) -> None:
        ...


class RemoteObservationSource:
    def get_observation(self, episode_idx: int, frame_idx: int) -> dict[str, object]:
        raise NotImplementedError("RemoteObservationSource is a placeholder for future real-time robot ingestion.")


class LatestChunkBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: list[TimedAction] | None = None

    def replace(self, actions: list[TimedAction]) -> None:
        with self._lock:
            self._latest = actions

    def pop_latest(self) -> list[TimedAction] | None:
        with self._lock:
            actions = self._latest
            self._latest = None
            return actions


class FifoChunkBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: deque[dict[str, object]] = deque()

    def push(self, payload: dict[str, object]) -> None:
        with self._lock:
            self._queue.append(payload)

    def pop_next(self) -> dict[str, object] | None:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()


class SocketActionChunkConsumer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def consume(self, actions: list[TimedAction], metadata: dict[str, object] | None = None) -> None:
        metadata = metadata or {}
        payload = pickle.dumps(
            {
                "schema": "pose_act_pose7d_chunk_v1",
                "actions": [action.action.detach().to(torch.float32).cpu().tolist() for action in actions],
                "timestamps": [float(action.timestamp) for action in actions],
                "timesteps": [int(action.timestep) for action in actions],
                "request_id": metadata.get("request_id"),
                "episode_idx": metadata.get("episode_idx"),
                "frame_idx": metadata.get("frame_idx"),
                "source_timestep": metadata.get("source_timestep"),
            }
        )
        with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
            sock.sendall(len(payload).to_bytes(8, byteorder="big"))
            sock.sendall(payload)


class ActionChunkSocketServer:
    def __init__(self, host: str, port: int, buffer: LatestChunkBuffer, logger_name: str = "pose_act_bridge"):
        self.host = host
        self.port = port
        self.buffer = buffer
        self.logger = get_logger(logger_name, log_to_file=False)
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._server_socket: socket.socket | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        if self._server_socket is not None:
            self._server_socket.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()
            server.settimeout(0.5)
            self._server_socket = server
            self.logger.info(f"Listening for pose_act chunks on {self.host}:{self.port}")

            while not self._shutdown.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._shutdown.is_set():
                        break
                    raise

                with conn:
                    try:
                        payload = _recv_exact(conn, 8)
                        size = int.from_bytes(payload, byteorder="big")
                        data = _recv_exact(conn, size)
                        actions = _decode_socket_payload(data)
                        self.buffer.replace(actions)
                        self.logger.info(
                            f"Received chunk from {addr[0]}:{addr[1]} with {len(actions)} steps; replaced previous queue."
                        )
                    except Exception as exc:  # pragma: no cover - defensive logging
                        self.logger.error(f"Failed to receive pose_act chunk: {exc}")


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError(f"Socket closed before receiving {size} bytes.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode_socket_payload(data: bytes) -> list[TimedAction]:
    payload = pickle.loads(data)  # nosec
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict) or payload.get("schema") != "pose_act_pose7d_chunk_v1":
        raise TypeError("Unsupported pose_act socket payload.")
    actions = []
    for action, timestamp, timestep in zip(
        payload["actions"], payload["timestamps"], payload["timesteps"], strict=True
    ):
        actions.append(
            TimedAction(
                timestamp=float(timestamp),
                timestep=int(timestep),
                action=torch.tensor(action, dtype=torch.float32),
            )
        )
    return actions


class LeRobotReplaySource:
    def __init__(self, repo_id: str, root: str | Path | None = None, camera_key: str = "fisheye_rgb") -> None:
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else None
        self.camera_key = camera_key.removeprefix(f"{OBS_IMAGES}.")

    @property
    def lerobot_features(self) -> dict[str, dict]:
        return {
            OBS_STATE: {
                "dtype": "float32",
                "shape": (len(POSE7D_NAMES),),
                "names": list(POSE7D_NAMES),
            },
            f"{OBS_IMAGES}.{self.camera_key}": {
                "dtype": "image",
                "shape": (0, 0, 3),
                "names": ["height", "width", "channels"],
            },
        }

    def load_history(
        self,
        episode_idx: int,
        frame_idx: int,
        n_obs_steps: int,
        task: str | None = None,
    ) -> LoadedObservation:
        if n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {n_obs_steps}")

        dataset = LeRobotDataset(self.repo_id, root=self.root, episodes=[episode_idx], download_videos=True)
        if frame_idx < 0:
            raise IndexError(f"Frame {frame_idx} is out of range for episode {episode_idx}.")

        history_frames: list[dict[str, object]] = []
        start_idx = max(0, frame_idx - n_obs_steps + 1)
        for idx in range(start_idx, frame_idx + 1):
            history_frames.append(dataset[idx])
        if len(history_frames) == 0:
            raise IndexError(f"Frame {frame_idx} is out of range for episode {episode_idx}.")
        while len(history_frames) < n_obs_steps:
            history_frames.insert(0, history_frames[0])

        image_key = f"{OBS_IMAGES}.{self.camera_key}"
        state_history = []
        image_history = []
        loaded_task = task
        for frame in history_frames:
            if OBS_STATE not in frame:
                raise KeyError(f"Dataset frame is missing required key {OBS_STATE!r}.")
            if image_key not in frame:
                raise KeyError(
                    f"Dataset frame is missing required camera {image_key!r}. "
                    f"Available keys: {sorted(frame.keys())}"
                )
            state = _to_float_tensor(frame[OBS_STATE]).flatten()
            if state.numel() != len(POSE7D_NAMES):
                raise ValueError(
                    f"Expected pose_act state width {len(POSE7D_NAMES)}, got {state.numel()} for "
                    f"episode={episode_idx} frame={frame_idx}."
                )
            state_history.append(state)
            image_history.append(_to_image_hwc_uint8(frame[image_key]))
            if loaded_task is None and isinstance(frame.get("task"), str):
                loaded_task = frame["task"]

        preview_image = image_history[-1]
        raw_observation: dict[str, object] = {
            OBS_STATE: torch.stack(state_history, dim=0).numpy(),
            image_key: torch.stack(image_history, dim=0).numpy(),
        }
        if loaded_task:
            raw_observation["task"] = loaded_task

        return LoadedObservation(
            raw_observation=raw_observation,
            preview_image=preview_image,
            task=loaded_task,
            episode_idx=episode_idx,
            frame_idx=frame_idx,
        )

    def get_num_frames(self, episode_idx: int) -> int:
        dataset = LeRobotDataset(self.repo_id, root=self.root, episodes=[episode_idx], download_videos=True)
        return dataset.num_frames

    def load_frame(self, episode_idx: int, frame_idx: int, task: str | None = None) -> LoadedObservation:
        dataset = LeRobotDataset(self.repo_id, root=self.root, episodes=[episode_idx], download_videos=True)
        try:
            frame = dataset[frame_idx]
        except IndexError as exc:
            raise IndexError(f"Frame {frame_idx} is out of range for episode {episode_idx}.") from exc

        if OBS_STATE not in frame:
            raise KeyError(f"Dataset frame is missing required key {OBS_STATE!r}.")

        image_key = f"{OBS_IMAGES}.{self.camera_key}"
        if image_key not in frame:
            raise KeyError(
                f"Dataset frame is missing required camera {image_key!r}. "
                f"Available keys: {sorted(frame.keys())}"
            )

        state = _to_float_tensor(frame[OBS_STATE]).flatten()
        if state.numel() != len(POSE7D_NAMES):
            raise ValueError(
                f"Expected pose_act state width {len(POSE7D_NAMES)}, got {state.numel()} for episode={episode_idx} "
                f"frame={frame_idx}."
            )

        image = _to_image_hwc_uint8(frame[image_key])
        loaded_task = task
        if loaded_task is None and isinstance(frame.get("task"), str):
            loaded_task = frame["task"]

        raw_observation: dict[str, object] = {
            **{name: float(value) for name, value in zip(POSE7D_NAMES, state.tolist(), strict=True)},
            self.camera_key: image.numpy(),
        }
        if loaded_task:
            raw_observation["task"] = loaded_task

        return LoadedObservation(
            raw_observation=raw_observation,
            preview_image=image,
            task=loaded_task,
            episode_idx=episode_idx,
            frame_idx=frame_idx,
        )

    def get_observation(self, episode_idx: int, frame_idx: int) -> dict[str, object]:
        return self.load_frame(episode_idx, frame_idx).raw_observation


class PoseActReplayClient:
    def __init__(
        self,
        source: LeRobotReplaySource,
        server_address: str,
        pretrained_name_or_path: str,
        policy_device: str = "cpu",
        actions_per_chunk: int = 16,
        consumer: ActionChunkConsumer | None = None,
    ) -> None:
        self.source = source
        self.server_address = server_address
        self.pretrained_name_or_path = pretrained_name_or_path
        self.policy_device = policy_device
        self.actions_per_chunk = actions_per_chunk
        self.consumer = consumer
        self.logger = get_logger("pose_act_replay_client")
        self._policy_config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        if not hasattr(self._policy_config, "n_obs_steps"):
            raise ValueError(
                f"Checkpoint at {pretrained_name_or_path} does not expose n_obs_steps; "
                "it is not a compatible pose_act checkpoint."
            )

    @contextmanager
    def _session(self):
        policy_config = RemotePolicyConfig(
            policy_type="pose_act",
            pretrained_name_or_path=self.pretrained_name_or_path,
            lerobot_features=self.source.lerobot_features,
            actions_per_chunk=self.actions_per_chunk,
            device=self.policy_device,
        )
        channel = grpc.insecure_channel(
            self.server_address,
            grpc_channel_options(initial_backoff="0.1000s"),
        )
        try:
            stub = services_pb2_grpc.AsyncInferenceStub(channel)
            stub.Ready(services_pb2.Empty())
            stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy_config)))
            yield stub
        except grpc.RpcError as exc:
            raise RuntimeError(f"Failed to reach pose_act replay server at {self.server_address}: {exc}") from exc
        finally:
            channel.close()

    def run_once(
        self,
        episode_idx: int,
        frame_idx: int,
        task: str | None = None,
        preview_image: bool = False,
        server_result_dir: str | None = None,
    ) -> list[TimedAction]:
        with self._session() as stub:
            return self._run_loaded_frame(
                stub=stub,
                episode_idx=episode_idx,
                frame_idx=frame_idx,
                task=task,
                preview_image=preview_image,
                server_result_dir=server_result_dir,
            )

    def replay_range(
        self,
        episode_idx: int,
        frame_start: int,
        frame_end: int | None,
        task: str | None = None,
        preview_image: bool = False,
        server_result_dir: str | None = None,
    ) -> list[list[TimedAction]]:
        if frame_end is None:
            frame_end = self.source.get_num_frames(episode_idx) - 1
        if frame_end < frame_start:
            raise ValueError(f"frame_end ({frame_end}) must be >= frame_start ({frame_start}).")
        self.logger.info(
            f"Replay range resolved to episode={episode_idx} frame_start={frame_start} frame_end={frame_end}"
        )

        all_actions: list[list[TimedAction]] = []
        with self._session() as stub:
            for frame_idx in range(frame_start, frame_end + 1):
                all_actions.append(
                    self._run_loaded_frame(
                        stub=stub,
                        episode_idx=episode_idx,
                        frame_idx=frame_idx,
                        task=task,
                        preview_image=preview_image,
                        server_result_dir=server_result_dir,
                    )
                )
        return all_actions

    def _run_loaded_frame(
        self,
        stub: services_pb2_grpc.AsyncInferenceStub,
        episode_idx: int,
        frame_idx: int,
        task: str | None,
        preview_image: bool,
        server_result_dir: str | None,
    ) -> list[TimedAction]:
        loaded = self.source.load_history(
            episode_idx=episode_idx,
            frame_idx=frame_idx,
            n_obs_steps=self._policy_config.n_obs_steps,
            task=task,
        )
        if preview_image:
            _show_preview(loaded.preview_image, episode_idx=episode_idx, frame_idx=frame_idx)

        request_id = f"pose-act-ep{episode_idx:04d}-fr{frame_idx:06d}-{uuid4().hex[:8]}"
        loaded.raw_observation["request_id"] = request_id
        if server_result_dir:
            result_dir = Path(server_result_dir) / request_id
            result_dir.mkdir(parents=True, exist_ok=True)
            _save_observation_images(loaded.raw_observation, result_dir)
        loaded.raw_observation["mode"] = "replay"
        loaded.raw_observation["episode_idx"] = episode_idx
        loaded.raw_observation["frame_idx"] = frame_idx

        actions = self._request_actions(
            stub=stub,
            timed_observation=TimedObservation(
                timestamp=time.time(),
                timestep=frame_idx,
                observation=loaded.raw_observation,
                must_go=True,
            )
        )
        _validate_pose7d_chunk(actions)

        if self.consumer is not None:
            self.consumer.consume(
                actions,
                {
                    "request_id": request_id,
                    "episode_idx": episode_idx,
                    "frame_idx": frame_idx,
                    "source_timestep": frame_idx,
                },
            )

        first_pose = actions[0].action.tolist()
        last_pose = actions[-1].action.tolist()
        self.logger.info(
            f"request_id={request_id} episode={episode_idx} frame={frame_idx} -> chunk_steps={len(actions)} "
            f"first_pose7d={first_pose} last_pose7d={last_pose}"
        )
        if server_result_dir:
            result_path = Path(server_result_dir) / request_id / "result.pt"
            self.logger.info(f"Expected dumped result path: {result_path}")
        return actions

    def _request_actions(
        self,
        stub: services_pb2_grpc.AsyncInferenceStub,
        timed_observation: TimedObservation,
    ) -> list[TimedAction]:
        try:
            payload = pickle.dumps(timed_observation)
            stub.SendObservations(
                send_bytes_in_chunks(
                    payload,
                    services_pb2.Observation,
                    log_prefix="[POSE_ACT_REPLAY_CLIENT]",
                    silent=True,
                )
            )
            response = stub.GetActions(services_pb2.Empty())
        except grpc.RpcError as exc:
            raise RuntimeError(f"Failed to query pose_act replay server at {self.server_address}: {exc}") from exc

        if not response.data:
            raise RuntimeError(
                f"pose_act replay server at {self.server_address} returned an empty action chunk for "
                f"frame={timed_observation.timestep}."
            )
        actions = pickle.loads(response.data)  # nosec
        if not isinstance(actions, list) or not actions:
            raise RuntimeError("pose_act server returned an invalid or empty TimedAction list.")
        return actions


def _to_float_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32).cpu()
    return torch.as_tensor(value, dtype=torch.float32)


def _to_image_hwc_uint8(value: object) -> torch.Tensor:
    image = torch.as_tensor(value)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {tuple(image.shape)}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = image.permute(1, 2, 0)
    if image.shape[-1] not in (1, 3):
        raise ValueError(f"Expected HWC image with 1 or 3 channels, got {tuple(image.shape)}")
    if image.dtype != torch.uint8:
        image = image.to(torch.float32)
        if float(image.max()) <= 1.0:
            image = image * 255.0
        image = image.clamp(0, 255).round().to(torch.uint8)
    return image.cpu().contiguous()


def _validate_pose7d_chunk(actions: list[TimedAction]) -> None:
    for index, action in enumerate(actions):
        tensor = action.action
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Action #{index} is not a torch.Tensor: {type(tensor)}")
        if tensor.ndim != 1 or tensor.shape[0] != len(POSE7D_NAMES):
            raise ValueError(
                f"Expected pose7d action with shape ({len(POSE7D_NAMES)},), got {tuple(tensor.shape)} "
                f"at chunk index {index}."
            )


def _save_observation_images(raw_observation: dict[str, object], output_dir: Path) -> None:
    for key, value in raw_observation.items():
        if not key.startswith(OBS_IMAGES):
            continue
        image_history = torch.as_tensor(value)
        if image_history.ndim == 3:
            image_history = image_history.unsqueeze(0)
        if image_history.ndim != 4:
            continue
        for idx, image in enumerate(image_history):
            image_hwc = _to_image_hwc_uint8(image)
            stem = key.removeprefix(f"{OBS_IMAGES}.").replace(".", "_")
            Image.fromarray(image_hwc.numpy()).save(output_dir / f"{stem}_obs_{idx:02d}.png")


def _show_preview(image: torch.Tensor, episode_idx: int, frame_idx: int) -> None:
    try:
        import matplotlib.pyplot as plt

        plt.figure("pose_act observation preview")
        plt.imshow(image.numpy())
        plt.title(f"episode={episode_idx} frame={frame_idx}")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.001)
    except Exception:
        # Preview is a best-effort convenience and should not block inference.
        return


def _parse_host_port(value: str) -> tuple[str, int]:
    host, port_str = value.rsplit(":", maxsplit=1)
    return host, int(port_str)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline pose_act replay client for visualization.")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo id.")
    parser.add_argument("--dataset-root", default=None, help="Optional local dataset root.")
    parser.add_argument("--episode", type=int, required=True, help="Episode index inside the dataset.")
    parser.add_argument("--frame", type=int, default=None, help="Single frame index inside the selected episode.")
    parser.add_argument("--frame-start", type=int, default=None, help="Start frame for sequential replay.")
    parser.add_argument(
        "--frame-end",
        type=int,
        default=None,
        help="End frame for sequential replay. If omitted, replay runs until the end of the episode.",
    )
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="pose_act gRPC server address.")
    parser.add_argument("--pretrained-name-or-path", required=True, help="Checkpoint path passed to the server.")
    parser.add_argument("--policy-device", default="cpu", help="Policy device used by the server.")
    parser.add_argument("--actions-per-chunk", type=int, default=16, help="Requested action chunk length.")
    parser.add_argument("--camera-key", default="fisheye_rgb", help="Camera key without observation.images prefix.")
    parser.add_argument("--task", default=None, help="Optional task override sent to the server.")
    parser.add_argument(
        "--bridge-address",
        default=None,
        help="Optional local socket bridge address host:port for forwarding returned chunks to IsaacLab.",
    )
    parser.add_argument(
        "--server-result-dir",
        default=None,
        help="Optional server-side result dump directory. If provided, the client prints the expected result.pt path.",
    )
    parser.add_argument("--preview", action="store_true", help="Enable observation image preview.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    consumer = None
    if args.bridge_address:
        bridge_host, bridge_port = _parse_host_port(args.bridge_address)
        consumer = SocketActionChunkConsumer(bridge_host, bridge_port)

    source = LeRobotReplaySource(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        camera_key=args.camera_key,
    )
    client = PoseActReplayClient(
        source=source,
        server_address=args.server_address,
        pretrained_name_or_path=args.pretrained_name_or_path,
        policy_device=args.policy_device,
        actions_per_chunk=args.actions_per_chunk,
        consumer=consumer,
    )
    if args.frame is not None:
        client.run_once(
            episode_idx=args.episode,
            frame_idx=args.frame,
            task=args.task,
            preview_image=args.preview,
            server_result_dir=args.server_result_dir,
        )
        return

    if args.frame_start is None:
        raise ValueError("Either --frame or --frame-start must be provided.")

    client.replay_range(
        episode_idx=args.episode,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        task=args.task,
        preview_image=args.preview,
        server_result_dir=args.server_result_dir,
    )


if __name__ == "__main__":
    main()
