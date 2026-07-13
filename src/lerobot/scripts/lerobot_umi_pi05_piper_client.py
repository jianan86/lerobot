#!/usr/bin/env python

import json
import pickle  # nosec
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import grpc
import numpy as np
import torch
from PIL import Image

from lerobot.async_inference.adapters import (
    UmiPI05PiperAdapter,
    accelerate_gripper_closure,
    build_relative_state,
)
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.robots import (
    RobotConfig,
    make_robot_from_config,
    piper_follower,  # noqa: F401
)
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class UmiPI05PiperClientConfig:
    right_robot: RobotConfig
    pretrained_name_or_path: str
    instruction: str
    left_robot: RobotConfig | None = None
    device: str = "cuda"
    actions_per_chunk: int = 10
    fps: float = 10.0
    server_address: str = "127.0.0.1:8080"
    no_tunnel: bool = False
    ssh_host: str = "jianan@183.230.224.121"
    ssh_port: int = 50210
    remote_grpc_port: int = 8080
    arm_mode: str = "dual"
    right_pika_gripper_port: str | None = None
    left_pika_gripper_port: str | None = None
    right_pika_fisheye_device: int | str | None = None
    left_pika_fisheye_device: int | str | None = None
    pika_device_summary_path: str | None = "/home/kw/workspace/device_group/summary.json"
    pika_camera_width: int = 640
    pika_camera_height: int = 480
    pika_camera_fps: int = 30
    image_size: int = 224
    pika_gripper_min_width_m: float = 0.0
    pika_gripper_max_width_m: float = 0.085
    max_xyz_step: float = 0.03
    max_rpy_step: float = 0.35
    max_gripper_step: float = 0.005
    gripper_close_threshold: float = 0.04
    dry_run_actions: bool = False
    max_steps: int | None = None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_tunnel(cfg: UmiPI05PiperClientConfig) -> tuple[str, subprocess.Popen[bytes] | None]:
    if cfg.no_tunnel:
        return cfg.server_address, None
    local_port = _free_port()
    proc = subprocess.Popen(  # noqa: S603
        [
            "ssh",
            "-N",
            "-L",
            f"127.0.0.1:{local_port}:127.0.0.1:{cfg.remote_grpc_port}",
            "-p",
            str(cfg.ssh_port),
            cfg.ssh_host,
        ]
    )
    address = f"127.0.0.1:{local_port}"
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"SSH tunnel exited early with code {proc.returncode}.")
        try:
            with socket.create_connection(("127.0.0.1", local_port), timeout=0.2):
                return address, proc
        except OSError:
            time.sleep(0.05)
    proc.terminate()
    raise TimeoutError(f"Timed out waiting for SSH tunnel on {address}.")


@dataclass(frozen=True)
class PikaDeviceBinding:
    group_name: str
    physical_device_id: str
    gripper_port: str
    fisheye_device: str


def load_pika_device_bindings(summary_path: str | Path) -> dict[str, PikaDeviceBinding]:
    path = Path(summary_path).expanduser()
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"device group summary not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid device group summary JSON: {path}") from exc

    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError(f"device group summary must contain a groups list: {path}")

    by_name = {group.get("device_name"): group for group in groups if isinstance(group, dict)}
    return {
        "right": _parse_pika_binding(by_name, "right_gripper"),
        "left": _parse_pika_binding(by_name, "left_gripper"),
    }


def _parse_pika_binding(groups: dict[str, dict], group_name: str) -> PikaDeviceBinding:
    group = groups.get(group_name)
    if group is None:
        raise RuntimeError(f"device group summary missing group: {group_name}")
    if group.get("status") != "ok":
        raise RuntimeError(f"device group {group_name} status is not ok: {group.get('status')}")

    usb_port = group.get("usb_port")
    fisheye = group.get("fisheye")
    if not isinstance(usb_port, dict):
        raise ValueError(f"device group {group_name} missing usb_port object")
    if not isinstance(fisheye, dict):
        raise ValueError(f"device group {group_name} missing fisheye object")

    gripper_port = usb_port.get("devnode")
    fisheye_device = fisheye.get("devnode")
    if not gripper_port:
        raise ValueError(f"device group {group_name} missing usb_port.devnode")
    if not fisheye_device:
        raise ValueError(f"device group {group_name} missing fisheye.devnode")

    return PikaDeviceBinding(
        group_name=group_name,
        physical_device_id=str(group.get("physical_device_id", "")),
        gripper_port=str(gripper_port),
        fisheye_device=str(fisheye_device),
    )


def _print_pika_binding(side: str, binding: PikaDeviceBinding) -> None:
    print(
        f"[device-bind] {side} group={binding.group_name} "
        f"physical_device_id={binding.physical_device_id} "
        f"gripper_port={binding.gripper_port} fisheye_device={binding.fisheye_device}",
        flush=True,
    )


def _resolve_pika_devices(
    cfg: UmiPI05PiperClientConfig,
) -> tuple[str, int | str, str | None, int | str | None]:
    need_left = cfg.arm_mode == "dual"
    needs_binding = (
        cfg.right_pika_gripper_port is None
        or cfg.right_pika_fisheye_device is None
        or (need_left and cfg.left_pika_gripper_port is None)
        or (need_left and cfg.left_pika_fisheye_device is None)
    )
    bindings = None
    if needs_binding:
        if cfg.pika_device_summary_path is None:
            raise ValueError(
                "Pika gripper ports/fisheye devices are incomplete and pika_device_summary_path is disabled."
            )
        bindings = load_pika_device_bindings(cfg.pika_device_summary_path)

    right_binding = bindings["right"] if bindings is not None else None
    left_binding = bindings["left"] if bindings is not None and need_left else None

    right_port = cfg.right_pika_gripper_port or (right_binding.gripper_port if right_binding else None)
    right_fisheye = cfg.right_pika_fisheye_device
    if right_fisheye is None and right_binding is not None:
        right_fisheye = right_binding.fisheye_device
    left_port = cfg.left_pika_gripper_port or (left_binding.gripper_port if left_binding else None)
    left_fisheye = cfg.left_pika_fisheye_device
    if left_fisheye is None and left_binding is not None:
        left_fisheye = left_binding.fisheye_device

    if right_port is None or right_fisheye is None:
        raise ValueError("right Pika gripper port and fisheye device are required.")
    if need_left and (left_port is None or left_fisheye is None):
        raise ValueError("left Pika gripper port and fisheye device are required when arm_mode='dual'.")

    if right_binding is not None:
        _print_pika_binding("right", right_binding)
    if left_binding is not None:
        _print_pika_binding("left", left_binding)

    return right_port, right_fisheye, left_port, left_fisheye


class UmiPI05PikaGripper:
    def __init__(
        self,
        port: str,
        *,
        fisheye_device: int | str,
        camera_width: int,
        camera_height: int,
        camera_fps: int,
        min_width_m: float,
        max_width_m: float,
    ) -> None:
        self.port = port
        self.fisheye_device = fisheye_device
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_fps = int(camera_fps)
        self.min_width_m = float(min_width_m)
        self.max_width_m = float(max_width_m)
        self.device: Any | None = None
        self.fisheye: Any | None = None

    def connect(self) -> None:
        try:
            import cv2

            if not hasattr(cv2, "setLogLevel"):
                cv2.setLogLevel = lambda *_args, **_kwargs: None
        except ImportError:
            pass

        try:
            from pika.gripper import Gripper
        except ImportError as e:
            raise ImportError(
                "Pika SDK is required for UMI PI0.5 Piper gripper/camera control. "
                "Install it in the client environment, for example: "
                "python -m pip install -e /path/to/pika_sdk --no-deps, "
                "and install pyserial."
            ) from e

        self.device = Gripper(self.port)
        if not self.device.connect():
            self.device = None
            raise RuntimeError(f"Failed to connect Pika gripper on {self.port}.")
        if not self.device.enable():
            self.disconnect()
            raise RuntimeError(f"Failed to enable Pika gripper on {self.port}.")

        self.device.set_camera_param(self.camera_width, self.camera_height, self.camera_fps)
        self.device.set_fisheye_camera_index(self.fisheye_device)
        self.fisheye = self.device.get_fisheye_camera()
        self._wait_for_frame()

    def _wait_for_frame(self) -> None:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self.fisheye is not None and getattr(self.fisheye, "is_connected", False):
                ok, frame = self.fisheye.get_frame()
                if ok and frame is not None and getattr(frame, "ndim", 0) == 3 and frame.shape[2] == 3:
                    return
            time.sleep(0.02)
        raise RuntimeError(f"Failed to read initial Pika fisheye frame on {self.port}.")

    def read_width(self) -> float:
        if self.device is None:
            raise RuntimeError("Pika gripper is not connected.")
        return max(float(self.device.get_gripper_distance()) / 1000.0, 0.0)

    def read_rgb(self) -> np.ndarray:
        if self.fisheye is None:
            raise RuntimeError("Pika fisheye camera is not connected.")
        ok, frame = self.fisheye.get_frame()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read Pika fisheye frame on {self.port}.")
        return np.ascontiguousarray(frame[..., ::-1])

    def execute_width(self, width_m: float) -> float:
        if self.device is None:
            raise RuntimeError("Pika gripper is not connected.")
        clipped_width_m = min(max(float(width_m), self.min_width_m), self.max_width_m)
        if not self.device.set_gripper_distance(clipped_width_m * 1000.0):
            raise RuntimeError(f"Failed to command Pika gripper width {clipped_width_m:.4f} m.")
        return clipped_width_m

    def disconnect(self) -> None:
        if self.device is None:
            return
        try:
            self.device.disconnect()
        finally:
            self.device = None
            self.fisheye = None


def _setup_gripper(
    port: str | None, cfg: UmiPI05PiperClientConfig, *, fisheye_device: int | str | None
) -> UmiPI05PikaGripper | None:
    if port is None:
        return None
    if fisheye_device is None:
        raise ValueError("A Pika fisheye device index is required when a Pika gripper port is configured.")
    gripper = UmiPI05PikaGripper(
        port,
        fisheye_device=fisheye_device,
        camera_width=cfg.pika_camera_width,
        camera_height=cfg.pika_camera_height,
        camera_fps=cfg.pika_camera_fps,
        min_width_m=cfg.pika_gripper_min_width_m,
        max_width_m=cfg.pika_gripper_max_width_m,
    )
    gripper.connect()
    return gripper


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = (255 * array).astype(np.uint8)
    return array


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    pil_image = Image.fromarray(_to_hwc_uint8(image))
    current_width, current_height = pil_image.size
    if current_width == width and current_height == height:
        return np.asarray(pil_image, dtype=np.uint8)

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    canvas.paste(resized, (pad_width, pad_height))
    return np.asarray(canvas, dtype=np.uint8)


def preprocess_fisheye(image: np.ndarray, size: int = 224) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    resized = _resize_with_pad(image, size, size)
    return np.ascontiguousarray(resized)


def _black_image_like(image: np.ndarray) -> np.ndarray:
    return np.zeros_like(image, dtype=np.uint8)


def _read_pika_image(gripper: UmiPI05PikaGripper | None, size: int) -> np.ndarray:
    if gripper is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return preprocess_fisheye(gripper.read_rgb(), size)


def _limit_pose7_target(
    current: torch.Tensor,
    target: torch.Tensor,
    *,
    max_xyz_step: float,
    max_rpy_step: float,
    max_gripper_step: float,
    gripper_min: float,
    gripper_max: float,
) -> torch.Tensor:
    limited = target.detach().to(torch.float32).cpu().clone()
    base = current.detach().to(torch.float32).cpu()
    if limited.shape != (7,) or base.shape != (7,):
        raise ValueError(f"Expected pose7 target/current shapes (7,), got {tuple(limited.shape)} and {tuple(base.shape)}")
    limited[:3] = base[:3] + (limited[:3] - base[:3]).clamp(-max_xyz_step, max_xyz_step)
    rpy_delta = (limited[3:6] - base[3:6] + torch.pi) % (2 * torch.pi) - torch.pi
    limited[3:6] = base[3:6] + rpy_delta.clamp(-max_rpy_step, max_rpy_step)
    limited[6] = base[6] + (limited[6] - base[6]).clamp(-max_gripper_step, max_gripper_step)
    limited[6] = limited[6].clamp(gripper_min, gripper_max)
    return limited


def _limit_bimanual_target(current: torch.Tensor, target: torch.Tensor, cfg: UmiPI05PiperClientConfig) -> torch.Tensor:
    limited = target.detach().to(torch.float32).cpu().clone()
    for offset in (0, 7):
        limited[offset : offset + 7] = _limit_pose7_target(
            current[offset : offset + 7],
            limited[offset : offset + 7],
            max_xyz_step=cfg.max_xyz_step,
            max_rpy_step=cfg.max_rpy_step,
            max_gripper_step=cfg.max_gripper_step,
            gripper_min=cfg.pika_gripper_min_width_m,
            gripper_max=cfg.pika_gripper_max_width_m,
        )
    return limited


def _send_observation(stub, observation: TimedObservation) -> None:
    payload = pickle.dumps(observation)  # nosec
    chunks = send_bytes_in_chunks(payload, services_pb2.Observation, log_prefix="[umi_pi05]")
    stub.SendObservations(chunks)


def _get_actions(stub) -> list:
    response = stub.GetActions(services_pb2.Empty())
    if not response.data:
        return []
    return pickle.loads(response.data)  # nosec


def _send_bimanual_target(
    adapter: UmiPI05PiperAdapter,
    target: torch.Tensor,
    cfg: UmiPI05PiperClientConfig,
    right_robot,
    left_robot,
    right_gripper,
    left_gripper,
) -> None:
    right_action, left_action = adapter.tcp_pose7_to_ee_actions(target)
    right_width = right_action.pop("gripper.pos", None)
    left_width = left_action.pop("gripper.pos", None)
    if cfg.dry_run_actions:
        return

    right_robot.send_action(right_action)
    if cfg.arm_mode == "dual":
        left_robot.send_action(left_action)
    if right_gripper is not None and right_width is not None:
        right_gripper.execute_width(right_width)
    if cfg.arm_mode == "dual" and left_gripper is not None and left_width is not None:
        left_gripper.execute_width(left_width)


def _execute_tcp_action_chunk(
    tcp_actions: torch.Tensor,
    last_target: torch.Tensor,
    cfg: UmiPI05PiperClientConfig,
    adapter: UmiPI05PiperAdapter,
    right_robot,
    left_robot,
    right_gripper,
    left_gripper,
    *,
    dt: float,
    executed_steps: int,
    sleep_fn=time.sleep,
    perf_counter=time.perf_counter,
) -> tuple[torch.Tensor, int]:
    for action_index, raw_target in enumerate(tcp_actions):
        if cfg.max_steps is not None and executed_steps >= cfg.max_steps:
            break

        action_start = perf_counter()
        target = _limit_bimanual_target(last_target, raw_target, cfg)
        _send_bimanual_target(
            adapter,
            target,
            cfg,
            right_robot,
            left_robot,
            right_gripper,
            left_gripper,
        )
        last_target = target
        executed_steps += 1

        has_next_action = action_index < len(tcp_actions) - 1
        below_step_limit = cfg.max_steps is None or executed_steps < cfg.max_steps
        if has_next_action and below_step_limit:
            sleep_fn(max(0.0, dt - (perf_counter() - action_start)))

    return last_target, executed_steps


@draccus.wrap()
def run(cfg: UmiPI05PiperClientConfig) -> None:
    init_logging()
    print(pformat(asdict(cfg)))

    if cfg.arm_mode not in {"dual", "single"}:
        raise ValueError(f"arm_mode must be 'dual' or 'single', got {cfg.arm_mode!r}")
    if cfg.arm_mode == "dual" and cfg.left_robot is None:
        raise ValueError("left_robot is required when arm_mode='dual'.")
    right_pika_port, right_fisheye, left_pika_port, left_fisheye = _resolve_pika_devices(cfg)

    address, tunnel = _start_tunnel(cfg)
    channel = grpc.insecure_channel(address)
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    right_robot = None
    left_robot = None
    right_gripper = None
    left_gripper = None

    try:
        stub.Ready(services_pb2.Empty())
        policy_config = RemotePolicyConfig(
            policy_type="umi_pi05",
            pretrained_name_or_path=cfg.pretrained_name_or_path,
            lerobot_features={},
            actions_per_chunk=cfg.actions_per_chunk,
            device=cfg.device,
        )
        stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy_config)))  # nosec

        right_robot = make_robot_from_config(cfg.right_robot)
        left_robot = make_robot_from_config(cfg.left_robot) if cfg.arm_mode == "dual" else right_robot
        right_gripper = _setup_gripper(right_pika_port, cfg, fisheye_device=right_fisheye)
        left_gripper = (
            _setup_gripper(left_pika_port, cfg, fisheye_device=left_fisheye)
            if cfg.arm_mode == "dual"
            else right_gripper
        )
        adapter = UmiPI05PiperAdapter(
            right_robot, left_robot, right_gripper=right_gripper, left_gripper=left_gripper
        )
        right_robot.connect()
        if cfg.arm_mode == "dual":
            left_robot.connect()

        timestep = 0
        dt = 1.0 / cfg.fps
        previous_tcp_pose = adapter.current_tcp_pose7_state()
        last_target = previous_tcp_pose.clone()
        executed_steps = 0
        while cfg.max_steps is None or executed_steps < cfg.max_steps:
            start = time.perf_counter()
            current_tcp_pose = adapter.current_tcp_pose7_state()
            state = build_relative_state(previous_tcp_pose, current_tcp_pose)
            right_image = _read_pika_image(right_gripper, cfg.image_size)
            left_image = (
                _read_pika_image(left_gripper, cfg.image_size) if cfg.arm_mode == "dual" else right_image.copy()
            )
            obs = {
                OBS_STATE: state.numpy(),
                f"{OBS_STATE}.absolute_tcp_pose7": current_tcp_pose.numpy(),
                f"{OBS_IMAGES}.cam_high": _black_image_like(left_image),
                f"{OBS_IMAGES}.cam_left_wrist": left_image,
                f"{OBS_IMAGES}.cam_right_wrist": right_image,
                "task": cfg.instruction,
                "async_loop_request_id": timestep,
            }
            _send_observation(stub, TimedObservation(time.time(), timestep, obs, must_go=True))
            timed_actions = _get_actions(stub)
            if timed_actions:
                tcp_actions = torch.stack([a.get_action().to(torch.float32).cpu() for a in timed_actions], dim=0)
                tcp_actions = accelerate_gripper_closure(
                    tcp_actions,
                    close_threshold=cfg.gripper_close_threshold,
                )
                last_target, executed_steps = _execute_tcp_action_chunk(
                    tcp_actions,
                    last_target,
                    cfg,
                    adapter,
                    right_robot,
                    left_robot,
                    right_gripper,
                    left_gripper,
                    dt=dt,
                    executed_steps=executed_steps,
                )

            previous_tcp_pose = current_tcp_pose
            timestep += 1
            if not timed_actions:
                time.sleep(max(0.0, dt - (time.perf_counter() - start)))
    finally:
        if right_robot is not None:
            right_robot.disconnect()
        if cfg.arm_mode == "dual" and left_robot is not None:
            left_robot.disconnect()
        if right_gripper is not None:
            right_gripper.disconnect()
        if cfg.arm_mode == "dual" and left_gripper is not None:
            left_gripper.disconnect()
        channel.close()
        if tunnel is not None:
            tunnel.terminate()


def main() -> None:
    register_third_party_plugins()
    run()


if __name__ == "__main__":
    main()
