# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import math
import pickle  # nosec
import struct
import subprocess
import sys
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import numpy as np
import torch

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pose_act.configuration_pose_act import (
    POSE_ACT_DEPTH_CAMERA_RGB_KEY,
    POSE_ACT_DEPTH_KEY,
    POSE_ACT_FISHEYE_RGB_KEY,
)
from lerobot.policies.pose_act.utils import pose7d_to_pose10d, pose10d_to_pose7d
from lerobot.processor import PolicyProcessorPipeline
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.types import PolicyAction
from lerobot.utils.constants import OBS_STATE

from .async_diagnostics import AsyncDiagnosticsWriter, chunk_intra_diff_stats
from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    prepare_image,
    raw_observation_to_observation,
    resize_robot_observation_image,
)
from .tensorrt import PoseACTTensorRTPolicyAdapter


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)
    _POSE_ACT_VIS_WINDOW = "pose_act_observation"
    _POSE_POLICY_TYPES = {"pose_act", "pose_smolvla"}

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()
        self._result_dump_root = config.result_dump_path
        if self._result_dump_root is not None:
            self._result_dump_root.mkdir(parents=True, exist_ok=True)

        self._diagnostics = AsyncDiagnosticsWriter(config.effective_diagnostics_dump_dir)

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()
        self._pose_act_vis_lock = threading.Lock()
        self._pose_act_vis_frame_queue: Queue[np.ndarray | None] = Queue(maxsize=1)
        self._pose_act_vis_process: subprocess.Popen[bytes] | None = None
        self._pose_act_vis_enabled = bool(config.pose_act_visualize_observation)
        self._pose_act_vis_warned = False
        self._pose_act_vis_warned_no_image_history = False

        self.last_processed_obs = None

        # State for pose_act shell motion generators (reset on each new client session).
        self._shell_step_count: int = 0
        self._shell_t0: float | None = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

        if self._pose_act_vis_enabled:
            self._start_pose_act_visualizer()

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

        self._shell_step_count = 0
        self._shell_t0 = None

    def _start_pose_act_visualizer(self) -> None:
        if self._pose_act_vis_process is not None:
            return
        try:
            self._pose_act_vis_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "lerobot.async_inference.pose_act_visualizer_stream",
                    self._POSE_ACT_VIS_WINDOW,
                ],
                stdin=subprocess.PIPE,
            )
            sender_thread = threading.Thread(
                target=self._pose_act_visualizer_sender_main,
                name="pose-act-visualizer-sender",
                daemon=True,
            )
            sender_thread.start()
        except Exception as e:
            self._pose_act_vis_process = None
            self._disable_pose_act_visualization(f"failed to start visualization process: {e}")

    def _pose_act_visualizer_sender_main(self) -> None:
        while self.running and self._pose_act_vis_process is not None:
            try:
                frame = self._pose_act_vis_frame_queue.get(timeout=0.1)
            except Empty:
                continue
            if frame is None:
                break
            process = self._pose_act_vis_process
            if process.stdin is None:
                self._disable_pose_act_visualization("visualization process stdin is unavailable")
                return
            try:
                payload = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)  # nosec
                process.stdin.write(struct.pack("!I", len(payload)))
                process.stdin.write(payload)
                process.stdin.flush()
            except Exception as e:
                self._disable_pose_act_visualization(f"failed to send frame to visualization process: {e}")
                return

    def _disable_pose_act_visualization(self, reason: str) -> None:
        if not self._pose_act_vis_warned:
            self.logger.warning(f"Disabling pose_act observation visualization: {reason}")
            self._pose_act_vis_warned = True
        self._pose_act_vis_enabled = False

    def _publish_pose_act_visualization(self, timed_observation: TimedObservation) -> None:
        if not self._pose_act_vis_enabled:
            return
        if self._pose_act_vis_process is not None and self._pose_act_vis_process.poll() is not None:
            self._disable_pose_act_visualization(
                "visualization process exited unexpectedly "
                f"(exitcode={self._pose_act_vis_process.returncode})"
            )
            return
        try:
            frame = self._build_pose_act_visualization_frame(timed_observation.get_observation())
        except Exception as e:
            self._disable_pose_act_visualization(f"failed to prepare frame: {e}")
            return
        if frame is None:
            if not self._pose_act_vis_warned_no_image_history:
                image_keys = [
                    key
                    for key in timed_observation.get_observation()
                    if key.startswith("observation.images.")
                ]
                self.logger.warning(
                    "PoseACT visualization skipped: no two-frame image history found in observation. "
                    f"image_keys={image_keys}"
                )
                self._pose_act_vis_warned_no_image_history = True
            return
        self._enqueue_pose_act_visualization_frame(frame)

    def _enqueue_pose_act_visualization_frame(self, frame: np.ndarray) -> None:
        try:
            with self._pose_act_vis_lock:
                if self._pose_act_vis_frame_queue.full():
                    try:
                        _ = self._pose_act_vis_frame_queue.get_nowait()
                    except Empty:
                        pass
                self._pose_act_vis_frame_queue.put_nowait(frame)
        except Exception as e:
            self._disable_pose_act_visualization(f"failed to enqueue visualization frame: {e}")

    def _build_pose_act_visualization_frame(self, raw_observation: dict[str, Any]) -> np.ndarray | None:
        histories: list[tuple[str, np.ndarray]] = []
        for key in (POSE_ACT_FISHEYE_RGB_KEY, POSE_ACT_DEPTH_CAMERA_RGB_KEY, POSE_ACT_DEPTH_KEY):
            value = raw_observation.get(key)
            if value is None:
                continue
            array = np.asarray(value)
            if array.ndim >= 3 and array.shape[0] >= 2:
                histories.append((key, array))

        if not histories:
            for key, value in raw_observation.items():
                if key == OBS_STATE or not key.startswith("observation.images."):
                    continue
                array = np.asarray(value)
                if array.ndim == 4 and array.shape[0] >= 2:
                    histories.append((key, array))
                    break

        if not histories:
            return None

        import cv2

        rows = []
        for key, history in histories:
            frames = [
                self._to_bgr_visualization_frame(history[0], is_depth=key == POSE_ACT_DEPTH_KEY),
                self._to_bgr_visualization_frame(history[1], is_depth=key == POSE_ACT_DEPTH_KEY),
            ]
            height = max(frame.shape[0] for frame in frames)
            padded = [self._pad_visualization_frame(frame, height) for frame in frames]
            row = np.concatenate(padded, axis=1)
            labels = ("t-1", "t0")
            x_offset = 0
            for label, frame in zip(labels, padded, strict=True):
                cv2.putText(
                    row,
                    label,
                    (x_offset + 12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                x_offset += frame.shape[1]
            cv2.putText(
                row,
                key,
                (12, max(48, height - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            rows.append(row)

        width = max(row.shape[1] for row in rows)
        rows = [self._pad_visualization_frame_width(row, width) for row in rows]
        canvas = np.concatenate(rows, axis=0)

        return canvas

    def _to_bgr_visualization_frame(self, frame: Any, is_depth: bool = False) -> np.ndarray:
        image = np.asarray(frame)
        if is_depth:
            return self._depth_to_bgr_visualization_frame(image)
        if image.ndim != 3:
            raise ValueError(f"expected image ndim=3, got shape {tuple(image.shape)}")
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[-1] != 3:
            raise ValueError(f"expected image channels=1 or 3, got shape {tuple(image.shape)}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image[..., ::-1])

    def _depth_to_bgr_visualization_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        if frame.ndim != 2:
            raise ValueError(
                f"expected depth ndim=2 or single-channel ndim=3, got shape {tuple(frame.shape)}"
            )

        finite = np.asarray(frame, dtype=np.float32)
        valid = finite > 0
        preview = np.zeros(finite.shape, dtype=np.uint8)
        if np.any(valid):
            near = float(np.percentile(finite[valid], 1))
            far = float(np.percentile(finite[valid], 99))
            if far <= near:
                far = near + 1.0
            preview = np.clip((finite - near) * 255.0 / (far - near), 0, 255).astype(np.uint8)

        import cv2

        return cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)

    def _pad_visualization_frame(self, frame: np.ndarray, target_height: int) -> np.ndarray:
        if frame.shape[0] == target_height:
            return frame
        pad = target_height - frame.shape[0]
        return np.pad(frame, ((0, pad), (0, 0), (0, 0)), mode="constant", constant_values=0)

    def _pad_visualization_frame_width(self, frame: np.ndarray, target_width: int) -> np.ndarray:
        if frame.shape[1] == target_width:
            return frame
        pad = target_width - frame.shape[1]
        return np.pad(frame, ((0, 0), (0, pad), (0, 0)), mode="constant", constant_values=0)

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        if self._use_dummy_policy():
            self.logger.warning(
                f"Running in dummy_policy shell mode (policy_type={self.policy_type}). "
                f"Skipping checkpoint load; will return zero action chunks of dim "
                f"{self._dummy_action_dim()}."
            )
            self.policy = None
            self.preprocessor = None
            self.postprocessor = None
            return services_pb2.Empty()

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)
        self.policy.config.device = self.device

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        if self.config.inference_backend == "tensorrt":
            if getattr(self.policy.config, "use_rgbd_inputs", False):
                raise ValueError("TensorRT backend does not support pose_act RGBD inputs yet.")
            engine_path = Path(policy_specs.pretrained_name_or_path) / "model.engine"
            self.policy = PoseACTTensorRTPolicyAdapter(
                self.policy,
                engine_path,
                build_engine=self.config.tensorrt_build_engine,
                fp16=self.config.tensorrt_fp16,
            )

        end = time.perf_counter()

        self.logger.info(
            f"Time taken to put policy on {self.device} "
            f"(backend={self.config.inference_backend}): {end - start:.4f} seconds"
        )

        return services_pb2.Empty()

    def _use_dummy_policy(self) -> bool:
        """True if we should skip loading a real policy and return fake action chunks.

        - ``dummy_policy=True`` unconditionally activates it.
        - ``pose_act_shell=True`` activates it only for pose TCP policies.
        """
        if self.config.dummy_policy:
            return True
        return self.config.pose_act_shell and self._is_pose_policy()

    def _is_pose_policy(self) -> bool:
        return self.policy_type in self._POSE_POLICY_TYPES

    def _dummy_action_dim(self) -> int:
        if self._is_pose_policy():
            return 10
        return int(self.config.dummy_action_dim)

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")
        if self._is_pose_policy():
            self._publish_pose_act_visualization(timed_observation)

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        request_id = timed_observation.get_observation().get("async_loop_request_id")
        queue_was_full = self.observation_queue.full()
        enqueued = self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        )
        if self._diagnostics.enabled:
            self._diagnostics.write_async_loop_event(
                "server_observation_received",
                request_id=request_id,
                receive_wallclock=float(receive_time),
                observation_timestep=int(obs_timestep),
                client_send_wallclock=float(obs_timestamp),
                client_to_server_ms=float((receive_time - obs_timestamp) * 1000),
                deserialize_ms=float(deserialize_time * 1000),
                enqueued=bool(enqueued),
                replaced_queued_observation=bool(queue_was_full and enqueued),
                queue_size_after=int(self.observation_queue.qsize()),
            )

        if not enqueued:
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            self._dump_pose_act_result(obs, action_chunk)

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            if self._diagnostics.enabled and action_chunk:
                raw_observation = obs.get_observation()
                self._diagnostics.write_async_loop_event(
                    "server_actions_ready",
                    request_id=raw_observation.get("async_loop_request_id"),
                    response_wallclock=float(time.time()),
                    observation_timestep=int(obs.get_timestep()),
                    server_processing_ms=float((time.perf_counter() - getactions_starts) * 1000),
                    model_path_ms=float(inference_time * 1000),
                    serialize_ms=float(serialize_time * 1000),
                    chunk_size=int(len(action_chunk)),
                    first_action_timestep=int(action_chunk[0].get_timestep()),
                    last_action_timestep=int(action_chunk[-1].get_timestep()),
                )

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False
        if self._is_pose_policy():
            return True

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(
        self,
        t_0: float,
        action_chunk: list[torch.Tensor],
        i_0: int,
        request_id: int | None = None,
    ) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(
                timestamp=t_0 + i * self.config.environment_dt,
                timestep=i_0 + i,
                action=action,
                request_id=request_id,
            )
            for i, action in enumerate(action_chunk)
        ]

    def _dump_pose_act_result(self, observation_t: TimedObservation, action_chunk: list[TimedAction]) -> Path | None:
        if not self._is_pose_policy() or self._result_dump_root is None or len(action_chunk) == 0:
            return None

        raw_observation = observation_t.get_observation()
        request_id = str(raw_observation.get("request_id") or f"ts-{observation_t.get_timestep():06d}")
        dump_dir = self._result_dump_root / request_id
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_path = dump_dir / "result.pt"

        action_tensor = torch.stack([step.get_action().detach().to(torch.float32).cpu() for step in action_chunk], dim=0)
        action_timestamps = torch.tensor([step.get_timestamp() for step in action_chunk], dtype=torch.float64)
        action_timesteps = torch.tensor([step.get_timestep() for step in action_chunk], dtype=torch.int64)

        state = raw_observation.get(OBS_STATE)
        if state is None:
            state = torch.tensor(
                [
                    raw_observation["x"],
                    raw_observation["y"],
                    raw_observation["z"],
                    raw_observation["roll"],
                    raw_observation["pitch"],
                    raw_observation["yaw"],
                    raw_observation["gripper_width"],
                ],
                dtype=torch.float32,
            ).unsqueeze(0)
        else:
            state = torch.as_tensor(state, dtype=torch.float32).cpu()

        image_key = None
        image = None
        for key, value in raw_observation.items():
            if key == OBS_STATE:
                continue
            if hasattr(value, "shape") and getattr(value, "ndim", 0) in (3, 4):
                image_key = key
                image = torch.as_tensor(value).cpu()
                break

        payload = {
            "request_id": request_id,
            "policy_type": self.policy_type,
            "policy_device": self.device,
            "actions": action_tensor,
            "action_timestamps": action_timestamps,
            "action_timesteps": action_timesteps,
            "observation_timestamp": observation_t.get_timestamp(),
            "observation_timestep": observation_t.get_timestep(),
            "observation_state": state,
            "observation_image_key": image_key,
            "observation_image": image,
            "task": raw_observation.get("task"),
            "metadata": {
                "actions_per_chunk": self.actions_per_chunk,
                "environment_dt": self.config.environment_dt,
                "result_path": str(dump_path),
            },
        }
        torch.save(payload, dump_path)
        self.logger.info(f"Dumped pose_act result to {dump_path}")
        return dump_path

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _is_pose_act_pose7d_observation(self, observation: Observation) -> bool:
        state = observation.get(OBS_STATE)
        return (
            self._is_pose_policy()
            and isinstance(state, torch.Tensor)
            and state.shape[-1] == 7
        )

    def _prepare_pose_act_observation(self, observation: Observation) -> Observation:
        """Convert client-supplied pose7d history into pose_act's batched pose10d history."""
        state7d = observation[OBS_STATE].to(torch.float32)
        if state7d.ndim == 2:
            state7d = state7d.unsqueeze(0)
        if state7d.ndim != 3 or state7d.shape[1] != self.policy.config.n_obs_steps or state7d.shape[2] != 7:
            raise ValueError(
                "pose_act async expects observation.state shape "
                f"(B,{self.policy.config.n_obs_steps},7), got {tuple(state7d.shape)}"
            )

        image_shapes = {}
        for key in self.policy.config.image_features:
            value = observation[key]
            image_shapes[key] = tuple(value.shape)
        self.logger.info(
            f"PoseACT input state7d={state7d.tolist()} image_shapes={image_shapes}"
        )

        prepared: Observation = {OBS_STATE: pose7d_to_pose10d(state7d)}
        for key in self.policy.config.image_features:
            image = observation[key]
            if image.ndim == 4:
                image = image.unsqueeze(0)
            if image.ndim != 5 or image.shape[1] != self.policy.config.n_obs_steps:
                raise ValueError(
                    f"pose_act async expects image {key} shape (B,{self.policy.config.n_obs_steps},C,H,W), "
                    f"got {tuple(image.shape)}"
                )
            prepared[key] = image.to(torch.float32)
        if "task" in observation:
            prepared["task"] = observation["task"]
        return prepared

    def _resize_pose_act_depth_history(
        self, depth_history: torch.Tensor, resize_shape: tuple[int, ...]
    ) -> torch.Tensor:
        if depth_history.ndim == 3:
            depth_history = depth_history.unsqueeze(-1)
        if depth_history.ndim != 4 or depth_history.shape[0] != self.policy.config.n_obs_steps:
            raise ValueError(
                f"pose_act async expects raw depth history {self.policy.config.depth_key} shape "
                f"({self.policy.config.n_obs_steps},H,W) or ({self.policy.config.n_obs_steps},H,W,1), "
                f"got {tuple(depth_history.shape)}"
            )
        if depth_history.shape[-1] != 1:
            raise ValueError(f"pose_act async expects single-channel depth, got {tuple(depth_history.shape)}")

        depth_chw = depth_history.permute(0, 3, 1, 2).to(torch.float32)
        target_hw = (resize_shape[1], resize_shape[2])
        return torch.nn.functional.interpolate(depth_chw, size=target_hw, mode="nearest")

    def _raw_pose_act_observation_to_observation(self, raw_observation: dict[str, Any]) -> Observation:
        observation: Observation = {
            OBS_STATE: torch.as_tensor(raw_observation[OBS_STATE], dtype=torch.float32)
        }
        for key in self.policy.config.image_features:
            image_history = torch.as_tensor(raw_observation[key])
            if getattr(self.policy.config, "use_rgbd_inputs", False) and key == self.policy.config.depth_key:
                resized = self._resize_pose_act_depth_history(
                    image_history, self.policy.config.image_features[key].shape
                )
            else:
                if image_history.ndim != 4 or image_history.shape[0] != self.policy.config.n_obs_steps:
                    raise ValueError(
                        f"pose_act async expects raw image history {key} shape "
                        f"({self.policy.config.n_obs_steps},H,W,C), got {tuple(image_history.shape)}"
                    )
                resized = torch.stack(
                    [
                        prepare_image(
                            resize_robot_observation_image(
                                image_history[t], self.policy.config.image_features[key].shape
                            )
                        )
                        for t in range(self.policy.config.n_obs_steps)
                    ],
                    dim=0,
                )
            observation[key] = resized
        if "task" in raw_observation:
            observation["task"] = raw_observation["task"]
        return observation

    def _is_umi_pi05_native_observation(self, raw_observation: dict[str, Any]) -> bool:
        return self.policy_type == "umi_pi05" and OBS_STATE in raw_observation

    def _raw_umi_pi05_observation_to_observation(self, raw_observation: dict[str, Any]) -> Observation:
        observation: Observation = {OBS_STATE: torch.as_tensor(raw_observation[OBS_STATE], dtype=torch.float32)}
        for key, feature in self.policy.config.image_features.items():
            image = torch.as_tensor(raw_observation[key])
            if image.ndim != 3:
                raise ValueError(f"umi_pi05 async expects raw image {key} shape (H,W,C), got {tuple(image.shape)}")
            observation[key] = prepare_image(resize_robot_observation_image(image, feature.shape))
        if "task" in raw_observation:
            observation["task"] = raw_observation["task"]
        return observation

    def _predict_pose_act_pose7d_chunk(
        self, observation_t: TimedObservation, observation: Observation, start_prepare: float, prepare_time: float
    ) -> list[TimedAction]:
        observation = self._prepare_pose_act_observation(observation)

        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference

        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape
        processed_actions = []
        for i in range(chunk_size):
            processed_actions.append(self.postprocessor(action_tensor[:, i, :]))
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0).detach().cpu()
        postprocessing_time = time.perf_counter() - start_postprocess

        start_pose_convert = time.perf_counter()
        action_tensor = pose10d_to_pose7d(action_tensor)
        pose_convert_time = time.perf_counter() - start_pose_convert

        if self._diagnostics.enabled:
            chunk_np = action_tensor.detach().cpu().numpy()
            obs_step = int(observation_t.get_timestep())
            self._diagnostics.write_chunk(
                obs_step=obs_step,
                first_action_step=obs_step + 1,
                obs_timestamp=float(observation_t.get_timestamp()),
                chunk=chunk_np,
                extra={"safe_return": bool(self.config.pose_act_safe_return_current_pose)},
            )
            stats = chunk_intra_diff_stats(chunk_np)
            self.logger.info(
                f"PoseACT chunk diag obs_step={obs_step} "
                f"xyz_step_max={stats['xyz_step_max']:.4f} "
                f"xyz_step_rms={stats['xyz_step_rms']:.4f} "
                f"rpy_step_max={stats['rpy_step_max']:.4f} "
                f"rpy_step_rms={stats['rpy_step_rms']:.4f} "
                f"gripper_step_max={stats['gripper_step_max']:.4f}"
            )

        if self.config.pose_act_safe_return_current_pose:
            action_tensor = self._build_pose_act_safe_return_pose7d_chunk(
                observation_t, action_tensor.shape[0]
            )
            self.logger.warning(
                "PoseACT safe return enabled: real inference/postprocess completed, "
                f"but returned actions are overwritten with {self.config.pose_act_safe_return_motion} pose7d."
            )

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(),
            list(action_tensor),
            observation_t.get_timestep(),
            request_id=observation_t.get_observation().get("async_loop_request_id"),
        )
        total_time = time.perf_counter() - start_prepare

        self.logger.info(
            f"PoseACT observation {observation_t.get_timestep()} | "
            f"prepare={prepare_time * 1000:.2f}ms "
            f"preprocess={preprocessing_time * 1000:.2f}ms inference={inference_time * 1000:.2f}ms "
            f"postprocess={postprocessing_time * 1000:.2f}ms pose_convert={pose_convert_time * 1000:.2f}ms "
            f"total={total_time * 1000:.2f}ms action_shape={tuple(action_tensor.shape)}"
        )
        if self._diagnostics.enabled:
            self._diagnostics.write_async_loop_event(
                "server_pose_act_timing",
                request_id=observation_t.get_observation().get("async_loop_request_id"),
                observation_timestep=int(observation_t.get_timestep()),
                prepare_ms=float(prepare_time * 1000),
                preprocess_ms=float(preprocessing_time * 1000),
                inference_ms=float(inference_time * 1000),
                postprocess_ms=float(postprocessing_time * 1000),
                pose_convert_ms=float(pose_convert_time * 1000),
                total_ms=float(total_time * 1000),
                chunk_size=int(action_tensor.shape[0]),
            )
        return action_chunk

    def _current_pose7d_from_raw_observation(self, observation_t: TimedObservation) -> torch.Tensor:
        raw_obs = observation_t.get_observation()
        state = raw_obs.get(OBS_STATE)
        if state is not None:
            state = torch.as_tensor(state, dtype=torch.float32)
            if state.ndim >= 2:
                state = state[-1]
            if state.shape != (7,):
                raise ValueError(f"Expected pose_act raw observation.state to end with shape (7,), got {tuple(state.shape)}")
            return state
        return torch.tensor(
            [
                raw_obs["x"],
                raw_obs["y"],
                raw_obs["z"],
                raw_obs["roll"],
                raw_obs["pitch"],
                raw_obs["yaw"],
                raw_obs["gripper_width"],
            ],
            dtype=torch.float32,
        )

    def _build_pose_act_safe_return_pose7d_chunk(
        self, observation_t: TimedObservation, chunk_size: int
    ) -> torch.Tensor:
        current_pose7d = self._current_pose7d_from_raw_observation(observation_t)
        if self.config.pose_act_safe_return_motion == "hold":
            return current_pose7d.unsqueeze(0).repeat(chunk_size, 1)

        if self.config.pose_act_safe_return_motion != "small_x_osc_gripper":
            self.logger.warning(
                f"Unknown pose_act_safe_return_motion='{self.config.pose_act_safe_return_motion}', "
                "falling back to hold."
            )
            return current_pose7d.unsqueeze(0).repeat(chunk_size, 1)

        # Absolute pose7d targets. With the client's latest_only aggregation, the first action
        # usually executes, so every chunk element carries the same small +x offset.
        chunk = current_pose7d.unsqueeze(0).repeat(chunk_size, 1)
        chunk[:, 0] = current_pose7d[0] + 0.003

        if self._shell_t0 is None:
            self._shell_t0 = observation_t.get_timestamp()
        base_t = observation_t.get_timestamp() - self._shell_t0
        for i in range(chunk_size):
            t = base_t + i * self.config.environment_dt
            chunk[i, 6] = max(0.0, 0.02 + 0.01 * math.sin(2.0 * math.pi * 0.5 * t))
        return chunk

    def _pose_act_shell_predict(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Shell path for pose_act / dummy modes.

        - Validates & logs the raw observation (state shape, camera keys, image dtypes/shapes).
        - Returns a chunk of actions with shape (actions_per_chunk, action_dim) generated by the
          motion mode selected via ``PolicyServerConfig.pose_act_shell_motion``.

        For ``pose_act``, ``action_dim=10`` with semantics pose10d = pos(3) + rot6d(6) + gripper(1),
        expressed **relative to the client's current TCP** (gripper is treated as absolute by the
        client adapter). Zero rot6d is NOT a valid rotation; the client adapter substitutes an
        identity 6D when it sees a zero rot6d.
        """
        obs = observation_t.get_observation()
        action_dim = self._dummy_action_dim()

        state = obs.get("observation.state")
        cam_keys = [
            k for k, v in obs.items()
            if hasattr(v, "shape") and getattr(v, "ndim", 0) == 3 and k != "observation.state"
        ]
        state_shape = tuple(state.shape) if hasattr(state, "shape") else None
        img_shapes = {k: tuple(obs[k].shape) for k in cam_keys}
        motion_mode = self.config.pose_act_shell_motion
        self.logger.info(
            f"[shell-{self.policy_type}] step={observation_t.get_timestep()} "
            f"state_shape={state_shape} cams={cam_keys} img_shapes={img_shapes} "
            f"-> chunk(shape=({self.actions_per_chunk},{action_dim}), motion={motion_mode})"
        )

        if self._is_pose_policy() and state is not None and hasattr(state, "shape"):
            last_dim = int(state.shape[-1])
            if last_dim != 10:
                self.logger.warning(
                    f"[shell-pose_act] observation.state last dim={last_dim} != 10 (pose10d expected)"
                )

        action_tensor = self._build_shell_chunk(motion_mode, action_dim, observation_t)
        self.last_processed_obs = observation_t
        return self._time_action_chunk(
            observation_t.get_timestamp(),
            list(action_tensor),
            observation_t.get_timestep(),
            request_id=observation_t.get_observation().get("async_loop_request_id"),
        )

    def _build_shell_chunk(
        self, motion_mode: str, action_dim: int, observation_t: TimedObservation
    ) -> torch.Tensor:
        """Dispatch on the configured shell motion mode to synthesise a fake action chunk."""
        if motion_mode == "zero" or not self._is_pose_policy() or action_dim != 10:
            return torch.zeros(self.actions_per_chunk, action_dim, dtype=torch.float32)
        if motion_mode == "drift_stop_osc":
            return self._gen_drift_stop_osc_chunk(observation_t)
        self.logger.warning(
            f"[shell-{self.policy_type}] unknown pose_act_shell_motion='{motion_mode}', falling back to zero"
        )
        return torch.zeros(self.actions_per_chunk, action_dim, dtype=torch.float32)

    def _gen_drift_stop_osc_chunk(self, observation_t: TimedObservation) -> torch.Tensor:
        """Relative pose10d chunk: +x drift for ~1s then stop, with gripper sine oscillation.

        Continuity:
        - Every chunk element carries the same small ``DX_PER_STEP`` relative translation so that
          under the ``latest_only`` aggregator the client sees a steady 1mm/control-step advance
          regardless of which chunk is active.
        - The gripper sinusoid is evaluated against absolute wall-clock time (``t - t0``), so
          successive chunks produce a globally smooth absolute gripper trajectory (no jumps when
          the client swaps chunks).
        """
        # NOTE: Piper MOVEP accumulates motion only if the new target is noticeably different
        # from the previous one; 1mm/step at 30Hz turned out to be below the visible threshold
        # (gripper worked, arm didn't). 3mm/step keeps us well under MAX_REL=0.01 and produces
        # ~9cm of total drift in 1s, which is clearly visible while still safe.
        DX_PER_STEP = 0.003  # meters per control step along +x
        DRIFT_STEPS = 30  # ~1s at 30fps before stopping
        GRIP_CENTER = 0.02  # 2cm absolute gripper opening
        GRIP_AMP = 0.015  # ±1.5cm absolute
        GRIP_FREQ = 0.5  # Hz

        chunk = torch.zeros(self.actions_per_chunk, 10, dtype=torch.float32)
        rel_x = DX_PER_STEP if self._shell_step_count < DRIFT_STEPS else 0.0
        chunk[:, 0] = rel_x
        chunk[:, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # rot6d identity

        if self._shell_t0 is None:
            self._shell_t0 = observation_t.get_timestamp()
        dt = self.config.environment_dt
        base_t = observation_t.get_timestamp() - self._shell_t0
        for i in range(self.actions_per_chunk):
            chunk[i, 9] = GRIP_CENTER + GRIP_AMP * math.sin(
                2.0 * math.pi * GRIP_FREQ * (base_t + i * dt)
            )

        self._shell_step_count += 1
        return chunk

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        if self._use_dummy_policy():
            return self._pose_act_shell_predict(observation_t)

        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        raw_observation = observation_t.get_observation()
        if self._is_pose_policy():
            observation = self._raw_pose_act_observation_to_observation(raw_observation)
        elif self._is_umi_pi05_native_observation(raw_observation):
            observation = self._raw_umi_pi05_observation_to_observation(raw_observation)
        else:
            observation = raw_observation_to_observation(
                observation_t.get_observation(),
                self.lerobot_features,
                self.policy_image_features,
            )
        prepare_time = time.perf_counter() - start_prepare

        if self._is_pose_act_pose7d_observation(observation):
            return self._predict_pose_act_pose7d_chunk(
                observation_t, observation, start_prepare, prepare_time
            )

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(),
            list(action_tensor),
            observation_t.get_timestep(),
            request_id=observation_t.get_observation().get("async_loop_request_id"),
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        try:
            if self._pose_act_vis_frame_queue.full():
                try:
                    _ = self._pose_act_vis_frame_queue.get_nowait()
                except Empty:
                    pass
            self._pose_act_vis_frame_queue.put_nowait(None)
        except Exception:
            pass
        if self._pose_act_vis_process is not None:
            try:
                if self._pose_act_vis_process.stdin is not None:
                    self._pose_act_vis_process.stdin.close()
            except Exception:
                pass
            self._pose_act_vis_process.wait(timeout=1.0)
            if self._pose_act_vis_process.poll() is None:
                self._pose_act_vis_process.terminate()
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    try:
        server.wait_for_termination()
    finally:
        policy_server.stop()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
