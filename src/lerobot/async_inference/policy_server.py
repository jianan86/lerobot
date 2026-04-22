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
import threading
import time
from collections import deque
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pose_act.utils import pose10d_to_pose7d, pose7d_to_pose10d
from lerobot.processor import PolicyProcessorPipeline
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.types import PolicyAction
from lerobot.utils.constants import OBS_STATE

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
    raw_observation_to_observation,
)


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # State for pose_act shell motion generators (reset on each new client session).
        self._shell_step_count: int = 0
        self._shell_t0: float | None = None
        self._pose_act_obs_history: dict[str, deque[torch.Tensor]] = {}

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

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
        self._pose_act_obs_history = {}

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

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def _use_dummy_policy(self) -> bool:
        """True if we should skip loading a real policy and return fake action chunks.

        - ``dummy_policy=True`` unconditionally activates it.
        - ``pose_act_shell=True`` activates it only for ``policy_type == "pose_act"``.
        """
        if self.config.dummy_policy:
            return True
        if self.config.pose_act_shell and self.policy_type == "pose_act":
            return True
        return False

    def _dummy_action_dim(self) -> int:
        if self.policy_type == "pose_act":
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

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
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

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _is_pose_act_pose7d_observation(self, observation: Observation) -> bool:
        state = observation.get(OBS_STATE)
        return (
            self.policy_type == "pose_act"
            and isinstance(state, torch.Tensor)
            and state.shape[-1] == 7
        )

    def _append_pose_act_history(self, key: str, value: torch.Tensor) -> None:
        horizon = int(self.policy.config.n_obs_steps)
        queue = self._pose_act_obs_history.setdefault(key, deque(maxlen=horizon))
        value = value.detach().clone()
        if len(queue) == 0:
            queue.extend(value.clone() for _ in range(horizon))
        else:
            queue.append(value)

    def _prepare_pose_act_observation(self, observation: Observation) -> Observation:
        """Convert base-frame pose7d observation to pose_act's batched pose10d history."""
        state7d = observation[OBS_STATE]
        if state7d.ndim == 2:
            state7d = state7d.squeeze(0)
        if state7d.ndim != 1 or state7d.shape[0] != 7:
            raise ValueError(f"pose_act async expects observation.state shape (7,), got {tuple(state7d.shape)}")

        image_shapes = {}
        for key in self.policy.config.image_features:
            value = observation[key]
            image_shapes[key] = tuple(value.shape)
        self.logger.info(
            f"PoseACT input state7d={state7d.tolist()} image_shapes={image_shapes}"
        )

        state10d = pose7d_to_pose10d(state7d.to(torch.float32).cpu())
        self._append_pose_act_history(OBS_STATE, state10d)

        for key in self.policy.config.image_features:
            image = observation[key]
            if image.ndim == 4:
                image = image.squeeze(0)
            if image.ndim != 3:
                raise ValueError(f"pose_act async expects image {key} shape (C,H,W), got {tuple(image.shape)}")
            self._append_pose_act_history(key, image.to(torch.float32).cpu())

        prepared: Observation = {
            OBS_STATE: torch.stack(list(self._pose_act_obs_history[OBS_STATE]), dim=0).unsqueeze(0)
        }
        for key in self.policy.config.image_features:
            prepared[key] = torch.stack(list(self._pose_act_obs_history[key]), dim=0).unsqueeze(0)
        if "task" in observation:
            prepared["task"] = observation["task"]
        return prepared

    def _predict_pose_act_pose7d_chunk(
        self, observation_t: TimedObservation, observation: Observation, start_prepare: float, prepare_time: float
    ) -> list[TimedAction]:
        start_history = time.perf_counter()
        observation = self._prepare_pose_act_observation(observation)
        history_time = time.perf_counter() - start_history

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

        if self.config.pose_act_safe_return_current_pose:
            state7d = observation_t.get_observation()
            current_pose7d = torch.tensor(
                [
                    state7d["x"],
                    state7d["y"],
                    state7d["z"],
                    state7d["roll"],
                    state7d["pitch"],
                    state7d["yaw"],
                    state7d["gripper_width"],
                ],
                dtype=torch.float32,
            )
            action_tensor = current_pose7d.unsqueeze(0).repeat(action_tensor.shape[0], 1)
            self.logger.warning(
                "PoseACT safe return enabled: real inference/postprocess completed, "
                "but returned actions are overwritten with current pose7d."
            )

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        total_time = time.perf_counter() - start_prepare

        self.logger.info(
            f"PoseACT observation {observation_t.get_timestep()} | "
            f"prepare={prepare_time * 1000:.2f}ms history={history_time * 1000:.2f}ms "
            f"preprocess={preprocessing_time * 1000:.2f}ms inference={inference_time * 1000:.2f}ms "
            f"postprocess={postprocessing_time * 1000:.2f}ms pose_convert={pose_convert_time * 1000:.2f}ms "
            f"total={total_time * 1000:.2f}ms action_shape={tuple(action_tensor.shape)}"
        )
        return action_chunk

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

        if self.policy_type == "pose_act" and state is not None and hasattr(state, "shape"):
            last_dim = int(state.shape[-1])
            if last_dim != 10:
                self.logger.warning(
                    f"[shell-pose_act] observation.state last dim={last_dim} != 10 (pose10d expected)"
                )

        action_tensor = self._build_shell_chunk(motion_mode, action_dim, observation_t)
        self.last_processed_obs = observation_t
        return self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )

    def _build_shell_chunk(
        self, motion_mode: str, action_dim: int, observation_t: TimedObservation
    ) -> torch.Tensor:
        """Dispatch on the configured shell motion mode to synthesise a fake action chunk."""
        if motion_mode == "zero" or self.policy_type != "pose_act" or action_dim != 10:
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
        observation: Observation = raw_observation_to_observation(
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
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
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

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
