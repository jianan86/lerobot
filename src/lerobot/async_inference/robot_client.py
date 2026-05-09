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
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import math
import pickle  # nosec
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.robots import piper_follower  # noqa: F401
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from .adapters import POSE7D_NAMES, PoseActPiperAdapter, is_pose_act_piper
from .configs import RobotClientConfig
from .async_diagnostics import AsyncDiagnosticsWriter
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        self._pose_act_adapter: PoseActPiperAdapter | None = None
        if is_pose_act_piper(config.policy_type, config.robot.type):
            self._pose_act_adapter = PoseActPiperAdapter(self.robot)
        self._pose_act_history: deque[dict[str, Any]] = deque(maxlen=2)

        if self._pose_act_adapter is not None:
            lerobot_features = self._pose_act_piper_lerobot_features()
        else:
            lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []

        self._diagnostics = AsyncDiagnosticsWriter(config.effective_diagnostics_dump_dir)
        self._async_loop_request_id = 0
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

    def _pose_act_piper_lerobot_features(self) -> dict[str, dict]:
        """Feature contract for pose_act: pose7d state plus the robot cameras."""
        features = {
            OBS_STATE: {
                "dtype": "float32",
                "shape": (len(POSE7D_NAMES),),
                "names": list(POSE7D_NAMES),
            }
        }
        for key, shape in self.robot.observation_features.items():
            if isinstance(shape, tuple):
                features[f"{OBS_IMAGES}.{key}"] = {
                    "dtype": "image",
                    "shape": shape,
                    "names": ["height", "width", "channels"],
                }
        return features

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        receive_time: float | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if self.config.aggregate_fn_name == "rtc_smooth":
            self._aggregate_action_queues_rtc_smooth(incoming_actions, receive_time=receive_time)
            return

        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        incoming_first_step = incoming_actions[0].get_timestep() if incoming_actions else -1
        incoming_last_step = incoming_actions[-1].get_timestep() if incoming_actions else -1
        fn_name = self.config.aggregate_fn_name

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            old_action = current_action_queue[new_action.get_timestep()]
            new_tensor = new_action.get_action()
            agg_tensor = aggregate_fn(old_action, new_tensor)
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=agg_tensor,
                    request_id=new_action.request_id,
                )
            )
            if self._diagnostics.enabled:
                try:
                    old_seq = old_action.detach().cpu().flatten().tolist()
                    new_seq = new_tensor.detach().cpu().flatten().tolist()
                    agg_seq = agg_tensor.detach().cpu().flatten().tolist()
                except Exception:
                    continue
                self._diagnostics.write_aggregate_event(
                    timestep=int(new_action.get_timestep()),
                    old=old_seq[:7],
                    new=new_seq[:7],
                    agg=agg_seq[:7],
                    fn_name=fn_name,
                    incoming_first_step=int(incoming_first_step),
                    incoming_last_step=int(incoming_last_step),
                )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def _rtc_smooth_old_weight(self, blend_index: int, blend_steps: int) -> float:
        if blend_steps <= 0:
            return 0.0

        progress = (blend_index + 1) / (blend_steps + 1)
        old_weight = 1.0 - progress
        if self.config.rtc_smooth_exp_schedule:
            old_weight = old_weight * math.expm1(old_weight) / (math.e - 1)
        return float(max(0.0, min(1.0, old_weight)))

    def _write_rtc_smooth_aggregate_event(
        self,
        old_action: torch.Tensor,
        new_action: torch.Tensor,
        agg_tensor: torch.Tensor,
        new_timed_action: TimedAction,
        incoming_first_step: int,
        incoming_last_step: int,
    ) -> None:
        if not self._diagnostics.enabled:
            return
        try:
            old_seq = old_action.detach().cpu().flatten().tolist()
            new_seq = new_action.detach().cpu().flatten().tolist()
            agg_seq = agg_tensor.detach().cpu().flatten().tolist()
        except Exception:
            return
        self._diagnostics.write_aggregate_event(
            timestep=int(new_timed_action.get_timestep()),
            old=old_seq[:7],
            new=new_seq[:7],
            agg=agg_seq[:7],
            fn_name=self.config.aggregate_fn_name,
            incoming_first_step=int(incoming_first_step),
            incoming_last_step=int(incoming_last_step),
        )

    def _aggregate_action_queues_rtc_smooth(
        self,
        incoming_actions: list[TimedAction],
        receive_time: float | None = None,
    ):
        """RTC-style client-side chunk merge.

        Keeps imminent old actions stable, blends a short old/new overlap, and
        lets farther-future actions come from the newest chunk.
        """
        with self.latest_action_lock:
            latest_action = self.latest_action
        with self.action_queue_lock:
            old_actions = list(self.action_queue.queue)

        old_live = [action for action in old_actions if action.get_timestep() > latest_action]
        incoming_live = [
            action
            for action in incoming_actions
            if action.get_timestep() > latest_action
            and (receive_time is None or action.get_timestamp() > receive_time)
        ]

        future_action_queue = Queue()
        if not incoming_live:
            for old_action in old_live:
                future_action_queue.put(old_action)
            with self.action_queue_lock:
                self.action_queue = future_action_queue
            return

        safe_prefix_steps = self.config.rtc_smooth_safe_prefix_steps
        blend_steps = self.config.rtc_smooth_blend_steps
        incoming_first_step = incoming_live[0].get_timestep()
        incoming_last_step = incoming_live[-1].get_timestep()

        old_by_timestep = {action.get_timestep(): action for action in old_live}
        incoming_by_timestep = {action.get_timestep(): action for action in incoming_live}
        safe_timesteps = {action.get_timestep() for action in old_live[:safe_prefix_steps]}

        output_by_timestep: dict[int, TimedAction] = {}
        for old_action in old_live[:safe_prefix_steps]:
            output_by_timestep[old_action.get_timestep()] = old_action

        blend_index = 0
        for new_action in incoming_live:
            timestep = new_action.get_timestep()
            if timestep in safe_timesteps:
                continue

            old_timed_action = old_by_timestep.get(timestep)
            if old_timed_action is None:
                output_by_timestep[timestep] = new_action
                continue

            if blend_index >= blend_steps:
                output_by_timestep[timestep] = new_action
                continue

            old_tensor = old_timed_action.get_action()
            new_tensor = new_action.get_action()
            old_weight = self._rtc_smooth_old_weight(blend_index, blend_steps)
            agg_tensor = old_weight * old_tensor + (1.0 - old_weight) * new_tensor
            output_by_timestep[timestep] = TimedAction(
                timestamp=new_action.get_timestamp(),
                timestep=timestep,
                action=agg_tensor,
                request_id=new_action.request_id,
            )
            self._write_rtc_smooth_aggregate_event(
                old_tensor,
                new_tensor,
                agg_tensor,
                new_action,
                incoming_first_step,
                incoming_last_step,
            )
            blend_index += 1

        for old_action in old_live:
            timestep = old_action.get_timestep()
            if timestep in output_by_timestep or timestep in incoming_by_timestep:
                continue
            if timestep < incoming_first_step:
                output_by_timestep[timestep] = old_action

        for timestep in sorted(output_by_timestep):
            future_action_queue.put(output_by_timestep[timestep])

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                with self.latest_action_lock:
                    latest_action_before_merge = self.latest_action
                old_size, old_timesteps = self._inspect_action_queue()
                old_timestep_set = set(old_timesteps)
                incoming_timesteps = [a.get_timestep() for a in timed_actions]

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    if not old_timesteps:
                        old_timesteps = [latest_action_before_merge]  # queue was empty

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action_before_merge} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn, receive_time=receive_time)
                queue_update_time = time.perf_counter() - start_time
                new_size, new_timesteps = self._inspect_action_queue()

                if self._diagnostics.enabled and timed_actions:
                    request_id = timed_actions[0].request_id
                    stale_actions = sum(1 for a in timed_actions if a.get_timestep() <= latest_action_before_merge)
                    overlap_actions = sum(1 for a in timed_actions if a.get_timestep() in old_timestep_set)
                    overlap_actions -= sum(
                        1
                        for a in timed_actions
                        if a.get_timestep() <= latest_action_before_merge and a.get_timestep() in old_timestep_set
                    )
                    new_actions = max(0, len(timed_actions) - stale_actions - overlap_actions)
                    self._diagnostics.write_async_loop_event(
                        "client_chunk_merge",
                        request_id=request_id,
                        receive_wallclock=float(receive_time),
                        latest_action_at_receive=int(latest_action_before_merge),
                        queue_size_before=int(old_size),
                        queue_size_after=int(new_size),
                        incoming_first_step=int(incoming_timesteps[0]),
                        incoming_last_step=int(incoming_timesteps[-1]),
                        stale_actions=int(stale_actions),
                        overlap_actions=int(overlap_actions),
                        new_actions=int(new_actions),
                        dropped_old_actions=int(max(0, old_size - overlap_actions)),
                        deserialize_ms=float(deserialize_time * 1000),
                        queue_update_ms=float(queue_update_time * 1000),
                    )

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        if self._pose_act_adapter is not None:
            return self._pose_act_adapter.convert(action_tensor)

        keys = list(self.robot.action_features)
        n = min(len(keys), int(action_tensor.shape[0]))
        return {keys[i]: action_tensor[i].item() for i in range(n)}

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        action_dict = self._action_tensor_to_action_dict(timed_action.get_action())
        _performed_action = self.robot.send_action(action_dict)
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if self._diagnostics.enabled:
            try:
                pre_pose7d = timed_action.get_action().detach().cpu().flatten().tolist()
            except Exception:
                pre_pose7d = None
            else:
                pre_pose7d = pre_pose7d[:7] if len(pre_pose7d) >= 7 else None

            post_pose7d: list[float] | None = None
            try:
                if {"ee.abs_x", "ee.abs_y", "ee.abs_z"}.issubset(action_dict):
                    post_pose7d = [
                        float(action_dict.get("ee.abs_x", 0.0)),
                        float(action_dict.get("ee.abs_y", 0.0)),
                        float(action_dict.get("ee.abs_z", 0.0)),
                        float(action_dict.get("ee.abs_rx", 0.0)),
                        float(action_dict.get("ee.abs_ry", 0.0)),
                        float(action_dict.get("ee.abs_rz", 0.0)),
                        float(action_dict.get("gripper.pos", 0.0)),
                    ]
            except Exception:
                post_pose7d = None

            self._diagnostics.write_executed(
                timestep=int(timed_action.get_timestep()),
                action_timestamp=float(timed_action.get_timestamp()),
                pose7d_pre_adapter=pre_pose7d,
                pose7d_post_adapter=post_pose7d,
            )
            self._diagnostics.write_async_loop_event(
                "client_action_executed",
                request_id=timed_action.request_id,
                timestep=int(timed_action.get_timestep()),
                queue_size_after_pop=int(self.action_queue.qsize()),
                latest_action=int(self.latest_action),
            )

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def _capture_pose_act_observation_frame(self, task: str) -> RawObservation:
        raw_observation: RawObservation = self.robot.get_observation()
        pose7d = self._pose_act_adapter.current_pose7d()

        frame: RawObservation = {
            OBS_STATE: pose7d.clone(),
            "task": task,
        }
        frame.update(
            {name: float(value) for name, value in zip(POSE7D_NAMES, pose7d.tolist(), strict=True)}
        )

        for key, shape in self.robot.observation_features.items():
            if isinstance(shape, tuple) and key in raw_observation:
                frame[f"{OBS_IMAGES}.{key}"] = raw_observation[key]

        return frame

    def _build_pose_act_history_observation(self, task: str) -> RawObservation:
        frame = self._capture_pose_act_observation_frame(task)
        self._pose_act_history.append(frame)
        if len(self._pose_act_history) == 1:
            self._pose_act_history.append(frame)

        history = list(self._pose_act_history)
        observation: RawObservation = {
            OBS_STATE: torch.stack(
                [torch.as_tensor(history_frame[OBS_STATE], dtype=torch.float32) for history_frame in history], dim=0
            ),
            "task": task,
        }
        observation.update(
            {name: history[-1][name] for name in POSE7D_NAMES}
        )

        for key in self.policy_config.lerobot_features:
            if key.startswith(f"{OBS_IMAGES}."):
                observation[key] = torch.stack(
                    [torch.as_tensor(history_frame[key]) for history_frame in history], dim=0
                )

        return observation

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            if self._pose_act_adapter is not None:
                raw_observation = self._build_pose_act_history_observation(task)
            else:
                raw_observation = self.robot.get_observation()
                raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()
            queue_ratio = current_queue_size / max(1, self.action_chunk_size)
            request_id = self._async_loop_request_id
            self._async_loop_request_id += 1
            raw_observation["async_loop_request_id"] = request_id

            _ = self.send_observation(observation)
            if self._diagnostics.enabled:
                self._diagnostics.write_async_loop_event(
                    "client_request_sent",
                    request_id=int(request_id),
                    send_wallclock=float(observation.get_timestamp()),
                    observation_timestep=int(observation.get_timestep()),
                    latest_action_at_send=int(latest_action),
                    queue_size_at_send=int(current_queue_size),
                    queue_ratio_at_send=float(queue_ratio),
                    action_chunk_size=int(self.action_chunk_size),
                    chunk_size_threshold=float(self._chunk_size_threshold),
                    must_go=bool(observation.must_go),
                    obs_capture_ms=float(obs_capture_time * 1000),
                    fps=float(self.config.fps),
                    actions_per_chunk=int(self.config.actions_per_chunk),
                )

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
            elif self._diagnostics.enabled:
                with self.latest_action_lock:
                    latest_action = self.latest_action
                self._diagnostics.write_async_loop_event(
                    "client_control_underrun",
                    latest_action=int(latest_action),
                    queue_size=0,
                )

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    # TODO: Assert if checking robot support is still needed with the plugin system
    # if cfg.robot.type not in SUPPORTED_ROBOTS:
    #     raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()  # run the client
