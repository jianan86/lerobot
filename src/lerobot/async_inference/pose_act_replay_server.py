from __future__ import annotations

import argparse
import logging
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Full, Queue

import grpc
import torch

from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.utils.constants import OBS_STATE

from .configs import PolicyServerConfig
from .policy_server import PolicyServer


class PoseActReplayServer(PolicyServer):
    """Replay-only inference server with FIFO observation handling.

    This server intentionally does not change the semantics of the default
    async inference server used by real robots.
    """

    def __init__(self, config: PolicyServerConfig, max_pending_observations: int = 8):
        self._replay_max_pending_observations = max_pending_observations
        super().__init__(config)
        self.observation_queue = Queue(maxsize=self._replay_max_pending_observations)

    def _reset_server(self) -> None:
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=self._replay_max_pending_observations)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

        self._shell_step_count = 0
        self._shell_t0 = None

    def _enqueue_observation(self, obs) -> bool:
        self.logger.debug(
            "Replay enqueue observation. frame=%s queue_size=%s/%s",
            obs.get_timestep(),
            self.observation_queue.qsize(),
            self._replay_max_pending_observations,
        )
        self.observation_queue.put(obs, timeout=self.config.obs_queue_timeout)
        return True

    def SendObservations(self, request_iterator, context):  # noqa: N802
        try:
            return super().SendObservations(request_iterator, context)
        except Full:
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "Replay observation queue is full; slow down the client or raise the queue limit.",
            )

    def _dump_pose_act_result(self, observation_t, action_chunk):
        if self.policy_type != "pose_act" or self._result_dump_root is None or len(action_chunk) == 0:
            return None

        raw_observation = observation_t.get_observation()
        request_id = str(raw_observation.get("request_id") or f"ts-{observation_t.get_timestep():06d}")
        dump_dir = self._result_dump_root / request_id
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_path = dump_dir / "result.pt"

        action_tensor = torch.stack(
            [step.get_action().detach().to(torch.float32).cpu() for step in action_chunk], dim=0
        )
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
            "mode": "replay",
            "episode_idx": raw_observation.get("episode_idx"),
            "frame_idx": raw_observation.get("frame_idx"),
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
                "episode_idx": raw_observation.get("episode_idx"),
                "frame_idx": raw_observation.get("frame_idx"),
                "mode": "replay",
            },
        }
        torch.save(payload, dump_path)
        self.logger.info("Dumped replay pose_act result to %s", dump_path)
        return dump_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay-only pose_act inference server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the replay server.")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind the replay server.")
    parser.add_argument("--fps", type=int, default=30, help="Target playback FPS used for action timestamps.")
    parser.add_argument("--inference-latency", type=float, default=0.0, help="Optional artificial latency in seconds.")
    parser.add_argument("--obs-queue-timeout", type=float, default=10.0, help="Queue put/get timeout in seconds.")
    parser.add_argument(
        "--result-dump-dir",
        default=None,
        help="Optional directory for replay result dumps consumed by IsaacLab or debugging tools.",
    )
    parser.add_argument(
        "--max-pending-observations",
        type=int,
        default=8,
        help="Maximum number of replay observations buffered before backpressure applies.",
    )
    parser.add_argument(
        "--pose-act-safe-return-current-pose",
        action="store_true",
        default=False,
        help="Run real inference and postprocess, then overwrite returned pose7d with the current observation pose.",
    )
    parser.add_argument(
        "--pose-act-safe-return-motion",
        default="hold",
        help="Safe-return motion mode when --pose-act-safe-return-current-pose is enabled.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_latency=args.inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
        result_dump_dir=args.result_dump_dir,
        pose_act_safe_return_current_pose=args.pose_act_safe_return_current_pose,
        pose_act_safe_return_motion=args.pose_act_safe_return_motion,
    )
    logging.info(pformat(asdict(cfg)))
    replay_server = PoseActReplayServer(cfg, max_pending_observations=args.max_pending_observations)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(replay_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    replay_server.logger.info("PoseActReplayServer started on %s:%s", cfg.host, cfg.port)
    server.start()
    server.wait_for_termination()
    replay_server.logger.info("Replay server terminated")


if __name__ == "__main__":
    main()
