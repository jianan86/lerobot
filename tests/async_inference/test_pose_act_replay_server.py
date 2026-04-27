from __future__ import annotations

import time

import pytest
import torch

pytest.importorskip("grpc")

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import TimedAction, TimedObservation
from lerobot.async_inference.pose_act_replay_server import PoseActReplayServer
from lerobot.utils.constants import OBS_STATE


def _make_replay_obs(timestep: int) -> TimedObservation:
    return TimedObservation(
        observation={
            "mode": "replay",
            "request_id": f"req-{timestep}",
            "episode_idx": 1,
            "frame_idx": timestep,
            OBS_STATE: torch.zeros(2, 7, dtype=torch.float32),
            "observation.images.fisheye_rgb": torch.zeros(2, 8, 8, 3, dtype=torch.uint8).numpy(),
        },
        timestamp=time.time(),
        timestep=timestep,
        must_go=True,
    )


def test_pose_act_replay_server_enqueue_preserves_fifo():
    server = PoseActReplayServer(PolicyServerConfig(host="localhost", port=9988), max_pending_observations=2)
    obs_a = _make_replay_obs(3)
    obs_b = _make_replay_obs(4)

    assert server._enqueue_observation(obs_a) is True
    assert server._enqueue_observation(obs_b) is True
    assert server.observation_queue.qsize() == 2
    assert server.observation_queue.get_nowait() is obs_a
    assert server.observation_queue.get_nowait() is obs_b


def test_pose_act_replay_server_result_dump_adds_replay_metadata(tmp_path):
    server = PoseActReplayServer(
        PolicyServerConfig(host="localhost", port=9988, result_dump_dir=str(tmp_path)),
        max_pending_observations=2,
    )
    server.policy_type = "pose_act"
    server.device = "cpu"
    server.actions_per_chunk = 2

    observation = _make_replay_obs(7)
    chunk = [
        TimedAction(timestamp=observation.timestamp, timestep=7, action=torch.zeros(7, dtype=torch.float32)),
        TimedAction(timestamp=observation.timestamp + 0.1, timestep=8, action=torch.ones(7, dtype=torch.float32)),
    ]

    dump_path = server._dump_pose_act_result(observation, chunk)

    assert dump_path == tmp_path / "req-7" / "result.pt"
    payload = torch.load(dump_path, map_location="cpu")
    assert payload["mode"] == "replay"
    assert payload["episode_idx"] == 1
    assert payload["frame_idx"] == 7
    assert payload["metadata"]["mode"] == "replay"
