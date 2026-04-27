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
"""Unit-tests for the `PolicyServer` core logic.
Monkey-patch the `policy` attribute with a stub so that no real model inference is performed.
"""

from __future__ import annotations

import time

import pytest
import torch

from lerobot.configs.types import PolicyFeature
from lerobot.utils.constants import OBS_STATE
from tests.utils import skip_if_package_missing

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


class MockPolicy:
    """A minimal mock for an actual policy, returning zeros.
    Refer to tests/policies for tests of the individual policies supported."""

    class _Config:
        robot_type = "dummy_robot"

        @property
        def image_features(self) -> dict[str, PolicyFeature]:
            """Empty image features since this test doesn't use images."""
            return {}

    def predict_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk of 20 dummy actions."""
        batch_size = len(observation[OBS_STATE])
        return torch.zeros(batch_size, 20, 6)

    def __init__(self):
        self.config = self._Config()

    def to(self, *args, **kwargs):
        # The server calls `policy.to(device)`. This stub ignores it.
        return self

    def model(self, batch: dict) -> torch.Tensor:
        # Return a chunk of 20 dummy actions.
        batch_size = len(batch["robot_type"])
        return torch.zeros(batch_size, 20, 6)


@pytest.fixture
@skip_if_package_missing("grpcio", "grpc")
def policy_server():
    """Fresh `PolicyServer` instance with a stubbed-out policy model."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer

    test_config = PolicyServerConfig(host="localhost", port=9999)
    server = PolicyServer(test_config)
    # Replace the real policy with our fast, deterministic stub.
    server.policy = MockPolicy()
    server.actions_per_chunk = 20
    server.device = "cpu"

    # Add mock lerobot_features that the observation similarity functions need
    server.lerobot_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        }
    }

    return server


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_obs(state: torch.Tensor, timestep: int = 0, must_go: bool = False):
    """Create a TimedObservation with a given state vector."""
    # Import only when needed
    from lerobot.async_inference.helpers import TimedObservation

    return TimedObservation(
        observation={
            "joint1": state[0].item() if len(state) > 0 else 0.0,
            "joint2": state[1].item() if len(state) > 1 else 0.0,
            "joint3": state[2].item() if len(state) > 2 else 0.0,
            "joint4": state[3].item() if len(state) > 3 else 0.0,
            "joint5": state[4].item() if len(state) > 4 else 0.0,
            "joint6": state[5].item() if len(state) > 5 else 0.0,
        },
        timestamp=time.time(),
        timestep=timestep,
        must_go=must_go,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_time_action_chunk(policy_server):
    """Verify that `_time_action_chunk` assigns correct timestamps and timesteps."""
    start_ts = time.time()
    start_t = 10
    # A chunk of 3 action tensors.
    action_tensors = [torch.randn(6) for _ in range(3)]

    timed_actions = policy_server._time_action_chunk(start_ts, action_tensors, start_t)

    assert len(timed_actions) == 3
    # Check timesteps
    assert [ta.get_timestep() for ta in timed_actions] == [10, 11, 12]
    # Check timestamps
    expected_timestamps = [
        start_ts,
        start_ts + policy_server.config.environment_dt,
        start_ts + 2 * policy_server.config.environment_dt,
    ]
    for ta, expected_ts in zip(timed_actions, expected_timestamps, strict=True):
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_maybe_enqueue_observation_must_go(policy_server):
    """An observation with `must_go=True` is always enqueued."""
    obs = _make_obs(torch.zeros(6), must_go=True)
    assert policy_server._enqueue_observation(obs) is True
    assert policy_server.observation_queue.qsize() == 1
    assert policy_server.observation_queue.get_nowait() is obs


def test_maybe_enqueue_observation_dissimilar(policy_server):
    """A dissimilar observation (not `must_go`) is enqueued."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, dissimilar observation.
    new_obs = _make_obs(torch.ones(6) * 5)  # High norm difference

    assert policy_server._enqueue_observation(new_obs) is True
    assert policy_server.observation_queue.qsize() == 1


def test_maybe_enqueue_observation_is_skipped(policy_server):
    """A similar observation (not `must_go`) is skipped."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, very similar observation.
    new_obs = _make_obs(torch.zeros(6) + 1e-4)

    assert policy_server._enqueue_observation(new_obs) is False
    assert policy_server.observation_queue.empty() is True


def test_obs_sanity_checks(policy_server):
    """Unit-test the private `_obs_sanity_checks` helper."""
    prev = _make_obs(torch.zeros(6), timestep=0)

    # Case 1 – timestep already predicted
    policy_server._predicted_timesteps.add(1)
    obs_same_ts = _make_obs(torch.ones(6), timestep=1)
    assert policy_server._obs_sanity_checks(obs_same_ts, prev) is False

    # Case 2 – observation too similar
    policy_server._predicted_timesteps.clear()
    obs_similar = _make_obs(torch.zeros(6) + 1e-4, timestep=2)
    assert policy_server._obs_sanity_checks(obs_similar, prev) is False

    # Case 3 – genuinely new & dissimilar observation passes
    obs_ok = _make_obs(torch.ones(6) * 5, timestep=3)
    assert policy_server._obs_sanity_checks(obs_ok, prev) is True


def test_predict_action_chunk(monkeypatch, policy_server):
    """End-to-end test of `_predict_action_chunk` with a stubbed _get_action_chunk."""
    # Import only when needed
    from lerobot.async_inference.policy_server import PolicyServer

    # Force server to act-style policy; patch method to return deterministic tensor
    policy_server.policy_type = "act"
    # NOTE(Steven): Smelly tests as the Server is a state machine being partially mocked. Adding these processors as a quick fix.
    policy_server.preprocessor = lambda obs: obs
    policy_server.postprocessor = lambda tensor: tensor
    action_dim = 6
    batch_size = 1
    actions_per_chunk = policy_server.actions_per_chunk

    def _fake_get_action_chunk(_self, _obs, _type="act"):
        return torch.zeros(batch_size, actions_per_chunk, action_dim)

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)

    obs = _make_obs(torch.zeros(6), timestep=5)
    timed_actions = policy_server._predict_action_chunk(obs)

    assert len(timed_actions) == actions_per_chunk
    assert [ta.get_timestep() for ta in timed_actions] == list(range(5, 5 + actions_per_chunk))

    for i, ta in enumerate(timed_actions):
        expected_ts = obs.get_timestamp() + i * policy_server.config.environment_dt
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_predict_pose_act_pose7d_chunk(monkeypatch):
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.helpers import TimedObservation
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pose_act.configuration_pose_act import PoseACTConfig
    from lerobot.policies.pose_act.utils import pose7d_to_pose10d
    from lerobot.utils.constants import ACTION

    config = PoseACTConfig(
        device="cpu",
        use_vae=False,
        input_features={
            "observation.images.fisheye_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )

    class PoseACTStub:
        def __init__(self, cfg):
            self.config = cfg

    server = PolicyServer(PolicyServerConfig(host="localhost", port=9998))
    server.policy_type = "pose_act"
    server.policy = PoseACTStub(config)
    server.actions_per_chunk = 3
    server.device = "cpu"
    server.lerobot_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_width"],
        },
        "observation.images.fisheye_rgb": {
            "dtype": "image",
            "shape": (32, 32, 3),
            "names": ["height", "width", "channels"],
        },
    }
    server.preprocessor, server.postprocessor = make_pre_post_processors(config, dataset_stats=None)

    def _fake_get_action_chunk(_self, _obs):
        base = pose7d_to_pose10d(torch.tensor([[0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 0.04]]))
        return base[:, None, :].repeat(1, server.actions_per_chunk, 1)

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)

    obs = TimedObservation(
        observation={
            OBS_STATE: torch.tensor(
                [[0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 0.04], [0.21, 0.0, 0.31, 0.0, 0.0, 0.0, 0.04]],
                dtype=torch.float32,
            ),
            "observation.images.fisheye_rgb": torch.zeros(2, 32, 32, 3, dtype=torch.uint8).numpy(),
        },
        timestamp=time.time(),
        timestep=7,
        must_go=True,
    )

    timed_actions = server._predict_action_chunk(obs)
    assert len(timed_actions) == server.actions_per_chunk
    assert timed_actions[0].get_action().shape == (7,)
    torch.testing.assert_close(
        timed_actions[0].get_action(),
        torch.tensor([0.41, 0.0, 0.61, 0.0, -0.0, 0.0, 0.04]),
        rtol=0,
        atol=1e-5,
    )


def test_prepare_pose_act_observation_uses_client_history(monkeypatch):
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pose_act.configuration_pose_act import PoseACTConfig

    config = PoseACTConfig(
        device="cpu",
        use_vae=False,
        input_features={
            "observation.images.fisheye_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )

    class PoseACTStub:
        def __init__(self, cfg):
            self.config = cfg

    server = PolicyServer(PolicyServerConfig(host="localhost", port=9998))
    server.policy_type = "pose_act"
    server.policy = PoseACTStub(config)

    observation = server._raw_pose_act_observation_to_observation(
        {
            OBS_STATE: torch.tensor(
                [[0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.01], [0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 0.02]],
                dtype=torch.float32,
            ),
            "observation.images.fisheye_rgb": torch.zeros(2, 20, 20, 3, dtype=torch.uint8),
        }
    )
    prepared = server._prepare_pose_act_observation(observation)

    assert tuple(prepared[OBS_STATE].shape) == (1, 2, 10)
    assert tuple(prepared["observation.images.fisheye_rgb"].shape) == (1, 2, 3, 16, 16)


def test_pose_act_result_dump(tmp_path):
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.helpers import TimedAction, TimedObservation
    from lerobot.async_inference.policy_server import PolicyServer

    server = PolicyServer(
        PolicyServerConfig(host="localhost", port=9998, result_dump_dir=str(tmp_path))
    )
    server.policy_type = "pose_act"
    server.device = "cpu"
    server.actions_per_chunk = 2

    observation = TimedObservation(
        observation={
            "request_id": "req-0001",
            OBS_STATE: torch.tensor(
                [[0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.01], [0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 0.02]],
                dtype=torch.float32,
            ),
            "observation.images.fisheye_rgb": torch.zeros(2, 20, 20, 3, dtype=torch.uint8).numpy(),
        },
        timestamp=time.time(),
        timestep=11,
        must_go=True,
    )
    chunk = [
        TimedAction(timestamp=observation.timestamp, timestep=11, action=torch.zeros(7, dtype=torch.float32)),
        TimedAction(timestamp=observation.timestamp + 0.1, timestep=12, action=torch.ones(7, dtype=torch.float32)),
    ]

    dump_path = server._dump_pose_act_result(observation, chunk)

    assert dump_path == tmp_path / "req-0001" / "result.pt"
    assert dump_path.is_file()
    payload = torch.load(dump_path, map_location="cpu")
    assert payload["request_id"] == "req-0001"
    assert tuple(payload["actions"].shape) == (2, 7)
    assert tuple(payload["observation_state"].shape) == (2, 7)
    assert payload["observation_image_key"] == "observation.images.fisheye_rgb"
