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
"""Unit-tests for the `RobotClient` action-queue logic (pure Python, no gRPC).

We monkey-patch `lerobot.robots.utils.make_robot_from_config` so that
no real hardware is accessed. Only the queue-update mechanism is verified.
"""

from __future__ import annotations

import time
from queue import Queue

import numpy as np
import pytest
import torch

# Skip entire module if required deps are not available
pytest.importorskip("grpc")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def robot_client():
    """Fresh `RobotClient` instance for each test case (no threads started).
    Uses DummyRobot."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from tests.mocks.mock_robot import MockRobotConfig

    test_config = MockRobotConfig()

    # gRPC channel is not actually used in tests, so using a dummy address
    test_config = RobotClientConfig(
        robot=test_config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
    )

    client = RobotClient(test_config)

    # Initialize attributes that are normally set in start() method
    client.chunks_received = 0
    client.available_actions_size = []

    yield client

    if client.robot.is_connected:
        client.stop()


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_actions(start_ts: float, start_t: int, count: int):
    """Generate `count` consecutive TimedAction objects starting at timestep `start_t`."""
    from lerobot.async_inference.helpers import TimedAction

    fps = 30  # emulates most common frame-rate
    actions = []
    for i in range(count):
        timestep = start_t + i
        timestamp = start_ts + i * (1 / fps)
        action_tensor = torch.full((6,), timestep, dtype=torch.float32)
        actions.append(TimedAction(action=action_tensor, timestep=timestep, timestamp=timestamp))
    return actions


def _make_constant_actions(start_ts: float, start_t: int, values: list[float]):
    from lerobot.async_inference.helpers import TimedAction

    return [
        TimedAction(
            action=torch.full((6,), value, dtype=torch.float32),
            timestep=start_t + i,
            timestamp=start_ts + i * (1 / 30),
        )
        for i, value in enumerate(values)
    ]


def _queue_timesteps_and_values(queue: Queue) -> tuple[list[int], list[float]]:
    actions = list(queue.queue)
    return [a.get_timestep() for a in actions], [float(a.get_action()[0].item()) for a in actions]


class _StubPoseActPiperRobot:
    def __init__(self, fps: int = 30):
        self.fps = fps
        self._connected = False
        self._rng = np.random.default_rng(0)
        self._joint_state = {
            "joint_1.pos": 0.0,
            "joint_2.pos": 0.0,
            "joint_3.pos": 0.0,
            "joint_4.pos": 0.0,
            "joint_5.pos": 0.0,
            "joint_6.pos": 0.0,
            "gripper.pos": 0.0,
        }
        self._ee_state = {"x": 0.2, "y": 0.0, "z": 0.3, "rx": 0.0, "ry": 0.0, "rz": 0.0}
        self.observation_features = {"front": (480, 640, 3)}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _get_end_pose(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._ee_state.items()}

    def _get_motor_positions(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._joint_state.items()}

    def get_observation(self) -> dict[str, object]:
        assert self._connected
        obs = {k: float(v) for k, v in self._joint_state.items()}
        obs["front"] = self._rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        return obs


def _make_pose_act_piper_client(monkeypatch, async_observation: bool = False):
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.robots.piper_follower import PiperFollowerConfig

    monkeypatch.setattr(
        "lerobot.async_inference.robot_client.make_robot_from_config",
        lambda config: _StubPoseActPiperRobot(),
    )

    return RobotClient(
        RobotClientConfig(
            robot=PiperFollowerConfig(enable_on_connect=False, disable_on_disconnect=False),
            server_address="localhost:9999",
            policy_type="pose_act",
            pretrained_name_or_path="test",
            actions_per_chunk=3,
            async_observation=async_observation,
        )
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_update_action_queue_discards_stale(robot_client):
    """`_update_action_queue` must drop actions with `timestep` <= `latest_action`."""

    # Pretend we already executed up to action #4
    robot_client.latest_action = 4

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    robot_client._aggregate_action_queues(incoming)

    # Extract timesteps from queue
    resulting_timesteps = [a.get_timestep() for a in robot_client.action_queue.queue]

    assert resulting_timesteps == [5, 6, 7]


@pytest.mark.parametrize(
    "weight_old, weight_new",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.1, 0.9),
        (0.9, 0.1),
    ],
)
def test_aggregate_action_queues_combines_actions_in_overlap(
    robot_client, weight_old: float, weight_new: float
):
    """`_aggregate_action_queues` must combine actions on overlapping timesteps according
    to the provided aggregate_fn, here tested with multiple coefficients."""
    from lerobot.async_inference.helpers import TimedAction

    robot_client.chunks_received = 0

    # Pretend we already executed up to action #4, and queue contains actions for timesteps 5..6
    robot_client.latest_action = 4
    current_actions = _make_actions(
        start_ts=time.time(), start_t=5, count=2
    )  # actions are [torch.ones(6), torch.ones(6), ...]
    current_actions = [
        TimedAction(action=10 * a.get_action(), timestep=a.get_timestep(), timestamp=a.get_timestamp())
        for a in current_actions
    ]

    for a in current_actions:
        robot_client.action_queue.put(a)

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    overlap_timesteps = [5, 6]  # properly tested in test_aggregate_action_queues_discards_stale
    nonoverlap_timesteps = [7]

    robot_client._aggregate_action_queues(
        incoming, aggregate_fn=lambda x1, x2: weight_old * x1 + weight_new * x2
    )

    queue_overlap_actions = []
    queue_non_overlap_actions = []
    for a in robot_client.action_queue.queue:
        if a.get_timestep() in overlap_timesteps:
            queue_overlap_actions.append(a)
        elif a.get_timestep() in nonoverlap_timesteps:
            queue_non_overlap_actions.append(a)

    queue_overlap_actions = sorted(queue_overlap_actions, key=lambda x: x.get_timestep())
    queue_non_overlap_actions = sorted(queue_non_overlap_actions, key=lambda x: x.get_timestep())

    assert torch.allclose(
        queue_overlap_actions[0].get_action(),
        weight_old * current_actions[0].get_action() + weight_new * incoming[-3].get_action(),
    )
    assert torch.allclose(
        queue_overlap_actions[1].get_action(),
        weight_old * current_actions[1].get_action() + weight_new * incoming[-2].get_action(),
    )
    assert torch.allclose(queue_non_overlap_actions[0].get_action(), incoming[-1].get_action())


def test_robot_client_config_accepts_rtc_smooth_aggregate_fn():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    cfg = RobotClientConfig(
        robot=MockRobotConfig(),
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        aggregate_fn_name="rtc_smooth",
    )

    assert cfg.aggregate_fn is None


def test_robot_client_config_to_dict_includes_async_observation():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    cfg = RobotClientConfig(
        robot=MockRobotConfig(),
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        async_observation=True,
    )

    assert cfg.to_dict()["async_observation"] is True


def test_robot_client_config_rejects_invalid_aggregate_fn():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    with pytest.raises(ValueError, match="Unknown aggregate function"):
        RobotClientConfig(
            robot=MockRobotConfig(),
            server_address="localhost:9999",
            policy_type="test",
            pretrained_name_or_path="test",
            actions_per_chunk=20,
            aggregate_fn_name="unknown",
        )


def test_robot_client_config_rejects_invalid_rtc_smooth_steps():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    with pytest.raises(ValueError, match="rtc_smooth_safe_prefix_steps"):
        RobotClientConfig(
            robot=MockRobotConfig(),
            server_address="localhost:9999",
            policy_type="test",
            pretrained_name_or_path="test",
            actions_per_chunk=20,
            aggregate_fn_name="rtc_smooth",
            rtc_smooth_safe_prefix_steps=-1,
        )

    with pytest.raises(ValueError, match="rtc_smooth_blend_steps"):
        RobotClientConfig(
            robot=MockRobotConfig(),
            server_address="localhost:9999",
            policy_type="test",
            pretrained_name_or_path="test",
            actions_per_chunk=20,
            aggregate_fn_name="rtc_smooth",
            rtc_smooth_blend_steps=-1,
        )


def test_rtc_smooth_keeps_safe_prefix_blends_overlap_and_replaces_future(robot_client):
    robot_client.config.aggregate_fn_name = "rtc_smooth"
    robot_client.config.rtc_smooth_safe_prefix_steps = 1
    robot_client.config.rtc_smooth_blend_steps = 3
    robot_client.config.rtc_smooth_exp_schedule = False
    robot_client.latest_action = 4

    for action in _make_constant_actions(start_ts=100.0, start_t=5, values=[0, 0, 0, 0, 0]):
        robot_client.action_queue.put(action)
    incoming = _make_constant_actions(start_ts=101.0, start_t=5, values=[10, 10, 10, 10, 10, 10])

    robot_client._aggregate_action_queues(incoming, receive_time=100.5)

    timesteps, values = _queue_timesteps_and_values(robot_client.action_queue)
    assert timesteps == [5, 6, 7, 8, 9, 10]
    assert values == pytest.approx([0.0, 2.5, 5.0, 7.5, 10.0, 10.0])


def test_rtc_smooth_drops_stale_and_expired_incoming_actions(robot_client):
    robot_client.config.aggregate_fn_name = "rtc_smooth"
    robot_client.config.rtc_smooth_safe_prefix_steps = 2
    robot_client.config.rtc_smooth_blend_steps = 2
    robot_client.latest_action = 4

    for action in _make_constant_actions(start_ts=100.0, start_t=5, values=[50, 60]):
        robot_client.action_queue.put(action)

    incoming = _make_constant_actions(start_ts=99.90, start_t=3, values=[3, 4, 5, 6, 7, 8])

    robot_client._aggregate_action_queues(incoming, receive_time=100.0)

    timesteps, values = _queue_timesteps_and_values(robot_client.action_queue)
    assert timesteps == [5, 6, 7, 8]
    assert values == pytest.approx([50.0, 60.0, 7.0, 8.0])


def test_rtc_smooth_keeps_old_queue_when_incoming_all_expired(robot_client):
    robot_client.config.aggregate_fn_name = "rtc_smooth"
    robot_client.latest_action = 4

    for action in _make_constant_actions(start_ts=100.0, start_t=5, values=[50, 60, 70]):
        robot_client.action_queue.put(action)
    incoming = _make_constant_actions(start_ts=99.0, start_t=5, values=[5, 6, 7])

    robot_client._aggregate_action_queues(incoming, receive_time=100.0)

    timesteps, values = _queue_timesteps_and_values(robot_client.action_queue)
    assert timesteps == [5, 6, 7]
    assert values == pytest.approx([50.0, 60.0, 70.0])


def test_rtc_smooth_single_overlap_blends_half_old_half_new(robot_client):
    robot_client.config.aggregate_fn_name = "rtc_smooth"
    robot_client.config.rtc_smooth_safe_prefix_steps = 0
    robot_client.config.rtc_smooth_blend_steps = 1
    robot_client.config.rtc_smooth_exp_schedule = False
    robot_client.latest_action = 4

    robot_client.action_queue.put(_make_constant_actions(start_ts=100.0, start_t=5, values=[0])[0])
    incoming = _make_constant_actions(start_ts=101.0, start_t=5, values=[10])

    robot_client._aggregate_action_queues(incoming, receive_time=100.5)

    timesteps, values = _queue_timesteps_and_values(robot_client.action_queue)
    assert timesteps == [5]
    assert values == pytest.approx([5.0])


@pytest.mark.parametrize(
    "chunk_size, queue_len, expected",
    [
        (20, 12, False),  # 12 / 20 = 0.6  > g=0.5 threshold, not ready to send
        (20, 8, True),  # 8  / 20 = 0.4 <= g=0.5, ready to send
        (10, 5, True),
        (10, 6, False),
    ],
)
def test_ready_to_send_observation(robot_client, chunk_size: int, queue_len: int, expected: bool):
    """Validate `_ready_to_send_observation` ratio logic for various sizes."""

    robot_client.action_chunk_size = chunk_size

    # Clear any existing actions then fill with `queue_len` dummy entries ----
    robot_client.action_queue = Queue()

    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


@pytest.mark.parametrize(
    "g_threshold, expected",
    [
        # The condition is `queue_size / chunk_size <= g`.
        # Here, ratio = 6 / 10 = 0.6.
        (0.0, False),  # 0.6 <= 0.0 is False
        (0.1, False),
        (0.2, False),
        (0.3, False),
        (0.4, False),
        (0.5, False),
        (0.6, True),  # 0.6 <= 0.6 is True
        (0.7, True),
        (0.8, True),
        (0.9, True),
        (1.0, True),
    ],
)
def test_ready_to_send_observation_with_varying_threshold(robot_client, g_threshold: float, expected: bool):
    """Validate `_ready_to_send_observation` with fixed sizes and varying `g`."""
    # Fixed sizes for this test: ratio = 6 / 10 = 0.6
    chunk_size = 10
    queue_len = 6

    robot_client.action_chunk_size = chunk_size
    # This is the parameter we are testing
    robot_client._chunk_size_threshold = g_threshold

    # Fill queue with dummy actions
    robot_client.action_queue = Queue()
    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


def test_pose_act_piper_client_sends_pose7d_observation(monkeypatch):
    from lerobot.async_inference.helpers import TimedObservation
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    client = _make_pose_act_piper_client(monkeypatch)
    sent = []
    monkeypatch.setattr(client, "send_observation", lambda obs: sent.append(obs) or True)

    try:
        features = client.policy_config.lerobot_features
        assert features["observation.state"]["shape"] == (7,)
        assert features["observation.state"]["names"] == [
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "gripper_width",
        ]

        raw_observation = client.control_loop_observation(task="test")
    finally:
        client.stop()

    assert sent, "control_loop_observation should send a TimedObservation"
    assert isinstance(sent[0], TimedObservation)
    observation = sent[0].get_observation()
    assert tuple(observation[OBS_STATE].shape) == (2, 7)
    torch.testing.assert_close(observation[OBS_STATE][0], observation[OBS_STATE][1], rtol=0, atol=0)
    for key in ["x", "y", "z", "roll", "pitch", "yaw", "gripper_width"]:
        assert key in observation
        assert key in raw_observation
    image_keys = [key for key in observation if key.startswith(f"{OBS_IMAGES}.")]
    assert image_keys
    for key in image_keys:
        assert observation[key].shape[0] == 2


def test_pose_act_piper_client_sends_previous_and_current_frames(monkeypatch):
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    client = _make_pose_act_piper_client(monkeypatch)
    sent = []
    monkeypatch.setattr(client, "send_observation", lambda obs: sent.append(obs) or True)

    try:
        client.control_loop_observation(task="test")
        first_obs = sent[-1].get_observation()
        first_pose = first_obs[OBS_STATE][1].clone()
        client.robot._ee_state["x"] += 0.05
        client.control_loop_observation(task="test")
        second_obs = sent[-1].get_observation()
    finally:
        client.stop()

    torch.testing.assert_close(second_obs[OBS_STATE][0], first_pose, rtol=0, atol=1e-6)
    assert second_obs[OBS_STATE][1, 0] > second_obs[OBS_STATE][0, 0]
    image_key = next(key for key in second_obs if key.startswith(f"{OBS_IMAGES}."))
    assert second_obs[image_key].shape[0] == 2


def test_async_observation_skips_busy_worker_without_warning_for_non_must_go(monkeypatch, caplog):
    client = _make_pose_act_piper_client(monkeypatch, async_observation=True)
    client.must_go.clear()
    client._observation_worker_busy.set()

    try:
        with caplog.at_level("WARNING"):
            scheduled = client._schedule_observation(task="test")
    finally:
        client._observation_worker_busy.clear()
        client.stop()

    assert scheduled is False
    assert not client.must_go.is_set()
    assert "Skipping async observation request" not in caplog.text


def test_async_observation_skips_full_queue_without_warning_for_non_must_go(monkeypatch, caplog):
    client = _make_pose_act_piper_client(monkeypatch, async_observation=True)
    client.must_go.clear()
    assert client._observation_request_queue is not None
    client._observation_request_queue.put_nowait(client._make_observation_request("first", False))

    try:
        with caplog.at_level("WARNING"):
            scheduled = client._schedule_observation(task="test")
    finally:
        client.stop()

    assert scheduled is False
    assert not client.must_go.is_set()
    assert "Skipping async observation request" not in caplog.text


def test_async_observation_skips_must_go_with_warning_and_keeps_must_go(monkeypatch, caplog):
    client = _make_pose_act_piper_client(monkeypatch, async_observation=True)
    client.must_go.set()
    client._observation_worker_busy.set()

    try:
        with caplog.at_level("WARNING"):
            scheduled = client._schedule_observation(task="test")
    finally:
        client._observation_worker_busy.clear()
        client.stop()

    assert scheduled is False
    assert client.must_go.is_set()
    assert "Skipping async observation request" in caplog.text
    assert "reason=worker_busy" in caplog.text
    assert "must_go observation was not submitted and will be retried" in caplog.text


def test_async_observation_clears_must_go_only_after_successful_send(monkeypatch, caplog):
    client = _make_pose_act_piper_client(monkeypatch, async_observation=True)
    sent = []

    try:
        client.must_go.set()
        monkeypatch.setattr(client, "send_observation", lambda obs: sent.append(obs) or False)
        with caplog.at_level("INFO"):
            client._send_observation_request(client._make_observation_request("test", False))
        assert client.must_go.is_set()
        assert sent[-1].must_go is True
        assert "Sent async observation request" not in caplog.text

        monkeypatch.setattr(client, "send_observation", lambda obs: sent.append(obs) or True)
        with caplog.at_level("INFO"):
            client._send_observation_request(client._make_observation_request("test", False))
        assert not client.must_go.is_set()
        assert sent[-1].must_go is True
        assert "Sent async observation request" in caplog.text
        assert "request_id=" in caplog.text
        assert "queue_size=" in caplog.text
        assert "action_chunk_size=" in caplog.text
    finally:
        client.stop()


def test_pose_act_piper_client_handles_pose7d_response(monkeypatch):
    client = _make_pose_act_piper_client(monkeypatch)

    try:
        action = client._action_tensor_to_action_dict(
            torch.tensor([0.21, 0.02, 0.31, 0.0, 0.0, 0.0, 0.04], dtype=torch.float32)
        )
    finally:
        client.stop()

    assert action == {
        "ee.abs_x": pytest.approx(0.21),
        "ee.abs_y": pytest.approx(0.02),
        "ee.abs_z": pytest.approx(0.1157),
        "ee.abs_rx": pytest.approx(0.0),
        "ee.abs_ry": pytest.approx(0.0),
        "ee.abs_rz": pytest.approx(0.0),
        "gripper.pos": pytest.approx(0.04),
    }


# -----------------------------------------------------------------------------
# Regression test: robot type registry populated by robot_client imports
# -----------------------------------------------------------------------------


def test_robot_client_registers_builtin_robot_types():
    """Importing robot_client must populate RobotConfig's ChoiceRegistry.

    This is a regression test for a bug introduced in #2425, where removing
    robot module imports from robot_client.py caused RobotConfig's registry to
    be empty, breaking CLI argument parsing with:
      error: argument --robot.type: invalid choice: 'so101_follower' (choose from )

    Robot types are registered via @RobotConfig.register_subclass() decorators
    at import time, so all supported modules must be explicitly imported.
    """
    import lerobot.async_inference.robot_client  # noqa: F401
    from lerobot.robots.config import RobotConfig

    known_choices = RobotConfig.get_known_choices()

    expected_robot_types = [
        "so100_follower",
        "so101_follower",
        "koch_follower",
        "omx_follower",
        "bi_so_follower",
    ]
    for robot_type in expected_robot_types:
        assert robot_type in known_choices, (
            f"Robot type '{robot_type}' is not registered in RobotConfig's ChoiceRegistry. "
            f"Ensure the corresponding module is imported in robot_client.py. "
            f"Known choices: {sorted(known_choices)}"
        )
