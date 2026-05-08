from __future__ import annotations

import threading

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.replay.piper_remote import (
    POSE7D_ACTION_NAMES,
    EpisodeMeta,
    PiperReplayDatasetProvider,
    PiperReplayExecutor,
    PiperReplayTransportClient,
    ThreadedTCPServer,
    apply_relative_pose_targets,
    build_replay_request_handler,
    compute_relative_pose_targets,
)
from lerobot.utils.pose_act import absolute_pose10d, pose10d_to_pose7d


class _FakeSelectedActions:
    def __init__(self, actions: torch.Tensor):
        self._actions = actions

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"action": self._actions[idx]}


class _FakeDataset:
    def __init__(self, actions: torch.Tensor, fps: int = 30, names: tuple[str, ...] = POSE7D_ACTION_NAMES):
        self._actions = actions
        self.fps = fps
        self.num_frames = int(actions.shape[0])
        self.features = {"action": {"names": list(names)}}

    def select_columns(self, key: str) -> _FakeSelectedActions:
        assert key == "action"
        return _FakeSelectedActions(self._actions)


class _FakeTransport:
    def __init__(self, relative_poses: torch.Tensor, fps: int = 30):
        self._relative_poses = relative_poses
        self.meta = EpisodeMeta(
            episode_token="episode-0",
            episode_index=0,
            num_frames=int(relative_poses.shape[0]),
            fps=fps,
            action_names=POSE7D_ACTION_NAMES,
        )

    def get_episode_meta(self, episode_token: str) -> EpisodeMeta:
        assert episode_token == self.meta.episode_token
        return self.meta

    def get_relative_chunk(self, episode_token: str, frame_start: int, max_frames: int):
        assert episode_token == self.meta.episode_token
        frame_end = min(frame_start + max_frames, self.meta.num_frames)
        return type(
            "Chunk",
            (),
            {
                "poses": self._relative_poses[frame_start:frame_end].tolist(),
            },
        )()


class _FakePiperRobot:
    def __init__(self, base_pose7d: list[float], gripper_pos: float = 0.02):
        self._base_pose = base_pose7d
        self._gripper_pos = gripper_pos
        self.sent_actions: list[dict[str, float]] = []

    def _get_end_pose(self) -> dict[str, float]:
        return {
            "x": self._base_pose[0],
            "y": self._base_pose[1],
            "z": self._base_pose[2],
            "rx": self._base_pose[3],
            "ry": self._base_pose[4],
            "rz": self._base_pose[5],
        }

    def _get_motor_positions(self) -> dict[str, float]:
        return {"gripper.pos": self._gripper_pos}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.sent_actions.append(action)
        return action


def test_compute_relative_pose_targets_roundtrip_preserves_gripper():
    actions = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.03],
            [0.12, 0.18, 0.34, 0.1, 0.15, 0.25, 0.05],
        ],
        dtype=torch.float32,
    )
    base_pose = torch.tensor([0.5, -0.2, 0.7, -0.3, 0.4, 0.2, 0.01], dtype=torch.float32)

    relative = compute_relative_pose_targets(actions)
    rebuilt = apply_relative_pose_targets(relative, base_pose)

    expected = pose10d_to_pose7d(absolute_pose10d(relative, base_pose))
    torch.testing.assert_close(rebuilt, expected)
    assert rebuilt[1, 6] == pytest.approx(actions[1, 6].item())


def test_provider_returns_relative_chunk(monkeypatch):
    actions = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.03],
            [0.12, 0.18, 0.34, 0.1, 0.15, 0.25, 0.05],
            [0.14, 0.17, 0.31, 0.05, 0.2, 0.3, 0.04],
        ],
        dtype=torch.float32,
    )
    monkeypatch.setattr(
        "lerobot.replay.piper_remote.LeRobotDataset",
        lambda *args, **kwargs: _FakeDataset(actions, fps=60),
    )

    provider = PiperReplayDatasetProvider("dummy/repo")
    meta = provider.open_episode(3)
    chunk = provider.get_relative_chunk(meta.episode_token, frame_start=1, max_frames=2)

    assert meta.episode_index == 3
    assert meta.num_frames == 3
    assert meta.fps == 60
    assert chunk.frame_start == 1
    assert len(chunk.poses) == 2


def test_provider_rejects_bad_action_schema(monkeypatch):
    actions = torch.zeros(2, 7, dtype=torch.float32)
    monkeypatch.setattr(
        "lerobot.replay.piper_remote.LeRobotDataset",
        lambda *args, **kwargs: _FakeDataset(actions, names=("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")),
    )

    provider = PiperReplayDatasetProvider("dummy/repo")
    with pytest.raises(ValueError, match="action schema"):
        provider.open_episode(0)


def test_executor_sends_absolute_piper_actions():
    recorded_actions = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.03],
            [0.12, 0.18, 0.34, 0.1, 0.15, 0.25, 0.05],
        ],
        dtype=torch.float32,
    )
    relative_poses = compute_relative_pose_targets(recorded_actions)
    robot = _FakePiperRobot(base_pose7d=[0.4, -0.1, 0.5, -0.2, 0.3, 0.4, 0.01], gripper_pos=0.01)
    transport = _FakeTransport(relative_poses, fps=10_000)

    executor = PiperReplayExecutor(
        robot=robot,
        transport=transport,
        episode_token="episode-0",
        chunk_size=1,
        prefetch_threshold=0,
        fps=10_000,
    )
    executor.run_episode()

    assert len(robot.sent_actions) == 2
    assert robot.sent_actions[1]["gripper.pos"] == pytest.approx(0.05)
    assert {"ee.abs_x", "ee.abs_y", "ee.abs_z", "ee.abs_rx", "ee.abs_ry", "ee.abs_rz"} <= set(
        robot.sent_actions[1]
    )


def test_transport_roundtrip_over_socket(monkeypatch):
    actions = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.03],
            [0.12, 0.18, 0.34, 0.1, 0.15, 0.25, 0.05],
        ],
        dtype=torch.float32,
    )
    monkeypatch.setattr(
        "lerobot.replay.piper_remote.LeRobotDataset",
        lambda *args, **kwargs: _FakeDataset(actions, fps=25),
    )

    provider = PiperReplayDatasetProvider("dummy/repo")
    handler = build_replay_request_handler(provider)
    with ThreadedTCPServer(("127.0.0.1", 0), handler) as server:
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = PiperReplayTransportClient(host, port)
            meta = client.open_episode(0)
            chunk = client.get_relative_chunk(meta.episode_token, frame_start=0, max_frames=2)

            assert meta.fps == 25
            assert meta.action_names == POSE7D_ACTION_NAMES
            assert len(chunk.poses) == 2
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.0)
