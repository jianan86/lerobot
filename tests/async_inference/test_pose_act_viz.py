from __future__ import annotations

import pickle
from contextlib import contextmanager

import pytest
import torch

pytest.importorskip("grpc")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.async_inference.helpers import TimedAction
from lerobot.async_inference.pose_act_viz import (
    FifoChunkBuffer,
    LeRobotReplaySource,
    LatestChunkBuffer,
    LoadedObservation,
    PoseActVizClient,
    SocketActionChunkConsumer,
    _to_image_hwc_uint8,
    _save_observation_images,
)
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


class _FakeDataset:
    def __init__(self, frame: dict[str, object]):
        self._frame = frame

    def __getitem__(self, idx: int) -> dict[str, object]:
        if idx != 3:
            raise IndexError(idx)
        return self._frame


def test_lerobot_replay_source_builds_raw_observation(monkeypatch):
    frame = {
        OBS_STATE: torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32),
        f"{OBS_IMAGES}.fisheye_rgb": torch.arange(3 * 4 * 5, dtype=torch.uint8).reshape(3, 4, 5),
        "task": "stack the cube",
    }
    monkeypatch.setattr(
        "lerobot.async_inference.pose_act_viz.LeRobotDataset",
        lambda *args, **kwargs: _FakeDataset(frame),
    )

    source = LeRobotReplaySource(repo_id="dummy/repo", camera_key="fisheye_rgb")
    loaded = source.load_frame(episode_idx=1, frame_idx=3)

    assert loaded.task == "stack the cube"
    assert loaded.preview_image.shape == (4, 5, 3)
    assert loaded.raw_observation["fisheye_rgb"].shape == (4, 5, 3)
    assert loaded.raw_observation["x"] == pytest.approx(0.1)
    assert loaded.raw_observation["gripper_width"] == pytest.approx(0.7)


def test_lerobot_replay_source_builds_history(monkeypatch):
    frames = {
        0: {
            OBS_STATE: torch.tensor([0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.01], dtype=torch.float32),
            f"{OBS_IMAGES}.fisheye_rgb": torch.zeros(3, 4, 5, dtype=torch.uint8),
        },
        1: {
            OBS_STATE: torch.tensor([0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.02], dtype=torch.float32),
            f"{OBS_IMAGES}.fisheye_rgb": torch.ones(3, 4, 5, dtype=torch.uint8),
        },
    }

    class _FakeHistoryDataset:
        def __getitem__(self, idx: int) -> dict[str, object]:
            return frames[idx]

    monkeypatch.setattr(
        "lerobot.async_inference.pose_act_viz.LeRobotDataset",
        lambda *args, **kwargs: _FakeHistoryDataset(),
    )

    source = LeRobotReplaySource(repo_id="dummy/repo", camera_key="fisheye_rgb")
    loaded = source.load_history(episode_idx=0, frame_idx=1, n_obs_steps=3)

    assert loaded.preview_image.shape == (4, 5, 3)
    assert loaded.raw_observation[OBS_STATE].shape == (3, 7)
    assert loaded.raw_observation[f"{OBS_IMAGES}.fisheye_rgb"].shape == (3, 4, 5, 3)
    torch.testing.assert_close(torch.as_tensor(loaded.raw_observation[OBS_STATE][-1]), frames[1][OBS_STATE])
    torch.testing.assert_close(torch.as_tensor(loaded.raw_observation[OBS_STATE][0]), frames[0][OBS_STATE])


def test_lerobot_replay_source_rejects_missing_camera(monkeypatch):
    frame = {
        OBS_STATE: torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32),
    }
    monkeypatch.setattr(
        "lerobot.async_inference.pose_act_viz.LeRobotDataset",
        lambda *args, **kwargs: _FakeDataset(frame),
    )

    source = LeRobotReplaySource(repo_id="dummy/repo", camera_key="fisheye_rgb")
    with pytest.raises(KeyError, match="fisheye_rgb"):
        source.load_frame(episode_idx=1, frame_idx=3)


def test_latest_chunk_buffer_overwrites_previous():
    buffer = LatestChunkBuffer()
    old_chunk = [TimedAction(timestamp=1.0, timestep=1, action=torch.zeros(7))]
    new_chunk = [TimedAction(timestamp=2.0, timestep=2, action=torch.ones(7))]

    buffer.replace(old_chunk)
    buffer.replace(new_chunk)

    latest = buffer.pop_latest()
    assert latest == new_chunk
    assert buffer.pop_latest() is None


def test_socket_consumer_serializes_chunk(monkeypatch):
    payloads = []

    class _FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def sendall(self, data: bytes):
            payloads.append(data)

    monkeypatch.setattr(
        "lerobot.async_inference.pose_act_viz.socket.create_connection",
        lambda *args, **kwargs: _FakeSocket(),
    )

    consumer = SocketActionChunkConsumer("127.0.0.1", 9019)
    chunk = [TimedAction(timestamp=3.0, timestep=4, action=torch.arange(7, dtype=torch.float32))]
    consumer.consume(
        chunk,
        {"request_id": "req-1", "episode_idx": 2, "frame_idx": 4, "source_timestep": 4},
    )

    size = int.from_bytes(payloads[0], byteorder="big")
    assert size == len(payloads[1])
    round_trip = pickle.loads(payloads[1])  # nosec
    assert round_trip["schema"] == "pose_act_pose7d_chunk_v1"
    assert round_trip["timesteps"] == [4]
    assert round_trip["timestamps"] == [3.0]
    assert round_trip["actions"][0] == torch.arange(7, dtype=torch.float32).tolist()
    assert round_trip["request_id"] == "req-1"
    assert round_trip["episode_idx"] == 2
    assert round_trip["frame_idx"] == 4


def test_fifo_chunk_buffer_preserves_order():
    buffer = FifoChunkBuffer()
    buffer.push({"frame_idx": 3})
    buffer.push({"frame_idx": 4})

    assert buffer.pop_next() == {"frame_idx": 3}
    assert buffer.pop_next() == {"frame_idx": 4}
    assert buffer.pop_next() is None


def test_save_observation_images(tmp_path):
    raw_observation = {
        OBS_STATE: torch.zeros(2, 7, dtype=torch.float32).numpy(),
        f"{OBS_IMAGES}.fisheye_rgb": torch.stack(
            [
                torch.zeros(4, 5, 3, dtype=torch.uint8),
                torch.full((4, 5, 3), 255, dtype=torch.uint8),
            ],
            dim=0,
        ).numpy(),
    }

    _save_observation_images(raw_observation, tmp_path)

    assert (tmp_path / "fisheye_rgb_obs_00.png").is_file()
    assert (tmp_path / "fisheye_rgb_obs_01.png").is_file()


def test_to_image_hwc_uint8_scales_unit_float_images():
    image = torch.tensor(
        [
            [[0.0, 0.5], [1.0, 0.25]],
            [[0.0, 0.5], [1.0, 0.25]],
            [[0.0, 0.5], [1.0, 0.25]],
        ],
        dtype=torch.float32,
    )

    converted = _to_image_hwc_uint8(image)

    assert converted.dtype == torch.uint8
    assert tuple(converted.shape) == (2, 2, 3)
    assert int(converted[0, 0, 0]) == 0
    assert int(converted[0, 1, 0]) in (127, 128)
    assert int(converted[1, 0, 0]) == 255


def test_pose_act_viz_client_forwards_chunk_to_consumer(monkeypatch):
    class _FakeSource:
        lerobot_features = {
            OBS_STATE: {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_width"]},
            f"{OBS_IMAGES}.fisheye_rgb": {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channels"]},
        }

        def load_history(
            self, episode_idx: int, frame_idx: int, n_obs_steps: int, task: str | None = None
        ) -> LoadedObservation:
            assert n_obs_steps == 2
            return LoadedObservation(
                raw_observation={
                    OBS_STATE: torch.zeros(2, 7, dtype=torch.float32).numpy(),
                    f"{OBS_IMAGES}.fisheye_rgb": torch.zeros(2, 8, 8, 3, dtype=torch.uint8).numpy(),
                },
                preview_image=torch.zeros(8, 8, 3, dtype=torch.uint8),
                task=task,
                episode_idx=episode_idx,
                frame_idx=frame_idx,
            )

    monkeypatch.setattr(
        "lerobot.async_inference.pose_act_viz.PreTrainedConfig.from_pretrained",
        lambda path: type("Cfg", (), {"n_obs_steps": 2})(),
    )
    forwarded = []
    client = PoseActVizClient(
        source=_FakeSource(),
        server_address="127.0.0.1:8080",
        pretrained_name_or_path="dummy-checkpoint",
        consumer=type("Consumer", (), {"consume": lambda self, actions, metadata=None: forwarded.append((actions, metadata))})(),
    )
    returned_chunk = [
        TimedAction(timestamp=1.0, timestep=3, action=torch.linspace(0, 1, 7)),
        TimedAction(timestamp=1.1, timestep=4, action=torch.linspace(1, 2, 7)),
    ]
    @contextmanager
    def _fake_session():
        yield object()

    monkeypatch.setattr(client, "_session", _fake_session)
    monkeypatch.setattr(client, "_request_actions", lambda stub, timed_observation: returned_chunk)

    actions = client.run_once(episode_idx=2, frame_idx=3, preview_image=False)

    assert actions == returned_chunk
    assert len(forwarded) == 1
    forwarded_actions, forwarded_metadata = forwarded[0]
    assert forwarded_actions == returned_chunk
    assert forwarded_metadata["episode_idx"] == 2
    assert forwarded_metadata["frame_idx"] == 3
    assert forwarded_metadata["source_timestep"] == 3
    assert isinstance(forwarded_metadata["request_id"], str)


def test_pose_act_viz_client_replay_range_requests_frames_in_order(monkeypatch):
    class _FakeSource:
        lerobot_features = {
            OBS_STATE: {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_width"]},
            f"{OBS_IMAGES}.fisheye_rgb": {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channels"]},
        }

        def load_history(
            self, episode_idx: int, frame_idx: int, n_obs_steps: int, task: str | None = None
        ) -> LoadedObservation:
            return LoadedObservation(
                raw_observation={
                    OBS_STATE: torch.zeros(2, 7, dtype=torch.float32).numpy(),
                    f"{OBS_IMAGES}.fisheye_rgb": torch.zeros(2, 8, 8, 3, dtype=torch.uint8).numpy(),
                },
                preview_image=torch.zeros(8, 8, 3, dtype=torch.uint8),
                task=task,
                episode_idx=episode_idx,
                frame_idx=frame_idx,
            )

    monkeypatch.setattr(
        "lerobot.async_inference.pose_act_viz.PreTrainedConfig.from_pretrained",
        lambda path: type("Cfg", (), {"n_obs_steps": 2})(),
    )
    client = PoseActVizClient(
        source=_FakeSource(),
        server_address="127.0.0.1:8081",
        pretrained_name_or_path="dummy-checkpoint",
    )
    requested_timesteps = []

    @contextmanager
    def _fake_session():
        yield object()

    def _fake_request_actions(stub, timed_observation):
        requested_timesteps.append(timed_observation.timestep)
        return [TimedAction(timestamp=1.0, timestep=timed_observation.timestep, action=torch.zeros(7))]

    monkeypatch.setattr(client, "_session", _fake_session)
    monkeypatch.setattr(client, "_request_actions", _fake_request_actions)

    actions = client.replay_range(episode_idx=1, frame_start=3, frame_end=5)

    assert len(actions) == 3
    assert requested_timesteps == [3, 4, 5]
