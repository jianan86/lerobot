from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from lerobot.scripts.convert_umi_to_lerobot_v30 import (
    CONVERSION_ENCODER_QUEUE_MAXSIZE,
    CONVERSION_ENCODER_THREADS,
    CONVERSION_IMAGE_WRITER_PROCESSES,
    CONVERSION_IMAGE_WRITER_THREADS,
    add_episode_to_dataset,
    build_state_vector,
    compute_aligned_length,
    convert_episode,
    convert_episodes,
    detect_episode_arm_mode,
    discover_episode_dirs,
    discover_episode_files,
    euler_xyz_to_rotation_matrix,
    get_episode_task,
    infer_features,
    load_synced_files,
    resolve_frame_slice,
    rotation_matrix_to_euler_xyz,
    smooth_pika_pose_state_sequence,
)

try:
    import datasets  # noqa: F401

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import scipy  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib  # noqa: F401

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class _FakeDataset:
    def __init__(self) -> None:
        self.frames = []
        self.saved = False
        self.finalized = False

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.saved = True

    def finalize(self) -> None:
        self.finalized = True


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_rgb(path: Path, shape: tuple[int, int, int] = (12, 16, 3), value: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full(shape, value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _write_depth(path: Path, shape: tuple[int, int] = (12, 16), value: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full(shape, value, dtype=np.uint16)
    Image.fromarray(array).save(path)


def _make_episode(
    root: Path,
    frames: int = 3,
    episode_name: str = "episode1",
    value_offset: int = 0,
    gripper_distances: list[float] | None = None,
) -> Path:
    episode = root / episode_name
    _write_json(episode / "instructions.json", {"full-instructions": ["pick up the object"]})
    synced_names = []

    for idx in range(frames):
        value = value_offset + idx
        timestamp = 1000.0 + idx / 30
        stem = f"{timestamp:.6f}"
        synced_names.append(stem)
        _write_rgb(episode / "camera/color/pikaDepthCamera" / f"{stem}.jpg", value=idx + 1)
        _write_rgb(episode / "camera/color/pikaFisheyeCamera" / f"{stem}.jpg", value=idx + 2)
        _write_depth(episode / "camera/depth/pikaDepthCamera" / f"{stem}.png", value=idx + 100)
        _write_json(
            episode / "localization/pose/pika" / f"{stem}.json",
            {
                "x": value,
                "y": value + 1,
                "z": value + 2,
                "roll": value + 3,
                "pitch": value + 4,
                "yaw": value + 5,
            },
        )
        _write_json(
            episode / "gripper/encoder/pika" / f"{stem}.json",
            {
                "angle": value + 0.1,
                "distance": gripper_distances[idx] if gripper_distances is not None else value + 0.2,
            },
        )

    for directory, suffix in (
        ("camera/color/pikaDepthCamera", "jpg"),
        ("camera/color/pikaFisheyeCamera", "jpg"),
        ("camera/depth/pikaDepthCamera", "png"),
        ("localization/pose/pika", "json"),
        ("gripper/encoder/pika", "json"),
    ):
        (episode / directory / "sync.txt").write_text("\n".join(f"{stem}.{suffix}" for stem in synced_names))

    return episode


def _make_dual_arm_episode(
    root: Path,
    frames: int = 3,
    episode_name: str = "episode1",
) -> Path:
    episode = root / episode_name
    _write_json(episode / "instructions.json", {"full-instructions": ["pick up the object"]})
    synced_names = []

    for idx in range(frames):
        timestamp = 1000.0 + idx / 30
        stem = f"{timestamp:.6f}"
        synced_names.append(stem)
        for suffix, value_offset in (("l", 0), ("r", 100)):
            value = value_offset + idx
            _write_rgb(episode / f"camera/color/pikaDepthCamera_{suffix}" / f"{stem}.jpg", value=idx + value_offset + 1)
            _write_rgb(episode / f"camera/color/pikaFisheyeCamera_{suffix}" / f"{stem}.jpg", value=idx + value_offset + 2)
            _write_depth(episode / f"camera/depth/pikaDepthCamera_{suffix}" / f"{stem}.png", value=idx + value_offset + 100)
            _write_json(
                episode / f"localization/pose/pika_{suffix}" / f"{stem}.json",
                {
                    "x": value,
                    "y": value + 1,
                    "z": value + 2,
                    "roll": value + 3,
                    "pitch": value + 4,
                    "yaw": value + 5,
                },
            )
            _write_json(
                episode / f"gripper/encoder/pika_{suffix}" / f"{stem}.json",
                {"angle": value + 0.1, "distance": value + 0.2},
            )

    for suffix in ("l", "r"):
        for directory, file_suffix in (
            (f"camera/color/pikaDepthCamera_{suffix}", "jpg"),
            (f"camera/color/pikaFisheyeCamera_{suffix}", "jpg"),
            (f"camera/depth/pikaDepthCamera_{suffix}", "png"),
            (f"localization/pose/pika_{suffix}", "json"),
            (f"gripper/encoder/pika_{suffix}", "json"),
        ):
            (episode / directory / "sync.txt").write_text(
                "\n".join(f"{stem}.{file_suffix}" for stem in synced_names)
            )

    return episode


def test_load_synced_files_uses_sync_order(tmp_path):
    directory = tmp_path / "camera"
    directory.mkdir()
    _write_rgb(directory / "2.000000.jpg")
    _write_rgb(directory / "1.000000.jpg")
    _write_rgb(directory / "10.000000.jpg")
    (directory / "sync.txt").write_text("1.000000.jpg\n2.000000.jpg\n10.000000.jpg")

    ordered = load_synced_files(directory)

    assert [path.name for path in ordered] == ["1.000000.jpg", "2.000000.jpg", "10.000000.jpg"]


def test_discover_episode_dirs_uses_sorted_direct_children(tmp_path):
    _make_episode(tmp_path, frames=1, episode_name="episode_b")
    _make_episode(tmp_path, frames=1, episode_name="episode_a")
    (tmp_path / "not_episode.txt").write_text("ignored")

    episode_dirs = discover_episode_dirs(tmp_path)

    assert [path.name for path in episode_dirs] == ["episode_a", "episode_b"]


def test_get_episode_task_uses_fallback_for_null_payload(tmp_path):
    path = tmp_path / "instructions.json"
    path.write_text(json.dumps({"full-instructions": ["null"], "segment-instructions": []}))

    task = get_episode_task(path)

    assert task == "umi episode"


def test_build_state_vector_combines_pose_and_gripper(tmp_path):
    pose_path = tmp_path / "pose.json"
    gripper_path = tmp_path / "gripper.json"
    _write_json(pose_path, {"x": 1, "y": 2, "z": 3, "roll": 0, "pitch": 0, "yaw": 0})
    _write_json(gripper_path, {"angle": 7, "distance": 8})

    state = build_state_vector(pose_path, gripper_path)

    np.testing.assert_allclose(state, np.array([1, 2, 3, 0, 0, 0, 8], dtype=np.float32))
    assert state.dtype == np.float32


def test_compute_aligned_length_rejects_mismatched_modalities():
    with pytest.raises(ValueError, match="Synced modality counts do not match"):
        compute_aligned_length(
            {
                "a": [Path("1"), Path("2"), Path("3")],
                "b": [Path("1"), Path("2")],
                "c": [Path("1"), Path("2"), Path("3"), Path("4")],
            }
        )


def test_resolve_frame_slice_uses_floor_half_open_fraction_range():
    assert resolve_frame_slice(10, (0.2, 0.8)) == (2, 8)


@pytest.mark.parametrize("frame_range", [(-0.1, 0.8), (0.2, 1.1), (0.8, 0.8), (0.9, 0.8)])
def test_resolve_frame_slice_rejects_invalid_fraction_ranges(frame_range):
    with pytest.raises(ValueError, match="Frame range must satisfy"):
        resolve_frame_slice(10, frame_range)


def test_resolve_frame_slice_rejects_empty_selection():
    with pytest.raises(ValueError, match="selects no frames"):
        resolve_frame_slice(3, (0.0, 0.2))


def test_add_episode_to_dataset_auto_trim_removes_start_and_end_double_close_markers(tmp_path):
    distances = [0.02, 1.0, 0.02, 1.0, 1.0, 1.0, 1.0, 0.02, 1.0, 0.02]
    episode = _make_episode(tmp_path, frames=len(distances), gripper_distances=distances)
    dataset = _FakeDataset()

    manifest = add_episode_to_dataset(dataset, episode, cameras="none", auto_trim=True)

    assert manifest["auto_trim"] is True
    assert manifest["auto_trim_start_marker_found"] is True
    assert manifest["auto_trim_end_marker_found"] is True
    assert manifest["auto_trim_source_frame_start_before"] == 0
    assert manifest["auto_trim_source_frame_end_before"] == 10
    assert manifest["source_frame_start"] == 3
    assert manifest["source_frame_end"] == 7
    assert manifest["selected_frames"] == 4
    assert manifest["auto_trim_keep_ratio"] == 0.4
    assert len(dataset.frames) == 4
    np.testing.assert_allclose(dataset.frames[0]["observation.state"], np.array([3, 4, 5, 6, 7, 8, 1.0]))
    np.testing.assert_allclose(dataset.frames[-1]["observation.state"], np.array([6, 7, 8, 9, 10, 11, 1.0]))


def test_add_episode_to_dataset_auto_trim_keeps_frame_range_without_double_close_marker(tmp_path):
    episode = _make_episode(tmp_path, frames=6, gripper_distances=[1.0, 1.0, 0.02, 1.0, 1.0, 1.0])
    dataset = _FakeDataset()

    manifest = add_episode_to_dataset(dataset, episode, cameras="none", auto_trim=True)

    assert manifest["auto_trim_start_marker_found"] is False
    assert manifest["auto_trim_end_marker_found"] is False
    assert manifest["source_frame_start"] == 0
    assert manifest["source_frame_end"] == 6
    assert manifest["selected_frames"] == 6
    assert len(dataset.frames) == 6


def test_add_episode_to_dataset_auto_trim_ignores_single_long_close_run(tmp_path):
    episode = _make_episode(
        tmp_path,
        frames=10,
        gripper_distances=[1.0, 0.02, 0.02, 0.02, 0.02, 0.02, 1.0, 1.0, 1.0, 1.0],
    )
    dataset = _FakeDataset()

    manifest = add_episode_to_dataset(dataset, episode, cameras="none", auto_trim=True)

    assert manifest["auto_trim_start_marker_found"] is False
    assert manifest["auto_trim_end_marker_found"] is False
    assert manifest["source_frame_start"] == 0
    assert manifest["source_frame_end"] == 10
    assert len(dataset.frames) == 10


def test_add_episode_to_dataset_auto_trim_warns_when_keep_ratio_is_low(tmp_path, caplog):
    distances = [0.02, 1.0, 0.02, 1.0, 1.0, 1.0, 1.0, 0.02, 1.0, 0.02]
    episode = _make_episode(tmp_path, frames=len(distances), gripper_distances=distances)

    add_episode_to_dataset(
        _FakeDataset(),
        episode,
        cameras="none",
        auto_trim=True,
        auto_trim_min_keep_ratio=0.5,
    )

    assert "below minimum 50.0%" in caplog.text


def test_infer_features_defaults_to_fisheye_rgb(tmp_path):
    episode = _make_episode(tmp_path, frames=1)
    files = discover_episode_files(episode)

    features = infer_features(files)

    assert "observation.images.depth_camera_rgb" not in features
    assert "observation.depth.depth_camera" not in features
    assert features["observation.images.fisheye_rgb"]["shape"] == (3, 12, 16)
    assert features["observation.images.fisheye_rgb"]["dtype"] == "image"
    assert features["observation.state"]["shape"] == (7,)
    assert features["action"]["shape"] == (7,)
    assert features["observation.state"]["names"] == ["x", "y", "z", "roll", "pitch", "yaw", "gripper_width"]


def test_infer_features_uses_selected_camera_shapes_and_video_dtype(tmp_path):
    episode = _make_episode(tmp_path, frames=1)
    files = discover_episode_files(episode, cameras="depth_camera_rgb,fisheye_rgb,depth_camera")

    features = infer_features(
        files,
        cameras="depth_camera_rgb,fisheye_rgb,depth_camera",
        camera_storage="video",
    )

    assert features["observation.images.depth_camera_rgb"]["shape"] == (3, 12, 16)
    assert features["observation.images.depth_camera_rgb"]["dtype"] == "video"
    assert features["observation.images.fisheye_rgb"]["shape"] == (3, 12, 16)
    assert features["observation.images.fisheye_rgb"]["dtype"] == "video"
    assert features["observation.depth.depth_camera"]["shape"] == (1, 12, 16)
    assert features["observation.depth.depth_camera"]["dtype"] == "depth_video"


def test_build_state_vector_uses_raw_xyz_euler_and_gripper_width(tmp_path):
    pose_path = tmp_path / "pose_rotated.json"
    gripper_path = tmp_path / "gripper_rotated.json"
    _write_json(pose_path, {"x": 0.1, "y": 0.2, "z": 0.3, "roll": 0.0, "pitch": 0.0, "yaw": float(np.pi / 2)})
    _write_json(gripper_path, {"angle": 0.7, "distance": 0.4})

    state = build_state_vector(pose_path, gripper_path)

    assert state.shape == (7,)
    np.testing.assert_allclose(state, np.array([0.1, 0.2, 0.3, 0.0, 0.0, np.pi / 2, 0.4], dtype=np.float32))


def test_smooth_pika_pose_rejects_invalid_alpha():
    states = np.zeros((2, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="smooth_pika_pose_alpha"):
        smooth_pika_pose_state_sequence(states, alpha=0.0)


def test_smooth_pika_pose_rejects_invalid_mode():
    states = np.zeros((2, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="smooth_pika_pose_mode"):
        smooth_pika_pose_state_sequence(states, mode="centered")


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy extra required")
def test_smooth_pika_pose_uses_position_ema_rotation_slerp_and_raw_gripper():
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
            [2.0, 4.0, 6.0, 0.0, 0.0, np.pi / 2, 0.2],
            [4.0, 8.0, 12.0, 0.0, 0.0, np.pi / 2, 0.3],
        ],
        dtype=np.float32,
    )

    smoothed = smooth_pika_pose_state_sequence(states, alpha=0.5)

    assert smoothed.dtype == np.float32
    np.testing.assert_allclose(smoothed[0], states[0], atol=1e-6)
    np.testing.assert_allclose(smoothed[:, :3], np.array([[0, 0, 0], [1, 2, 3], [2.5, 5, 7.5]]), atol=1e-6)
    np.testing.assert_allclose(smoothed[:, 6], states[:, 6], atol=1e-6)
    np.testing.assert_allclose(smoothed[1, 3:6], np.array([0.0, 0.0, np.pi / 4]), atol=1e-5)
    np.testing.assert_allclose(smoothed[2, 3:6], np.array([0.0, 0.0, 3 * np.pi / 8]), atol=1e-5)

    for euler in smoothed[:, 3:6]:
        rot_mat = euler_xyz_to_rotation_matrix(*euler)
        np.testing.assert_allclose(rot_mat.T @ rot_mat, np.eye(3), atol=1e-5)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy extra required")
def test_smooth_pika_pose_zero_phase_uses_future_frames():
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
            [2.0, 4.0, 6.0, 0.0, 0.0, np.pi / 2, 0.2],
            [4.0, 8.0, 12.0, 0.0, 0.0, np.pi / 2, 0.3],
        ],
        dtype=np.float32,
    )

    smoothed = smooth_pika_pose_state_sequence(states, alpha=0.5, mode="zero_phase")

    np.testing.assert_allclose(
        smoothed[:, :3],
        np.array([[0.875, 1.75, 2.625], [1.75, 3.5, 5.25], [2.5, 5.0, 7.5]]),
        atol=1e-6,
    )
    np.testing.assert_allclose(smoothed[:, 6], states[:, 6], atol=1e-6)
    np.testing.assert_allclose(smoothed[0, 3:6], np.array([0.0, 0.0, 5 * np.pi / 32]), atol=1e-5)
    np.testing.assert_allclose(smoothed[1, 3:6], np.array([0.0, 0.0, 5 * np.pi / 16]), atol=1e-5)
    np.testing.assert_allclose(smoothed[2, 3:6], np.array([0.0, 0.0, 3 * np.pi / 8]), atol=1e-5)


def test_rotation_matrix_to_euler_xyz_roundtrip():
    euler = np.array([0.2, -0.3, 0.4], dtype=np.float32)

    restored = rotation_matrix_to_euler_xyz(euler_xyz_to_rotation_matrix(*euler))

    np.testing.assert_allclose(restored, euler, atol=1e-6)


def test_convert_episode_uses_fast_video_defaults(tmp_path, monkeypatch):
    import lerobot.datasets as datasets_module

    episode = _make_episode(tmp_path, frames=2)
    output_root = tmp_path / "output"
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return _FakeDataset()

    monkeypatch.setattr(datasets_module.LeRobotDataset, "create", staticmethod(fake_create))

    manifest = convert_episode(
        episode,
        output_root,
        repo_id="local/test-umi-fast-video-defaults",
        cameras="fisheye_rgb",
        camera_storage="video",
    )

    assert manifest["streaming_encoding"] is True
    assert manifest["vcodec"] == "h264"
    assert calls[0]["use_videos"] is True
    assert calls[0]["streaming_encoding"] is True
    assert calls[0]["vcodec"] == "h264"
    assert calls[0]["image_writer_processes"] == CONVERSION_IMAGE_WRITER_PROCESSES
    assert calls[0]["image_writer_threads"] == CONVERSION_IMAGE_WRITER_THREADS
    assert calls[0]["encoder_queue_maxsize"] == CONVERSION_ENCODER_QUEUE_MAXSIZE
    assert calls[0]["encoder_threads"] == CONVERSION_ENCODER_THREADS
    assert calls[0]["streaming_drop_frames"] is False


def test_convert_episode_can_disable_streaming_encoding(tmp_path, monkeypatch):
    import lerobot.datasets as datasets_module

    episode = _make_episode(tmp_path, frames=2)
    output_root = tmp_path / "output"
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return _FakeDataset()

    monkeypatch.setattr(datasets_module.LeRobotDataset, "create", staticmethod(fake_create))

    manifest = convert_episode(
        episode,
        output_root,
        repo_id="local/test-umi-no-streaming",
        cameras="fisheye_rgb",
        camera_storage="video",
        streaming_encoding=False,
        vcodec="libsvtav1",
    )

    assert manifest["streaming_encoding"] is False
    assert manifest["vcodec"] == "libsvtav1"
    assert calls[0]["use_videos"] is True
    assert calls[0]["streaming_encoding"] is False
    assert calls[0]["vcodec"] == "libsvtav1"


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets extra required")
def test_convert_episode_writes_local_lerobot_dataset(tmp_path):
    from lerobot.datasets import LeRobotDataset

    episode = _make_episode(tmp_path, frames=3)
    output_root = tmp_path / "output"

    manifest = convert_episode(episode, output_root, repo_id="local/test-umi", fps=30)
    dataset = LeRobotDataset("local/test-umi", root=output_root)

    assert manifest["aligned_frames"] == 3
    assert len(dataset) == 3
    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 3

    item = dataset[0]
    assert tuple(item["observation.state"].shape) == (7,)
    assert tuple(item["action"].shape) == (7,)
    assert "observation.images.depth_camera_rgb" not in item
    assert tuple(item["observation.images.fisheye_rgb"].shape) == (3, 12, 16)
    assert "observation.depth.depth_camera" not in item


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets extra required")
def test_convert_episode_writes_rgb_only_without_depth_directory(tmp_path):
    from lerobot.datasets import LeRobotDataset

    episode = _make_episode(tmp_path, frames=2)
    shutil.rmtree(episode / "camera/depth")
    output_root = tmp_path / "output"

    manifest = convert_episode(
        episode,
        output_root,
        repo_id="local/test-umi-rgb-only",
        fps=30,
        cameras="fisheye_rgb",
    )
    dataset = LeRobotDataset("local/test-umi-rgb-only", root=output_root)

    assert manifest["cameras"] == ("fisheye_rgb",)
    assert "observation.depth.depth_camera" not in dataset.meta.features
    assert tuple(dataset[0]["observation.images.fisheye_rgb"].shape) == (3, 12, 16)


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets extra required")
def test_convert_episode_writes_fractional_frame_range(tmp_path):
    from lerobot.datasets import LeRobotDataset

    episode = _make_episode(tmp_path, frames=10)
    output_root = tmp_path / "output"

    manifest = convert_episode(
        episode,
        output_root,
        repo_id="local/test-umi-range",
        fps=30,
        frame_range=(0.2, 0.8),
    )
    dataset = LeRobotDataset("local/test-umi-range", root=output_root)

    assert manifest["aligned_frames"] == 10
    assert manifest["selected_frames"] == 6
    assert manifest["frame_range"] == (0.2, 0.8)
    assert manifest["source_frame_start"] == 2
    assert manifest["source_frame_end"] == 8
    assert len(dataset) == 6
    np.testing.assert_allclose(
        dataset[0]["observation.state"],
        np.array([2, 3, 4, 5, 6, 7, 2.2], dtype=np.float32),
    )
    np.testing.assert_allclose(
        dataset[5]["observation.state"],
        np.array([7, 8, 9, 10, 11, 12, 7.2], dtype=np.float32),
    )
    assert dataset[0]["frame_index"].item() == 0
    assert dataset[5]["frame_index"].item() == 5


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets extra required")
def test_convert_episode_writes_auto_trimmed_lerobot_dataset(tmp_path):
    from lerobot.datasets import LeRobotDataset

    distances = [0.02, 1.0, 0.02, 1.0, 1.0, 1.0, 1.0, 0.02, 1.0, 0.02]
    episode = _make_episode(tmp_path, frames=len(distances), gripper_distances=distances)
    output_root = tmp_path / "output"

    manifest = convert_episode(
        episode,
        output_root,
        repo_id="local/test-umi-auto-trim",
        fps=30,
        cameras="none",
        auto_trim=True,
    )
    dataset = LeRobotDataset("local/test-umi-auto-trim", root=output_root)

    assert manifest["auto_trim"] is True
    assert manifest["auto_trim_keep_ratio"] == 0.4
    assert manifest["auto_trim_source_frame_start_before"] == 0
    assert manifest["auto_trim_source_frame_end_before"] == 10
    assert manifest["auto_trim_start_marker_found"] is True
    assert manifest["auto_trim_end_marker_found"] is True
    assert len(dataset) == 4
    np.testing.assert_allclose(dataset[0]["observation.state"], np.array([3, 4, 5, 6, 7, 8, 1.0]))
    np.testing.assert_allclose(dataset[3]["observation.state"], np.array([6, 7, 8, 9, 10, 11, 1.0]))


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets extra required")
@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy extra required")
def test_convert_episode_writes_smoothed_pika_pose(tmp_path):
    from lerobot.datasets import LeRobotDataset

    episode = _make_episode(tmp_path, frames=3)
    output_root = tmp_path / "output"

    manifest = convert_episode(
        episode,
        output_root,
        repo_id="local/test-umi-smoothed",
        fps=30,
        cameras="none",
        smooth_pika_pose=True,
        smooth_pika_pose_alpha=0.5,
        smooth_pika_pose_mode="zero_phase",
    )
    dataset = LeRobotDataset("local/test-umi-smoothed", root=output_root)

    assert manifest["smooth_pika_pose"] is True
    assert manifest["smooth_pika_pose_alpha"] == 0.5
    assert manifest["smooth_pika_pose_mode"] == "zero_phase"
    np.testing.assert_allclose(dataset[0]["observation.state"][:3], np.array([0.4375, 1.4375, 2.4375]), atol=1e-6)
    np.testing.assert_allclose(dataset[1]["observation.state"][:3], np.array([0.875, 1.875, 2.875]), atol=1e-6)
    np.testing.assert_allclose(dataset[2]["observation.state"][:3], np.array([1.25, 2.25, 3.25]), atol=1e-6)
    np.testing.assert_allclose(dataset[2]["observation.state"][6], 2.2, atol=1e-6)
    np.testing.assert_allclose(dataset[2]["action"], dataset[2]["observation.state"], atol=1e-6)


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets extra required")
def test_convert_episodes_writes_all_child_directories_as_episodes(tmp_path):
    from lerobot.datasets import LeRobotDataset

    input_root = tmp_path / "episodes"
    _make_episode(input_root, frames=3, episode_name="episode_a", value_offset=0)
    _make_episode(input_root, frames=2, episode_name="episode_b", value_offset=10)
    output_root = tmp_path / "output"

    manifest = convert_episodes(input_root, output_root, repo_id="local/test-umi-batch", fps=30)
    dataset = LeRobotDataset("local/test-umi-batch", root=output_root)

    assert manifest["total_episodes"] == 2
    assert manifest["total_selected_frames"] == 5
    assert [Path(item["input_episode"]).name for item in manifest["episodes"]] == ["episode_a", "episode_b"]
    assert len(dataset) == 5
    assert dataset.meta.total_episodes == 2
    assert dataset.meta.total_frames == 5
    assert dataset[0]["episode_index"].item() == 0
    assert dataset[3]["episode_index"].item() == 1
    np.testing.assert_allclose(
        dataset[3]["observation.state"],
        np.array([10, 11, 12, 13, 14, 15, 10.2], dtype=np.float32),
    )


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets extra required")
def test_convert_episode_writes_selected_depth_camera(tmp_path):
    from lerobot.datasets import LeRobotDataset

    episode = _make_episode(tmp_path, frames=2)
    output_root = tmp_path / "output"

    manifest = convert_episode(
        episode,
        output_root,
        repo_id="local/test-umi-selected",
        fps=30,
        cameras="fisheye_rgb,depth_camera",
    )
    dataset = LeRobotDataset("local/test-umi-selected", root=output_root)

    assert manifest["cameras"] == ("fisheye_rgb", "depth_camera")
    assert dataset.meta.features["observation.images.fisheye_rgb"]["dtype"] == "image"
    assert dataset.meta.features["observation.depth.depth_camera"]["dtype"] == "depth_video"
    item = dataset[0]
    assert tuple(item["observation.images.fisheye_rgb"].shape) == (3, 12, 16)
    assert tuple(item["observation.depth.depth_camera"].shape) == (1, 12, 16)
    assert item["observation.depth.depth_camera"].dtype == torch.uint16
    assert (output_root / "videos/observation.depth.depth_camera/chunk-000/file-000.mkv").is_file()
    assert not (output_root / "images/observation.depth.depth_camera").exists()
    assert item["observation.depth.depth_camera"][0, 0, 0].item() == 100


def test_discover_episode_files_finds_expected_modalities(tmp_path):
    episode = _make_episode(tmp_path, frames=2)

    files = discover_episode_files(episode, cameras="depth_camera_rgb,fisheye_rgb,depth_camera")

    assert set(files) == {"depth_camera_rgb", "fisheye_rgb", "depth_camera", "pose", "gripper"}
    assert all(len(paths) == 2 for paths in files.values())


def test_detect_episode_arm_mode_distinguishes_single_and_dual_layouts(tmp_path):
    single_episode = _make_episode(tmp_path / "single", frames=1)
    dual_episode = _make_dual_arm_episode(tmp_path / "dual", frames=1)

    assert detect_episode_arm_mode(single_episode) == "single"
    assert detect_episode_arm_mode(dual_episode) == "dual"


def test_discover_episode_files_selects_left_or_right_dual_arm_modalities(tmp_path):
    episode = _make_dual_arm_episode(tmp_path, frames=2)

    left_files = discover_episode_files(episode, cameras="fisheye_rgb,depth_camera", single_arm_side="left")
    right_files = discover_episode_files(episode, cameras="fisheye_rgb,depth_camera", single_arm_side="right")

    assert all("_l" in str(path.parent) for paths in left_files.values() for path in paths)
    assert all("_r" in str(path.parent) for paths in right_files.values() for path in paths)


def test_add_episode_to_dataset_uses_selected_dual_arm_side(tmp_path):
    episode = _make_dual_arm_episode(tmp_path, frames=2)
    dataset = _FakeDataset()

    manifest = add_episode_to_dataset(dataset, episode, cameras="none", single_arm_side="right")

    assert manifest["single_arm_side"] == "right"
    assert len(dataset.frames) == 2
    np.testing.assert_allclose(
        dataset.frames[0]["observation.state"],
        np.array([100, 101, 102, 103, 104, 105, 100.2], dtype=np.float32),
    )
    np.testing.assert_allclose(dataset.frames[0]["action"], dataset.frames[0]["observation.state"])


def test_convert_episode_requires_side_for_dual_arm_input_single_arm_output(tmp_path):
    episode = _make_dual_arm_episode(tmp_path, frames=1)

    with pytest.raises(ValueError, match="requires --single-arm-side"):
        convert_episode(episode, tmp_path / "output", cameras="none")


def test_convert_episode_records_dual_input_single_output_side(tmp_path, monkeypatch):
    import lerobot.datasets as datasets_module

    episode = _make_dual_arm_episode(tmp_path, frames=1)

    monkeypatch.setattr(datasets_module.LeRobotDataset, "create", staticmethod(lambda **kwargs: _FakeDataset()))

    manifest = convert_episode(
        episode,
        tmp_path / "output",
        repo_id="local/test-umi-dual-to-single",
        cameras="none",
        single_arm_side="left",
    )

    assert manifest["input_arm_mode"] == "dual"
    assert manifest["output_arm_mode"] == "single"
    assert manifest["single_arm_side"] == "left"
    assert manifest["selected_frames"] == 1


def test_convert_episodes_records_dual_input_single_output_side(tmp_path, monkeypatch):
    import lerobot.datasets as datasets_module

    input_root = tmp_path / "episodes"
    _make_dual_arm_episode(input_root, frames=1, episode_name="episode_a")
    _make_dual_arm_episode(input_root, frames=2, episode_name="episode_b")

    monkeypatch.setattr(datasets_module.LeRobotDataset, "create", staticmethod(lambda **kwargs: _FakeDataset()))

    manifest = convert_episodes(
        input_root,
        tmp_path / "output",
        repo_id="local/test-umi-dual-batch",
        cameras="none",
        single_arm_side="right",
    )

    assert manifest["input_arm_mode"] == "dual"
    assert manifest["output_arm_mode"] == "single"
    assert manifest["single_arm_side"] == "right"
    assert manifest["total_episodes"] == 2
    assert manifest["total_selected_frames"] == 3
    assert [item["single_arm_side"] for item in manifest["episodes"]] == ["right", "right"]


@pytest.mark.parametrize("make_input", [_make_episode, _make_dual_arm_episode])
def test_convert_episode_rejects_dual_arm_output_until_implemented(tmp_path, make_input):
    episode = make_input(tmp_path, frames=1)

    with pytest.raises(NotImplementedError, match="Dual-arm LeRobot output is not implemented"):
        convert_episode(
            episode,
            tmp_path / "output",
            cameras="none",
            output_arm_mode="dual",
            single_arm_side="left" if make_input is _make_dual_arm_episode else None,
        )


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy extra required")
@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib extra required")
def test_visualize_0429_episode94_pika_pose_smoothing():
    data_episode = Path("/home/jianan/workspace/data/0429/episode94")
    if not data_episode.exists():
        pytest.skip(f"Local validation data not found: {data_episode}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = discover_episode_files(data_episode, cameras="none")
    raw_states = np.stack(
        [
            build_state_vector(pose_path, gripper_path)
            for pose_path, gripper_path in zip(files["pose"], files["gripper"], strict=True)
        ]
    )
    smoothed_states = smooth_pika_pose_state_sequence(raw_states, alpha=0.5, mode="zero_phase")

    assert raw_states.shape == smoothed_states.shape == (162, 7)
    assert smoothed_states.dtype == np.float32
    np.testing.assert_allclose(smoothed_states[:, 6], raw_states[:, 6], atol=1e-6)

    output_dir = Path("outputs/umi_pose_smoothing")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "episode94_pose_smoothing.png"

    frame_idx = np.arange(len(raw_states))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for axis, name, col in zip(axes.flat, ["x", "y", "z", "roll", "pitch", "yaw"], range(6), strict=True):
        axis.plot(frame_idx, raw_states[:, col], label="raw", linewidth=1.0)
        axis.plot(frame_idx, smoothed_states[:, col], label="smoothed", linewidth=1.0)
        axis.set_title(name)
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("0429 episode94 Pika pose smoothing, zero_phase alpha=0.5")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    assert output_path.is_file()
