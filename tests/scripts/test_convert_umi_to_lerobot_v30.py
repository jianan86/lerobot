from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lerobot.scripts.convert_umi_to_lerobot_v30 import (
    build_state_vector,
    compute_aligned_length,
    convert_episode,
    convert_episodes,
    discover_episode_dirs,
    discover_episode_files,
    get_episode_task,
    infer_features,
    load_synced_files,
    resolve_frame_slice,
)

try:
    import datasets  # noqa: F401

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


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


def _make_episode(root: Path, frames: int = 3, episode_name: str = "episode1", value_offset: int = 0) -> Path:
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
            {"angle": value + 0.1, "distance": value + 0.2},
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
    assert features["observation.depth.depth_camera"]["shape"] == (12, 16)
    assert features["observation.depth.depth_camera"]["dtype"] == "uint16"


def test_build_state_vector_uses_raw_xyz_euler_and_gripper_width(tmp_path):
    pose_path = tmp_path / "pose_rotated.json"
    gripper_path = tmp_path / "gripper_rotated.json"
    _write_json(pose_path, {"x": 0.1, "y": 0.2, "z": 0.3, "roll": 0.0, "pitch": 0.0, "yaw": float(np.pi / 2)})
    _write_json(gripper_path, {"angle": 0.7, "distance": 0.4})

    state = build_state_vector(pose_path, gripper_path)

    assert state.shape == (7,)
    np.testing.assert_allclose(state, np.array([0.1, 0.2, 0.3, 0.0, 0.0, np.pi / 2, 0.4], dtype=np.float32))


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
    assert dataset.meta.features["observation.depth.depth_camera"]["dtype"] == "uint16"
    item = dataset[0]
    assert tuple(item["observation.images.fisheye_rgb"].shape) == (3, 12, 16)
    assert tuple(item["observation.depth.depth_camera"].shape) == (12, 16)


def test_discover_episode_files_finds_expected_modalities(tmp_path):
    episode = _make_episode(tmp_path, frames=2)

    files = discover_episode_files(episode, cameras="depth_camera_rgb,fisheye_rgb,depth_camera")

    assert set(files) == {"depth_camera_rgb", "fisheye_rgb", "depth_camera", "pose", "gripper"}
    assert all(len(paths) == 2 for paths in files.values())
