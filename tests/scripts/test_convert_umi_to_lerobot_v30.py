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
    discover_episode_files,
    euler_xyz_to_rotation_matrix,
    get_episode_task,
    infer_features,
    list_timestamped_files,
    rotation_matrix_to_rot6d,
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


def _make_episode(root: Path, frames: int = 3) -> Path:
    episode = root / "episode1"
    _write_json(episode / "instructions.json", {"full-instructions": ["pick up the object"]})

    for idx in range(frames):
        timestamp = 1000.0 + idx / 30
        stem = f"{timestamp:.6f}"
        _write_rgb(episode / "camera/color/pikaDepthCamera" / f"{stem}.jpg", value=idx + 1)
        _write_rgb(episode / "camera/color/pikaFisheyeCamera" / f"{stem}.jpg", value=idx + 2)
        _write_depth(episode / "camera/depth/pikaDepthCamera" / f"{stem}.png", value=idx + 100)
        _write_json(
            episode / "localization/pose/pika" / f"{stem}.json",
            {"x": idx, "y": idx + 1, "z": idx + 2, "roll": idx + 3, "pitch": idx + 4, "yaw": idx + 5},
        )
        _write_json(episode / "gripper/encoder/pika" / f"{stem}.json", {"angle": idx + 0.1, "distance": idx + 0.2})

    return episode


def test_list_timestamped_files_orders_by_numeric_stem(tmp_path):
    directory = tmp_path / "camera"
    _write_rgb(directory / "2.000000.jpg")
    _write_rgb(directory / "1.000000.jpg")
    _write_rgb(directory / "10.000000.jpg")

    ordered = list_timestamped_files(directory)

    assert [path.name for path in ordered] == ["1.000000.jpg", "2.000000.jpg", "10.000000.jpg"]


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

    np.testing.assert_allclose(state, np.array([1, 2, 3, 1, 0, 0, 0, 1, 0, 8], dtype=np.float32))
    assert state.dtype == np.float32


def test_compute_aligned_length_uses_shortest_modality():
    aligned = compute_aligned_length(
        {
            "a": [Path("1"), Path("2"), Path("3")],
            "b": [Path("1"), Path("2")],
            "c": [Path("1"), Path("2"), Path("3"), Path("4")],
        }
    )

    assert aligned == 2


def test_infer_features_uses_image_and_depth_shapes(tmp_path):
    rgb_a = tmp_path / "a.jpg"
    rgb_b = tmp_path / "b.jpg"
    depth = tmp_path / "d.png"
    _write_rgb(rgb_a, shape=(24, 32, 3))
    _write_rgb(rgb_b, shape=(12, 20, 3))
    _write_depth(depth, shape=(10, 14))

    features = infer_features(rgb_a, rgb_b, depth)

    assert features["observation.images.depth_camera_rgb"]["shape"] == (3, 24, 32)
    assert features["observation.images.fisheye_rgb"]["shape"] == (3, 12, 20)
    assert features["observation.depth.depth_camera"]["shape"] == (10, 14)
    assert features["observation.depth.depth_camera"]["dtype"] == "uint16"
    assert features["observation.state"]["shape"] == (10,)
    assert features["action"]["shape"] == (10,)
    assert features["observation.state"]["names"] == [
        "x",
        "y",
        "z",
        "rot6d_0",
        "rot6d_1",
        "rot6d_2",
        "rot6d_3",
        "rot6d_4",
        "rot6d_5",
        "gripper",
    ]


def test_build_state_vector_uses_xyz_euler_rot6d_and_gripper_distance(tmp_path):
    pose_path = tmp_path / "pose_rotated.json"
    gripper_path = tmp_path / "gripper_rotated.json"
    _write_json(pose_path, {"x": 0.1, "y": 0.2, "z": 0.3, "roll": 0.0, "pitch": 0.0, "yaw": float(np.pi / 2)})
    _write_json(gripper_path, {"angle": 0.7, "distance": 0.4})

    state = build_state_vector(pose_path, gripper_path)
    expected_rot6d = rotation_matrix_to_rot6d(euler_xyz_to_rotation_matrix(0.0, 0.0, np.pi / 2))

    assert state.shape == (10,)
    np.testing.assert_allclose(state[:3], np.array([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(state[3:9], expected_rot6d, atol=1e-6)
    np.testing.assert_allclose(state[9], np.float32(0.4))


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
    assert tuple(item["observation.state"].shape) == (10,)
    assert tuple(item["action"].shape) == (10,)
    assert tuple(item["observation.images.depth_camera_rgb"].shape) == (3, 12, 16)
    assert tuple(item["observation.images.fisheye_rgb"].shape) == (3, 12, 16)
    assert tuple(item["observation.depth.depth_camera"].shape) == (12, 16)


def test_discover_episode_files_finds_expected_modalities(tmp_path):
    episode = _make_episode(tmp_path, frames=2)

    files = discover_episode_files(episode)

    assert set(files) == {"depth_camera_rgb", "fisheye_rgb", "depth_camera_depth", "pose", "gripper"}
    assert all(len(paths) == 2 for paths in files.values())
