#!/usr/bin/env python

import json

import numpy as np
import pandas as pd
import pytest

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_DATA_PATH
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.pose_act_conversion import (
    PoseACTConversionError,
    convert_pose_feature,
    detect_rotation_encoding,
    prepare_pose_act_dataset,
)


def test_detect_rotation_encoding_from_feature_names():
    spec = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "roll_rad", "pitch_rad", "yaw_rad", "gripper"],
    }

    encoding = detect_rotation_encoding("observation.state", spec, np.zeros((4, 7), dtype=np.float32))

    assert encoding.representation == "euler"
    assert encoding.units == "radians"
    assert encoding.euler_order == "xyz"


def test_detect_rotation_encoding_supports_umi_dual_gripper_euler():
    spec = {
        "dtype": "float32",
        "shape": (8,),
        "names": {
            "axes": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_angle", "gripper_distance"]
        },
    }

    encoding = detect_rotation_encoding("observation.state", spec, np.zeros((4, 8), dtype=np.float32))

    assert encoding.representation == "euler"
    assert encoding.units == "radians"
    assert encoding.euler_order == "xyz"
    assert encoding.rotation_slice == (3, 6)
    assert encoding.gripper_index == 7
    assert encoding.source == "feature_names+umi_dual_gripper"


def test_detect_rotation_encoding_rejects_ambiguous_7d():
    spec = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
    }

    with pytest.raises(PoseACTConversionError):
        detect_rotation_encoding("action", spec, np.zeros((4, 7), dtype=np.float32))


def test_convert_pose_feature_from_axis_angle_to_pose10d():
    poses = np.array(
        [
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5],
            [0.1, 0.2, 0.3, 0.0, 0.0, np.pi / 2, 0.5],
        ],
        dtype=np.float32,
    )
    encoding = detect_rotation_encoding(
        "action",
        {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rot_axis_angle_x", "rot_axis_angle_y", "rot_axis_angle_z", "gripper"],
        },
        poses,
    )

    converted = convert_pose_feature(poses, encoding)

    assert converted.shape == (2, 10)
    np.testing.assert_allclose(converted[:, :3], poses[:, :3])
    np.testing.assert_allclose(converted[:, -1], poses[:, -1])


def test_convert_pose_feature_drops_umi_gripper_angle():
    poses = np.array(
        [
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.7, 0.4],
            [0.3, 0.2, 0.1, 0.0, 0.0, np.pi / 2, 0.9, 0.6],
        ],
        dtype=np.float32,
    )
    encoding = detect_rotation_encoding(
        "action",
        {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "axes": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_angle", "gripper_distance"]
            },
        },
        poses,
    )

    converted = convert_pose_feature(poses, encoding)

    assert converted.shape == (2, 10)
    np.testing.assert_allclose(converted[:, :3], poses[:, :3])
    np.testing.assert_allclose(converted[:, -1], poses[:, 7])


def test_prepare_pose_act_dataset_converts_local_dataset(tmp_path):
    src_root = tmp_path / "source"
    dst_root = tmp_path / "converted"
    repo_id = "test/apple_umi"

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=src_root,
        fps=10,
        use_videos=False,
        features={
            OBS_STATE: {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "roll_rad", "pitch_rad", "yaw_rad", "gripper"],
            },
            ACTION: {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "roll_rad", "pitch_rad", "yaw_rad", "gripper"],
            },
        },
    )
    dataset.add_frame(
        {
            OBS_STATE: np.array([0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.4], dtype=np.float32),
            ACTION: np.array([0.1, 0.2, 0.3, 0.0, 0.0, np.pi / 2, 0.5], dtype=np.float32),
            "task": "demo",
        }
    )
    dataset.add_frame(
        {
            OBS_STATE: np.array([0.2, 0.1, 0.0, 0.1, 0.0, 0.0, 0.6], dtype=np.float32),
            ACTION: np.array([0.2, 0.1, 0.0, 0.0, np.pi / 4, 0.0, 0.7], dtype=np.float32),
            "task": "demo",
        }
    )
    dataset.save_episode()
    dataset.finalize()

    manifest = prepare_pose_act_dataset(
        repo_id,
        root=src_root,
        output_root=dst_root,
        output_repo_id=f"{repo_id}_pose_act",
    )

    assert manifest["features"][OBS_STATE]["target_width"] == 10
    assert manifest["features"][ACTION]["target_width"] == 10

    converted = LeRobotDataset(repo_id=f"{repo_id}_pose_act", root=dst_root)
    assert converted.meta.features[OBS_STATE]["shape"] == (10,)
    assert converted.meta.features[ACTION]["shape"] == (10,)

    data_path = dst_root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    df = pd.read_parquet(data_path)
    assert np.stack(df[OBS_STATE].to_list()).shape[1] == 10
    assert np.stack(df[ACTION].to_list()).shape[1] == 10

    manifest_path = dst_root / "meta" / "pose_act_conversion_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["features"][OBS_STATE]["encoding"]["representation"] == "euler"


def test_prepare_pose_act_dataset_converts_local_umi_dual_gripper_dataset(tmp_path):
    src_root = tmp_path / "source_umi"
    dst_root = tmp_path / "converted_umi"
    repo_id = "local/umi-episode1-v30"
    feature_names = {
        "axes": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_angle", "gripper_distance"]
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=src_root,
        fps=30,
        use_videos=False,
        features={
            OBS_STATE: {"dtype": "float32", "shape": (8,), "names": feature_names},
            ACTION: {"dtype": "float32", "shape": (8,), "names": feature_names},
        },
    )
    dataset.add_frame(
        {
            OBS_STATE: np.array([0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.8, 0.4], dtype=np.float32),
            ACTION: np.array([0.1, 0.2, 0.3, 0.0, 0.0, np.pi / 2, 0.9, 0.5], dtype=np.float32),
            "task": "demo",
        }
    )
    dataset.add_frame(
        {
            OBS_STATE: np.array([0.2, 0.1, 0.0, 0.1, 0.0, 0.0, 0.6, 0.3], dtype=np.float32),
            ACTION: np.array([0.2, 0.1, 0.0, 0.0, np.pi / 4, 0.0, 0.7, 0.2], dtype=np.float32),
            "task": "demo",
        }
    )
    dataset.save_episode()
    dataset.finalize()

    manifest = prepare_pose_act_dataset(
        repo_id,
        root=src_root,
        output_root=dst_root,
        output_repo_id=f"{repo_id}_pose_act",
    )

    converted = LeRobotDataset(repo_id=f"{repo_id}_pose_act", root=dst_root)
    assert converted.meta.features[OBS_STATE]["shape"] == (10,)
    assert converted.meta.features[ACTION]["shape"] == (10,)

    data_path = dst_root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    df = pd.read_parquet(data_path)
    converted_states = np.stack(df[OBS_STATE].to_list())
    converted_actions = np.stack(df[ACTION].to_list())
    np.testing.assert_allclose(converted_states[:, -1], [0.4, 0.3])
    np.testing.assert_allclose(converted_actions[:, -1], [0.5, 0.2])

    assert manifest["features"][OBS_STATE]["encoding"]["source"] == "feature_names+umi_dual_gripper"
    assert manifest["features"][ACTION]["encoding"]["source"] == "feature_names+umi_dual_gripper"
