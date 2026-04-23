#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE
from lerobot.utils.pose_act import matrix_to_rotation_6d
from lerobot.utils.rotation import Rotation


POSE10D_NAMES = [
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


class PoseACTConversionError(ValueError):
    """Raised when a dataset cannot be converted to pose_act pose10d safely."""


@dataclass
class RotationEncoding:
    representation: str
    units: str | None = None
    euler_order: str | None = None
    source: str = "explicit"
    rotation_slice: tuple[int, int] | None = None
    gripper_index: int = -1


def _normalize_names(feature_names: list[str] | dict[str, list[str]] | None, width: int) -> list[str]:
    if feature_names is None:
        return []
    if isinstance(feature_names, dict):
        feature_names = feature_names.get("axes")
        if feature_names is None:
            return []
    if len(feature_names) != width:
        return []
    return [name.lower() for name in feature_names]


def _is_umi_dual_gripper_euler(rotation_names: list[str], feature_names: list[str], width: int) -> bool:
    if width != 8 or len(feature_names) != width:
        return False
    return rotation_names == ["roll", "pitch", "yaw"] and feature_names[-2:] == ["gripper_angle", "gripper_distance"]


def _infer_units_from_names(rotation_names: list[str]) -> str | None:
    if any("deg" in name or "degree" in name for name in rotation_names):
        return "degrees"
    if any("rad" in name or "radian" in name for name in rotation_names):
        return "radians"
    return None


def _infer_units_from_values(rotation_values: np.ndarray) -> str:
    max_abs = float(np.max(np.abs(rotation_values)))
    if max_abs > 2 * math.pi * 1.25:
        return "degrees"
    return "radians"


def _infer_euler_order(rotation_names: list[str]) -> str | None:
    if not rotation_names:
        return None

    normalized = []
    for name in rotation_names:
        if "roll" in name:
            normalized.append("x")
        elif "pitch" in name:
            normalized.append("y")
        elif "yaw" in name:
            normalized.append("z")
        else:
            return None

    order = "".join(normalized)
    if len(order) == 3:
        return order
    return None


def detect_rotation_encoding(
    feature_name: str,
    feature_spec: dict[str, Any],
    sample_values: np.ndarray | None,
    rotation_format: str = "auto",
    rotation_unit: str = "auto",
    euler_order: str = "auto",
) -> RotationEncoding:
    width = feature_spec["shape"][0]
    feature_names = _normalize_names(feature_spec.get("names"), width)
    rotation_slice = (3, width - 1)
    gripper_index = width - 1
    rotation_names = feature_names[rotation_slice[0] : rotation_slice[1]] if len(feature_names) == width and width >= 7 else []

    if rotation_format != "auto":
        rep = rotation_format
        source = "override"
    elif _is_umi_dual_gripper_euler(rotation_names[:3], feature_names, width):
        rep = "euler"
        source = "feature_names+umi_dual_gripper"
        rotation_slice = (3, 6)
        gripper_index = 7
        rotation_names = feature_names[rotation_slice[0] : rotation_slice[1]]
    elif width == 10:
        rep = "rot6d"
        source = "shape"
    elif width == 8:
        rep = "quaternion"
        source = "shape"
    elif width == 7:
        if any(token in name for name in rotation_names for token in ["axis_angle", "rotvec", "rot_axis_angle"]):
            rep = "axis_angle"
            source = "feature_names"
        elif any(token in name for name in rotation_names for token in ["euler", "roll", "pitch", "yaw", "rpy"]):
            rep = "euler"
            source = "feature_names"
        else:
            raise PoseACTConversionError(
                f"Ambiguous 7D pose feature {feature_name!r}: could not determine whether rotation is "
                "axis-angle or Euler from metadata. Pass an explicit override."
            )
    else:
        raise PoseACTConversionError(
            f"Unsupported pose width {width} for {feature_name!r}. Expected 7, 8, or 10."
        )

    if rep in {"rot6d", "quaternion"}:
        units = None
    elif rotation_unit != "auto":
        units = rotation_unit
    else:
        name_units = _infer_units_from_names(rotation_names)
        if name_units is not None:
            units = name_units
        elif sample_values is not None:
            units = _infer_units_from_values(sample_values[:, rotation_slice[0] : rotation_slice[1]])
            source = f"{source}+value_inference"
        else:
            units = "radians"
            source = f"{source}+default_radians"

    if rep != "euler":
        order = None
    elif euler_order != "auto":
        order = euler_order.lower()
    else:
        inferred_order = _infer_euler_order(rotation_names)
        if inferred_order is None:
            raise PoseACTConversionError(
                f"Euler pose feature {feature_name!r} is missing an explicit axis order in metadata. "
                "Pass --euler-order to continue safely."
            )
        order = inferred_order

    return RotationEncoding(
        representation=rep,
        units=units,
        euler_order=order,
        source=source,
        rotation_slice=rotation_slice,
        gripper_index=gripper_index,
    )


def _rotation_about_axis(axis: str, angles: np.ndarray) -> np.ndarray:
    c = np.cos(angles)
    s = np.sin(angles)
    mats = np.zeros((angles.shape[0], 3, 3), dtype=np.float32)

    if axis == "x":
        mats[:, 0, 0] = 1.0
        mats[:, 1, 1] = c
        mats[:, 1, 2] = -s
        mats[:, 2, 1] = s
        mats[:, 2, 2] = c
    elif axis == "y":
        mats[:, 0, 0] = c
        mats[:, 0, 2] = s
        mats[:, 1, 1] = 1.0
        mats[:, 2, 0] = -s
        mats[:, 2, 2] = c
    elif axis == "z":
        mats[:, 0, 0] = c
        mats[:, 0, 1] = -s
        mats[:, 1, 0] = s
        mats[:, 1, 1] = c
        mats[:, 2, 2] = 1.0
    else:
        raise PoseACTConversionError(f"Unsupported Euler axis {axis!r}.")

    return mats


def _euler_to_matrices(rotation_values: np.ndarray, order: str) -> np.ndarray:
    mats = np.repeat(np.eye(3, dtype=np.float32)[None, ...], rotation_values.shape[0], axis=0)
    for axis, component in zip(order, rotation_values.transpose(1, 0), strict=True):
        mats = mats @ _rotation_about_axis(axis, component)
    return mats


def _pose_to_rot6d(poses: np.ndarray, encoding: RotationEncoding) -> np.ndarray:
    if poses.shape[1] == 10 and encoding.representation == "rot6d":
        return poses.astype(np.float32, copy=False)

    pos = poses[:, :3].astype(np.float32, copy=False)
    rotation_start, rotation_end = encoding.rotation_slice or (3, poses.shape[1] - 1)
    grip = poses[:, encoding.gripper_index : encoding.gripper_index + 1].astype(np.float32, copy=False)
    rotation_values = poses[:, rotation_start:rotation_end].astype(np.float32, copy=False)

    if encoding.representation == "axis_angle":
        if encoding.units == "degrees":
            rotation_values = np.deg2rad(rotation_values)
        rot_mats = np.stack([Rotation.from_rotvec(rot).as_matrix() for rot in rotation_values]).astype(np.float32)
    elif encoding.representation == "quaternion":
        rot_mats = np.stack([Rotation.from_quat(rot).as_matrix() for rot in rotation_values]).astype(np.float32)
    elif encoding.representation == "euler":
        if encoding.euler_order is None:
            raise PoseACTConversionError("Euler conversion requires a non-empty Euler order.")
        if encoding.units == "degrees":
            rotation_values = np.deg2rad(rotation_values)
        rot_mats = _euler_to_matrices(rotation_values, encoding.euler_order)
    elif encoding.representation == "rot6d":
        rot_mats = None
    else:
        raise PoseACTConversionError(f"Unsupported rotation representation {encoding.representation!r}.")

    if rot_mats is None:
        rot6d = poses[:, 3:9].astype(np.float32, copy=False)
    else:
        rot6d = matrix_to_rotation_6d(torch.from_numpy(rot_mats)).cpu().numpy().astype(np.float32)

    return np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32, copy=False)


def convert_pose_feature(poses: np.ndarray, encoding: RotationEncoding) -> np.ndarray:
    if poses.ndim != 2:
        raise PoseACTConversionError(f"Expected a 2D pose array, got shape {poses.shape}.")
    return _pose_to_rot6d(poses, encoding)


def _stack_feature_values(series: Any, feature_name: str) -> np.ndarray:
    values = list(series)
    if len(values) == 0:
        raise PoseACTConversionError(f"No values found for feature {feature_name!r}.")
    return np.stack([np.asarray(v, dtype=np.float32) for v in values], axis=0)


def _default_output_repo_id(repo_id: str) -> str:
    if "/" in repo_id:
        owner, name = repo_id.split("/", 1)
        return f"{owner}/{name}_pose_act"
    return f"{repo_id}_pose_act"


def _pose10d_feature_spec(original: dict[str, Any]) -> dict[str, Any]:
    updated = dict(original)
    updated["shape"] = (10,)
    updated["names"] = list(POSE10D_NAMES)
    return updated


def _infer_existing_path_pattern(root: Path, base_dir: str, suffix: str) -> str | None:
    candidates = sorted((root / base_dir).glob(f"**/*{suffix}"))
    if not candidates:
        return None

    sample = candidates[0].relative_to(root)
    parts = list(sample.parts)
    for index, part in enumerate(parts):
        if part.startswith("chunk-"):
            digits = len(part.removeprefix("chunk-"))
            parts[index] = f"chunk-{{chunk_index:0{digits}d}}"
        elif part.startswith("file-") and part.endswith(suffix):
            digits = len(part.removeprefix("file-").removesuffix(suffix))
            parts[index] = f"file-{{file_index:0{digits}d}}{suffix}"
    if base_dir == "videos" and len(parts) >= 2:
        parts[1] = "{video_key}"
    return "/".join(parts)


def _normalize_info_dict(root: Path, info: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    from lerobot.datasets.feature_utils import create_empty_dataset_info
    from lerobot.datasets.utils import DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH

    features = info["features"]
    use_videos = any(ft["dtype"] == "video" for ft in features.values())
    normalized = create_empty_dataset_info(
        codebase_version=info["codebase_version"],
        fps=info["fps"],
        features=features,
        use_videos=use_videos,
        robot_type=info.get("robot_type"),
        chunks_size=info.get("chunks_size"),
        data_files_size_in_mb=info.get("data_files_size_in_mb"),
        video_files_size_in_mb=info.get("video_files_size_in_mb"),
    )

    normalized.update(info)
    inferred_data_path = _infer_existing_path_pattern(root, "data", ".parquet")
    inferred_video_path = _infer_existing_path_pattern(root, "videos", ".mp4") if use_videos else None
    normalized["data_path"] = info.get("data_path", inferred_data_path or DEFAULT_DATA_PATH)
    normalized["video_path"] = info.get("video_path", inferred_video_path or (DEFAULT_VIDEO_PATH if use_videos else None))

    data_paths = sorted((root / "data").glob("*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}.")

    total_frames = 0
    episode_ids: set[int] = set()
    for data_path in data_paths:
        df = pd.read_parquet(data_path, columns=["episode_index"])
        total_frames += len(df)
        episode_ids.update(int(ep) for ep in df["episode_index"].unique())

    tasks_path = root / "meta" / "tasks.parquet"
    total_tasks = len(pd.read_parquet(tasks_path)) if tasks_path.exists() else 0

    normalized["total_frames"] = info.get("total_frames", total_frames)
    normalized["total_episodes"] = info.get("total_episodes", len(episode_ids))
    normalized["total_tasks"] = info.get("total_tasks", total_tasks)
    normalized["splits"] = info.get("splits", {"train": f"0:{normalized['total_episodes']}"})
    return normalized


def prepare_pose_act_dataset(
    repo_id: str,
    *,
    root: str | Path | None = None,
    output_root: str | Path | None = None,
    output_repo_id: str | None = None,
    revision: str | None = None,
    rotation_format: str = "auto",
    rotation_unit: str = "auto",
    euler_order: str = "auto",
    overwrite: bool = False,
    sample_rows: int = 2048,
) -> dict[str, Any]:
    import pandas as pd

    from lerobot.datasets import LeRobotDataset, recompute_stats
    from lerobot.datasets.io_utils import load_info, write_info

    source_root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    if not (source_root / "meta" / "info.json").exists():
        source_dataset = LeRobotDataset(repo_id=repo_id, root=root, revision=revision)
        source_root = source_dataset.root

    if output_repo_id is None:
        output_repo_id = _default_output_repo_id(repo_id)
    if output_root is None:
        output_root = HF_LEROBOT_HOME / output_repo_id
    output_root = Path(output_root)

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)

    shutil.copytree(source_root, output_root)

    info = _normalize_info_dict(output_root, load_info(output_root))
    feature_specs = info["features"]
    source_feature_specs = {
        feature_name: dict(feature_specs[feature_name])
        for feature_name in [OBS_STATE, ACTION]
        if feature_name in feature_specs
    }
    tracked_features = [OBS_STATE, ACTION]
    encodings: dict[str, RotationEncoding] = {}

    data_paths = sorted((output_root / "data").glob("*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No data parquet files found under {output_root / 'data'}.")

    sample_df = pd.read_parquet(data_paths[0])
    if sample_rows > 0:
        sample_df = sample_df.head(sample_rows)

    for feature_name in tracked_features:
        if feature_name not in feature_specs:
            raise PoseACTConversionError(f"Dataset is missing required feature {feature_name!r}.")

        sample_values = None
        if feature_name in sample_df:
            sample_values = _stack_feature_values(sample_df[feature_name], feature_name)
        encodings[feature_name] = detect_rotation_encoding(
            feature_name,
            feature_specs[feature_name],
            sample_values,
            rotation_format=rotation_format,
            rotation_unit=rotation_unit,
            euler_order=euler_order,
        )

    for data_path in data_paths:
        df = pd.read_parquet(data_path)
        for feature_name, encoding in encodings.items():
            poses = _stack_feature_values(df[feature_name], feature_name)
            converted = convert_pose_feature(poses, encoding)
            df[feature_name] = [row for row in converted]
        df.to_parquet(data_path, index=False)

    for feature_name in tracked_features:
        feature_specs[feature_name] = _pose10d_feature_spec(feature_specs[feature_name])
    info = _normalize_info_dict(output_root, info)
    write_info(info, output_root)

    converted_dataset = LeRobotDataset(repo_id=output_repo_id, root=output_root)
    recompute_stats(converted_dataset)

    manifest = {
        "source_repo_id": repo_id,
        "source_root": str(source_root),
        "output_repo_id": output_repo_id,
        "output_root": str(output_root),
        "features": {
            feature_name: {
                "source_width": source_feature_specs[feature_name]["shape"][0],
                "target_width": 10,
                "encoding": asdict(encodings[feature_name]),
            }
            for feature_name in tracked_features
        },
        "target_representation": "pose10d",
    }
    manifest_path = output_root / "meta" / "pose_act_conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
