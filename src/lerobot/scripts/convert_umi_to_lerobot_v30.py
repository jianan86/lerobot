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

"""
Convert a single UMI episode directory into a local LeRobotDataset v3.0 dataset.

The converter intentionally targets the stable modalities confirmed in the sample
UMI episode layout:

- camera/color/pikaDepthCamera      -> observation.images.depth_camera_rgb
- camera/color/pikaFisheyeCamera    -> observation.images.fisheye_rgb
- camera/depth/pikaDepthCamera      -> observation.depth.depth_camera
- localization/pose/pika            -> observation.state[:9]
- gripper/encoder/pika              -> observation.state[9]

The generated dataset contains one episode. Since the sample data does not
include a separate robot control stream, `action` defaults to the same 10D
pose_act-ready vector as `observation.state`:

- xyz position (3)
- rotation_6d (6), converted from UMI roll/pitch/yaw using XYZ Euler order
- gripper distance (1)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_INPUT_EPISODE = Path("/home/jianan/workspace/data/episode1")
DEFAULT_OUTPUT_ROOT = Path("/home/jianan/workspace/data/lerobot_umi_episode1_v30")
DEFAULT_REPO_ID = "local/umi-episode1-v30"
DEFAULT_TASK = "umi episode"

DEPTH_CAMERA_DIR = Path("camera/color/pikaDepthCamera")
FISHEYE_CAMERA_DIR = Path("camera/color/pikaFisheyeCamera")
DEPTH_DIR = Path("camera/depth/pikaDepthCamera")
POSE_DIR = Path("localization/pose/pika")
GRIPPER_DIR = Path("gripper/encoder/pika")
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

def load_synced_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Required modality directory not found: {directory}")

    sync_path = directory / "sync.txt"
    if not sync_path.exists():
        raise FileNotFoundError(f"Required sync file not found: {sync_path}")

    files: list[Path] = []
    with sync_path.open() as f:
        for line in f:
            filename = line.strip()
            if not filename:
                continue
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"File listed in sync.txt does not exist: {path}")
            try:
                float(path.stem)
            except ValueError as exc:
                raise ValueError(f"Synced file name must start with a numeric timestamp: {path.name}") from exc
            files.append(path)

    if not files:
        raise ValueError(f"No synced files listed in required sync file: {sync_path}")
    return files


def load_pose_json(path: Path) -> np.ndarray:
    with path.open() as f:
        payload = json.load(f)

    values = [payload[key] for key in ("x", "y", "z", "roll", "pitch", "yaw")]
    return np.asarray(values, dtype=np.float32)


def load_gripper_json(path: Path) -> np.ndarray:
    with path.open() as f:
        payload = json.load(f)

    values = [payload["angle"], payload["distance"]]
    return np.asarray(values, dtype=np.float32)


def _rotation_about_axis(axis: str, angle: float) -> np.ndarray:
    c = np.float32(np.cos(angle))
    s = np.float32(np.sin(angle))

    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    raise ValueError(f"Unsupported Euler axis {axis!r}.")


def euler_xyz_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    for axis, angle in zip(("x", "y", "z"), (roll, pitch, yaw), strict=True):
        matrix = matrix @ _rotation_about_axis(axis, float(angle))
    return matrix.astype(np.float32, copy=False)


def rotation_matrix_to_rot6d(rot_mat: np.ndarray) -> np.ndarray:
    if rot_mat.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got shape {rot_mat.shape}")
    return np.concatenate((rot_mat[:, 0], rot_mat[:, 1]), axis=0).astype(np.float32, copy=False)


def load_rgb_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_depth_image(path: Path) -> np.ndarray:
    image = Image.open(path)
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected depth image '{path}' to be single-channel, got shape {array.shape}")
    if array.dtype != np.uint16:
        array = array.astype(np.uint16)
    return array


def build_state_vector(pose_path: Path, gripper_path: Path) -> np.ndarray:
    pose = load_pose_json(pose_path)
    _, gripper_distance = load_gripper_json(gripper_path)
    pos = pose[:3]
    rot_mat = euler_xyz_to_rotation_matrix(*pose[3:])
    rot6d = rotation_matrix_to_rot6d(rot_mat)
    return np.concatenate([pos, rot6d, np.array([gripper_distance], dtype=np.float32)]).astype(np.float32)


def get_episode_task(instructions_path: Path, task_override: str | None = None) -> str:
    if task_override:
        return task_override

    if not instructions_path.exists():
        return DEFAULT_TASK

    with instructions_path.open() as f:
        payload = json.load(f)

    candidates = _extract_instruction_strings(payload)
    for item in candidates:
        normalized = item.strip()
        if normalized and normalized.lower() != "null":
            return normalized

    return DEFAULT_TASK


def _extract_instruction_strings(payload: Any) -> list[str]:
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, list):
        values: list[str] = []
        for item in payload:
            values.extend(_extract_instruction_strings(item))
        return values
    if isinstance(payload, dict):
        values: list[str] = []
        for key in ("instruction", "instructions", "task", "tasks", "full-instructions", "segment-instructions"):
            if key in payload:
                values.extend(_extract_instruction_strings(payload[key]))
        return values
    return []


def infer_features(depth_camera_rgb: Path, fisheye_rgb: Path, depth_image: Path) -> dict[str, dict[str, Any]]:
    depth_rgb_shape = _get_rgb_feature_shape(depth_camera_rgb)
    fisheye_rgb_shape = _get_rgb_feature_shape(fisheye_rgb)
    depth_array = load_depth_image(depth_image)

    return {
        "observation.images.depth_camera_rgb": {"dtype": "image", "shape": depth_rgb_shape, "names": None},
        "observation.images.fisheye_rgb": {"dtype": "image", "shape": fisheye_rgb_shape, "names": None},
        "observation.depth.depth_camera": {"dtype": str(depth_array.dtype), "shape": depth_array.shape, "names": None},
        "observation.state": {"dtype": "float32", "shape": (10,), "names": POSE10D_NAMES},
        "action": {"dtype": "float32", "shape": (10,), "names": POSE10D_NAMES},
    }


def _get_rgb_feature_shape(path: Path) -> tuple[int, int, int]:
    image = load_rgb_image(path)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image '{path}' to decode as HxWx3, got shape {image.shape}")
    height, width, channels = image.shape
    return (channels, height, width)


def compute_aligned_length(modality_files: dict[str, list[Path]]) -> int:
    lengths = {name: len(files) for name, files in modality_files.items()}
    aligned_length = next(iter(lengths.values()))
    if aligned_length <= 0:
        raise ValueError(f"Computed invalid aligned length from synced modality counts: {lengths}")
    if any(length != aligned_length for length in lengths.values()):
        raise ValueError(f"Synced modality counts do not match: {lengths}")
    return aligned_length


def discover_episode_files(episode_dir: Path) -> dict[str, list[Path]]:
    return {
        "depth_camera_rgb": load_synced_files(episode_dir / DEPTH_CAMERA_DIR),
        "fisheye_rgb": load_synced_files(episode_dir / FISHEYE_CAMERA_DIR),
        "depth_camera_depth": load_synced_files(episode_dir / DEPTH_DIR),
        "pose": load_synced_files(episode_dir / POSE_DIR),
        "gripper": load_synced_files(episode_dir / GRIPPER_DIR),
    }


def convert_episode(
    input_episode: Path,
    output_root: Path,
    repo_id: str = DEFAULT_REPO_ID,
    fps: int = 30,
    task: str | None = None,
    use_videos: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    from lerobot.datasets import LeRobotDataset

    input_episode = Path(input_episode)
    output_root = Path(output_root)

    modality_files = discover_episode_files(input_episode)
    aligned_length = compute_aligned_length(modality_files)
    task_text = get_episode_task(input_episode / "instructions.json", task_override=task)
    features = infer_features(
        modality_files["depth_camera_rgb"][0],
        modality_files["fisheye_rgb"][0],
        modality_files["depth_camera_depth"][0],
    )

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    logging.info("Detected modality counts: %s", {name: len(files) for name, files in modality_files.items()})
    logging.info("Using aligned frame count: %s", aligned_length)

    dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=output_root, use_videos=use_videos)

    try:
        for frame_idx in range(aligned_length):
            state = build_state_vector(modality_files["pose"][frame_idx], modality_files["gripper"][frame_idx])
            frame = {
                "task": task_text,
                "observation.images.depth_camera_rgb": load_rgb_image(modality_files["depth_camera_rgb"][frame_idx]),
                "observation.images.fisheye_rgb": load_rgb_image(modality_files["fisheye_rgb"][frame_idx]),
                "observation.depth.depth_camera": load_depth_image(modality_files["depth_camera_depth"][frame_idx]),
                "observation.state": state,
                "action": state.copy(),
            }
            dataset.add_frame(frame)

        dataset.save_episode()
        dataset.finalize()
    except Exception:
        dataset.finalize()
        raise

    return {
        "repo_id": repo_id,
        "input_episode": str(input_episode),
        "output_root": str(output_root),
        "fps": fps,
        "aligned_frames": aligned_length,
        "task": task_text,
        "features": features,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-episode", type=Path, default=DEFAULT_INPUT_EPISODE, help="Path to a single UMI episode directory.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root for the local LeRobotDataset v3.0 dataset.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Repo id recorded in the generated local LeRobot dataset metadata.")
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS recorded in metadata.")
    parser.add_argument("--task", help="Optional task text override.")
    parser.add_argument("--use-videos", action="store_true", help="Store visual observations as videos instead of images.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dataset root.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    manifest = convert_episode(
        input_episode=args.input_episode,
        output_root=args.output_root,
        repo_id=args.repo_id,
        fps=args.fps,
        task=args.task,
        use_videos=args.use_videos,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
