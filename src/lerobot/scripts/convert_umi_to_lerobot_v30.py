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
Convert a directory of UMI episodes into a local LeRobotDataset v3.0 dataset.

Each direct child directory under the input root is treated as one UMI episode
and is appended as one LeRobot episode in the output dataset.

The converter intentionally targets the stable modalities confirmed in the sample
UMI episode layout:

- camera/color/pikaFisheyeCamera    -> observation.images.fisheye_rgb (default)
- camera/color/pikaDepthCamera      -> observation.images.depth_camera_rgb (optional)
- camera/depth/pikaDepthCamera      -> observation.depth.depth_camera (optional)
- localization/pose/pika            -> observation.state[:6]
- gripper/encoder/pika              -> observation.state[6]

Since the sample data does not include a separate robot control stream,
`action` defaults to the same 7D raw pose vector as `observation.state`:

- xyz position (3)
- roll/pitch/yaw (3)
- gripper width (1)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_INPUT_ROOT = Path("/home/jianan/workspace/data/0421")
DEFAULT_OUTPUT_ROOT = Path("/home/jianan/workspace/data/lerobot_umi_0421_v30")
DEFAULT_REPO_ID = "local/umi-0421-v30"
DEFAULT_TASK = "umi episode"
DEFAULT_CAMERAS = ("fisheye_rgb",)
CAMERA_STORAGE_CHOICES = ("image", "video")
VIDEO_CODEC_CHOICES = (
    "h264",
    "hevc",
    "libsvtav1",
    "auto",
    "h264_videotoolbox",
    "hevc_videotoolbox",
    "h264_nvenc",
    "hevc_nvenc",
    "h264_vaapi",
    "h264_qsv",
)
PIKA_POSE_SMOOTHING_MODE_CHOICES = ("causal", "zero_phase")
DEFAULT_VIDEO_CODEC = "h264"
CONVERSION_IMAGE_WRITER_THREADS = 4
CONVERSION_IMAGE_WRITER_PROCESSES = 0
CONVERSION_ENCODER_QUEUE_MAXSIZE = 120
CONVERSION_ENCODER_THREADS = 4

DEPTH_CAMERA_DIR = Path("camera/color/pikaDepthCamera")
FISHEYE_CAMERA_DIR = Path("camera/color/pikaFisheyeCamera")
DEPTH_DIR = Path("camera/depth/pikaDepthCamera")
POSE_DIR = Path("localization/pose/pika")
GRIPPER_DIR = Path("gripper/encoder/pika")
POSE7D_NAMES = [
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper_width",
]
RGB_CAMERA_DIRS = {
    "depth_camera_rgb": DEPTH_CAMERA_DIR,
    "fisheye_rgb": FISHEYE_CAMERA_DIR,
}
RGB_CAMERA_FEATURES = {
    "depth_camera_rgb": "observation.images.depth_camera_rgb",
    "fisheye_rgb": "observation.images.fisheye_rgb",
}
DEPTH_CAMERA = "depth_camera"
DEPTH_CAMERA_FEATURE = "observation.depth.depth_camera"
SUPPORTED_CAMERAS = (*RGB_CAMERA_DIRS, DEPTH_CAMERA)


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
                raise ValueError(
                    f"Synced file name must start with a numeric timestamp: {path.name}"
                ) from exc
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


def rotation_matrix_to_euler_xyz(rot_mat: np.ndarray) -> np.ndarray:
    if rot_mat.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got shape {rot_mat.shape}")

    pitch = math.asin(float(np.clip(rot_mat[0, 2], -1.0, 1.0)))
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) > 1e-6:
        roll = math.atan2(float(-rot_mat[1, 2]), float(rot_mat[2, 2]))
        yaw = math.atan2(float(-rot_mat[0, 1]), float(rot_mat[0, 0]))
    else:
        yaw = 0.0
        if pitch > 0.0:
            roll = math.atan2(float(rot_mat[1, 0]), float(-rot_mat[2, 0]))
        else:
            roll = math.atan2(float(-rot_mat[1, 0]), float(rot_mat[2, 0]))

    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def rotation_matrix_to_rot6d(rot_mat: np.ndarray) -> np.ndarray:
    if rot_mat.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got shape {rot_mat.shape}")
    return np.concatenate((rot_mat[:, 0], rot_mat[:, 1]), axis=0).astype(np.float32, copy=False)


def validate_smoothing_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"smooth_pika_pose_alpha must satisfy 0.0 < alpha <= 1.0, got {alpha}")
    return alpha


def validate_pika_pose_smoothing_mode(mode: str) -> str:
    if mode not in PIKA_POSE_SMOOTHING_MODE_CHOICES:
        raise ValueError(
            f"smooth_pika_pose_mode must be one of {PIKA_POSE_SMOOTHING_MODE_CHOICES}, got {mode!r}"
        )
    return mode


def _require_scipy_rotation() -> tuple[Any, Any]:
    try:
        from scipy.spatial.transform import Rotation, Slerp
    except ImportError as exc:
        raise ImportError(
            "Pika pose smoothing requires scipy. Install the existing extra with "
            "`uv sync --locked --extra scipy-dep --extra matplotlib-dep`."
        ) from exc
    return Rotation, Slerp


def _smooth_pika_pose_state_sequence_causal(
    states: np.ndarray,
    alpha: float,
    rotation_cls: Any,
    slerp_cls: Any,
) -> np.ndarray:
    smoothed = states.astype(np.float32, copy=True)
    previous_rotation = rotation_cls.from_matrix(euler_xyz_to_rotation_matrix(*states[0, 3:6]))

    for idx in range(1, len(states)):
        smoothed[idx, :3] = alpha * states[idx, :3] + (1.0 - alpha) * smoothed[idx - 1, :3]

        current_rotation = rotation_cls.from_matrix(euler_xyz_to_rotation_matrix(*states[idx, 3:6]))
        slerp = slerp_cls([0.0, 1.0], rotation_cls.concatenate([previous_rotation, current_rotation]))
        previous_rotation = slerp([alpha])[0]
        smoothed[idx, 3:6] = rotation_matrix_to_euler_xyz(
            previous_rotation.as_matrix().astype(np.float32, copy=False)
        )

    smoothed[:, 6] = states[:, 6]
    return smoothed.astype(np.float32, copy=False)


def smooth_pika_pose_state_sequence(
    states: np.ndarray,
    alpha: float = 0.25,
    mode: str = "causal",
) -> np.ndarray:
    alpha = validate_smoothing_alpha(alpha)
    mode = validate_pika_pose_smoothing_mode(mode)
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 7:
        raise ValueError(f"Expected states to have shape (num_frames, 7), got {states.shape}")
    if len(states) == 0:
        return states.astype(np.float32, copy=True)

    rotation_cls, slerp_cls = _require_scipy_rotation()
    smoothed = _smooth_pika_pose_state_sequence_causal(states, alpha, rotation_cls, slerp_cls)
    if mode == "zero_phase":
        smoothed = _smooth_pika_pose_state_sequence_causal(smoothed[::-1], alpha, rotation_cls, slerp_cls)[::-1]
        smoothed[:, 6] = states[:, 6]
    return smoothed.astype(np.float32, copy=False)


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
    return np.concatenate([pose, np.array([gripper_distance], dtype=np.float32)]).astype(np.float32)


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
        instruction_keys = (
            "instruction",
            "instructions",
            "task",
            "tasks",
            "full-instructions",
            "segment-instructions",
        )
        for key in instruction_keys:
            if key in payload:
                values.extend(_extract_instruction_strings(payload[key]))
        return values
    return []


def parse_cameras(cameras: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if cameras is None:
        return DEFAULT_CAMERAS

    if isinstance(cameras, str):
        raw_items = [item.strip() for item in cameras.split(",")]
    else:
        raw_items = [str(item).strip() for item in cameras]

    selected = tuple(item for item in raw_items if item)
    if len(selected) == 1 and selected[0].lower() == "none":
        return ()

    invalid = sorted(set(selected) - set(SUPPORTED_CAMERAS))
    if invalid:
        raise ValueError(f"Unsupported camera(s): {invalid}. Supported cameras: {sorted(SUPPORTED_CAMERAS)}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate cameras are not supported: {selected}")
    return selected


def infer_features(
    modality_files: dict[str, list[Path]],
    cameras: str | list[str] | tuple[str, ...] | None = DEFAULT_CAMERAS,
    camera_storage: str = "image",
) -> dict[str, dict[str, Any]]:
    selected_cameras = parse_cameras(cameras)
    if camera_storage not in CAMERA_STORAGE_CHOICES:
        raise ValueError(
            f"Unsupported camera storage {camera_storage!r}. Choose from {CAMERA_STORAGE_CHOICES}."
        )

    features: dict[str, dict[str, Any]] = {}
    for camera in selected_cameras:
        if camera in RGB_CAMERA_FEATURES:
            features[RGB_CAMERA_FEATURES[camera]] = {
                "dtype": camera_storage,
                "shape": _get_rgb_feature_shape(modality_files[camera][0]),
                "names": None,
            }
        elif camera == DEPTH_CAMERA:
            depth_array = load_depth_image(modality_files[camera][0])
            features[DEPTH_CAMERA_FEATURE] = {
                "dtype": str(depth_array.dtype),
                "shape": depth_array.shape,
                "names": None,
            }

    features.update(
        {
            "observation.state": {"dtype": "float32", "shape": (7,), "names": POSE7D_NAMES},
            "action": {"dtype": "float32", "shape": (7,), "names": POSE7D_NAMES},
        }
    )
    return features


def camera_features_use_video(features: dict[str, dict[str, Any]]) -> bool:
    return any(feature["dtype"] == "video" for feature in features.values())


def conversion_uses_streaming_encoding(
    features: dict[str, dict[str, Any]],
    streaming_encoding: bool | None,
) -> bool:
    use_videos = camera_features_use_video(features)
    if streaming_encoding is None:
        return use_videos
    return bool(streaming_encoding) and use_videos


def create_conversion_dataset(
    repo_id: str,
    fps: int,
    features: dict[str, dict[str, Any]],
    output_root: Path,
    vcodec: str = DEFAULT_VIDEO_CODEC,
    streaming_encoding: bool | None = None,
) -> Any:
    from lerobot.datasets import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_root,
        use_videos=camera_features_use_video(features),
        image_writer_processes=CONVERSION_IMAGE_WRITER_PROCESSES,
        image_writer_threads=CONVERSION_IMAGE_WRITER_THREADS,
        vcodec=vcodec,
        streaming_encoding=conversion_uses_streaming_encoding(features, streaming_encoding),
        encoder_queue_maxsize=CONVERSION_ENCODER_QUEUE_MAXSIZE,
        encoder_threads=CONVERSION_ENCODER_THREADS,
        streaming_drop_frames=False,
    )


def add_camera_frame_data(
    frame: dict[str, Any],
    modality_files: dict[str, list[Path]],
    cameras: tuple[str, ...],
    frame_idx: int,
) -> None:
    for camera in cameras:
        if camera in RGB_CAMERA_FEATURES:
            frame[RGB_CAMERA_FEATURES[camera]] = load_rgb_image(modality_files[camera][frame_idx])
        elif camera == DEPTH_CAMERA:
            frame[DEPTH_CAMERA_FEATURE] = load_depth_image(modality_files[camera][frame_idx])


def discover_episode_files(
    episode_dir: Path,
    cameras: str | list[str] | tuple[str, ...] | None = DEFAULT_CAMERAS,
) -> dict[str, list[Path]]:
    selected_cameras = parse_cameras(cameras)
    files = {
        "pose": load_synced_files(episode_dir / POSE_DIR),
        "gripper": load_synced_files(episode_dir / GRIPPER_DIR),
    }

    for camera in selected_cameras:
        if camera in RGB_CAMERA_DIRS:
            files[camera] = load_synced_files(episode_dir / RGB_CAMERA_DIRS[camera])
        elif camera == DEPTH_CAMERA:
            files[camera] = load_synced_files(episode_dir / DEPTH_DIR)

    return files


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


def parse_frame_range(frame_range: tuple[float, float] | list[float] | None) -> tuple[float, float]:
    if frame_range is None:
        return (0.0, 1.0)
    if len(frame_range) != 2:
        raise ValueError(f"Expected frame range to contain START and END, got {frame_range}")

    start, end = float(frame_range[0]), float(frame_range[1])
    if not 0.0 <= start < end <= 1.0:
        raise ValueError(f"Frame range must satisfy 0.0 <= START < END <= 1.0, got {(start, end)}")
    return (start, end)


def resolve_frame_slice(
    total_frames: int,
    frame_range: tuple[float, float] | list[float] | None,
) -> tuple[int, int]:
    start_ratio, end_ratio = parse_frame_range(frame_range)
    start_idx = math.floor(total_frames * start_ratio)
    end_idx = math.floor(total_frames * end_ratio)
    if end_idx <= start_idx:
        raise ValueError(
            f"Frame range {(start_ratio, end_ratio)} selects no frames from aligned length {total_frames}"
        )
    return start_idx, end_idx


def add_episode_to_dataset(
    dataset: Any,
    input_episode: Path,
    task: str | None = None,
    cameras: str | list[str] | tuple[str, ...] | None = DEFAULT_CAMERAS,
    camera_storage: str = "image",
    frame_range: tuple[float, float] | list[float] | None = None,
    smooth_pika_pose: bool = False,
    smooth_pika_pose_alpha: float = 0.25,
    smooth_pika_pose_mode: str = "causal",
) -> dict[str, Any]:
    input_episode = Path(input_episode)
    selected_cameras = parse_cameras(cameras)
    modality_files = discover_episode_files(input_episode, cameras=selected_cameras)
    aligned_length = compute_aligned_length(modality_files)
    source_frame_start, source_frame_end = resolve_frame_slice(aligned_length, frame_range)
    selected_frame_count = source_frame_end - source_frame_start
    parsed_frame_range = parse_frame_range(frame_range)
    task_text = get_episode_task(input_episode / "instructions.json", task_override=task)

    logging.info(
        "Detected modality counts for %s: %s",
        input_episode,
        {name: len(files) for name, files in modality_files.items()},
    )
    logging.info("Using aligned frame count: %s", aligned_length)
    logging.info("Using source frame range: [%s, %s)", source_frame_start, source_frame_end)

    states = np.stack(
        [
            build_state_vector(modality_files["pose"][frame_idx], modality_files["gripper"][frame_idx])
            for frame_idx in range(source_frame_start, source_frame_end)
        ]
    )
    if smooth_pika_pose:
        logging.info(
            "Smoothing Pika pose with alpha=%s, mode=%s",
            smooth_pika_pose_alpha,
            smooth_pika_pose_mode,
        )
        states = smooth_pika_pose_state_sequence(
            states,
            alpha=smooth_pika_pose_alpha,
            mode=smooth_pika_pose_mode,
        )

    for state_idx, frame_idx in enumerate(range(source_frame_start, source_frame_end)):
        state = states[state_idx]
        frame = {
            "task": task_text,
            "observation.state": state,
            "action": state.copy(),
        }
        add_camera_frame_data(frame, modality_files, selected_cameras, frame_idx)
        dataset.add_frame(frame)

    dataset.save_episode()

    return {
        "input_episode": str(input_episode),
        "aligned_frames": aligned_length,
        "selected_frames": selected_frame_count,
        "frame_range": parsed_frame_range,
        "source_frame_start": source_frame_start,
        "source_frame_end": source_frame_end,
        "task": task_text,
        "cameras": selected_cameras,
        "camera_storage": camera_storage,
        "smooth_pika_pose": smooth_pika_pose,
        "smooth_pika_pose_alpha": float(smooth_pika_pose_alpha),
        "smooth_pika_pose_mode": smooth_pika_pose_mode,
    }


def convert_episode(
    input_episode: Path,
    output_root: Path,
    repo_id: str = DEFAULT_REPO_ID,
    fps: int = 30,
    task: str | None = None,
    use_videos: bool = False,
    overwrite: bool = False,
    cameras: str | list[str] | tuple[str, ...] | None = DEFAULT_CAMERAS,
    camera_storage: str = "image",
    frame_range: tuple[float, float] | list[float] | None = None,
    smooth_pika_pose: bool = False,
    smooth_pika_pose_alpha: float = 0.25,
    smooth_pika_pose_mode: str = "causal",
    vcodec: str = DEFAULT_VIDEO_CODEC,
    streaming_encoding: bool | None = None,
) -> dict[str, Any]:
    input_episode = Path(input_episode)
    output_root = Path(output_root)
    selected_cameras = parse_cameras(cameras)
    if use_videos:
        camera_storage = "video"

    modality_files = discover_episode_files(input_episode, cameras=selected_cameras)
    features = infer_features(modality_files, cameras=selected_cameras, camera_storage=camera_storage)

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    dataset = create_conversion_dataset(
        repo_id=repo_id,
        fps=fps,
        features=features,
        output_root=output_root,
        vcodec=vcodec,
        streaming_encoding=streaming_encoding,
    )

    try:
        episode_manifest = add_episode_to_dataset(
            dataset,
            input_episode=input_episode,
            task=task,
            cameras=selected_cameras,
            camera_storage=camera_storage,
            frame_range=frame_range,
            smooth_pika_pose=smooth_pika_pose,
            smooth_pika_pose_alpha=smooth_pika_pose_alpha,
            smooth_pika_pose_mode=smooth_pika_pose_mode,
        )
        dataset.finalize()
    except Exception:
        dataset.finalize()
        raise

    return {
        "repo_id": repo_id,
        "input_episode": str(input_episode),
        "output_root": str(output_root),
        "fps": fps,
        **episode_manifest,
        "vcodec": vcodec,
        "streaming_encoding": conversion_uses_streaming_encoding(features, streaming_encoding),
        "features": features,
    }


def discover_episode_dirs(input_root: Path) -> list[Path]:
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    episode_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not episode_dirs:
        raise ValueError(f"No episode subdirectories found under: {input_root}")
    return episode_dirs


def convert_episodes(
    input_root: Path,
    output_root: Path,
    repo_id: str = DEFAULT_REPO_ID,
    fps: int = 30,
    task: str | None = None,
    use_videos: bool = False,
    overwrite: bool = False,
    cameras: str | list[str] | tuple[str, ...] | None = DEFAULT_CAMERAS,
    camera_storage: str = "image",
    frame_range: tuple[float, float] | list[float] | None = None,
    smooth_pika_pose: bool = False,
    smooth_pika_pose_alpha: float = 0.25,
    smooth_pika_pose_mode: str = "causal",
    vcodec: str = DEFAULT_VIDEO_CODEC,
    streaming_encoding: bool | None = None,
) -> dict[str, Any]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    selected_cameras = parse_cameras(cameras)
    if use_videos:
        camera_storage = "video"

    episode_dirs = discover_episode_dirs(input_root)
    first_modality_files = discover_episode_files(episode_dirs[0], cameras=selected_cameras)
    features = infer_features(first_modality_files, cameras=selected_cameras, camera_storage=camera_storage)

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    dataset = create_conversion_dataset(
        repo_id=repo_id,
        fps=fps,
        features=features,
        output_root=output_root,
        vcodec=vcodec,
        streaming_encoding=streaming_encoding,
    )

    episode_manifests: list[dict[str, Any]] = []
    try:
        for episode_dir in episode_dirs:
            logging.info("Converting episode: %s", episode_dir)
            episode_manifests.append(
                add_episode_to_dataset(
                    dataset,
                    input_episode=episode_dir,
                    task=task,
                    cameras=selected_cameras,
                    camera_storage=camera_storage,
                    frame_range=frame_range,
                    smooth_pika_pose=smooth_pika_pose,
                    smooth_pika_pose_alpha=smooth_pika_pose_alpha,
                    smooth_pika_pose_mode=smooth_pika_pose_mode,
                )
            )
        dataset.finalize()
    except Exception:
        dataset.finalize()
        raise

    return {
        "repo_id": repo_id,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "fps": fps,
        "total_episodes": len(episode_manifests),
        "total_selected_frames": sum(item["selected_frames"] for item in episode_manifests),
        "cameras": selected_cameras,
        "camera_storage": camera_storage,
        "vcodec": vcodec,
        "streaming_encoding": conversion_uses_streaming_encoding(features, streaming_encoding),
        "smooth_pika_pose": smooth_pika_pose,
        "smooth_pika_pose_alpha": float(smooth_pika_pose_alpha),
        "smooth_pika_pose_mode": smooth_pika_pose_mode,
        "features": features,
        "episodes": episode_manifests,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory whose direct child directories are UMI episodes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for the local LeRobotDataset v3.0 dataset.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Repo id recorded in the generated local LeRobot dataset metadata.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS recorded in metadata.")
    parser.add_argument("--task", help="Optional task text override.")
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help=(
            "Convert only the half-open fractional frame range [START, END), "
            "where 0.0 <= START < END <= 1.0."
        ),
    )
    parser.add_argument(
        "--cameras",
        default=",".join(DEFAULT_CAMERAS),
        help=(
            f"Comma-separated cameras to convert. Supported: {', '.join(SUPPORTED_CAMERAS)}. "
            "Use 'none' for no cameras."
        ),
    )
    parser.add_argument(
        "--camera-storage",
        choices=CAMERA_STORAGE_CHOICES,
        default="image",
        help="Store RGB camera observations as parquet images or encoded videos. Depth stays a uint16 array.",
    )
    parser.add_argument(
        "--vcodec",
        choices=VIDEO_CODEC_CHOICES,
        default=DEFAULT_VIDEO_CODEC,
        help="Video codec for RGB camera video storage. Use 'auto' to prefer available hardware encoding.",
    )
    parser.add_argument(
        "--no-streaming-encoding",
        action="store_true",
        help="Disable direct streaming video encoding and use the slower temporary-image encoding path.",
    )
    parser.add_argument(
        "--use-videos",
        action="store_true",
        help="Deprecated alias for --camera-storage video.",
    )
    parser.add_argument(
        "--smooth-pika-pose",
        action="store_true",
        help="Smooth localization/pose/pika with position EMA and quaternion Slerp rotation smoothing.",
    )
    parser.add_argument(
        "--smooth-pika-pose-alpha",
        type=float,
        default=0.25,
        help="Smoothing factor for --smooth-pika-pose. Must satisfy 0.0 < alpha <= 1.0.",
    )
    parser.add_argument(
        "--smooth-pika-pose-mode",
        choices=PIKA_POSE_SMOOTHING_MODE_CHOICES,
        default="causal",
        help="Use causal online-style smoothing or offline forward-backward zero_phase smoothing.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dataset root.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    manifest = convert_episodes(
        input_root=args.input_root,
        output_root=args.output_root,
        repo_id=args.repo_id,
        fps=args.fps,
        task=args.task,
        use_videos=args.use_videos,
        overwrite=args.overwrite,
        cameras=args.cameras,
        camera_storage=args.camera_storage,
        frame_range=args.frame_range,
        smooth_pika_pose=args.smooth_pika_pose,
        smooth_pika_pose_alpha=args.smooth_pika_pose_alpha,
        smooth_pika_pose_mode=args.smooth_pika_pose_mode,
        vcodec=args.vcodec,
        streaming_encoding=not args.no_streaming_encoding,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
