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

"""Extract grasp segments from a LeRobot dataset.

The grasp starts when the gripper closes and ends when it opens again, unless
``duration_s`` is set. When set to two values, it is interpreted as a time
window relative to the grasp start. For example, ``"-0.5,2"`` keeps 0.5 seconds
before the grasp and 2 seconds after it. The script uses the ``gripper_width``
dimension in ``observation.state`` and emits warnings for suspicious detections.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.utils.utils import init_logging

MIN_GRIPPER_RANGE = 0.02
MIN_STABLE_CLOSED_S = 0.5
START_FRACTION_RANGE = (0.10, 0.65)
END_FRACTION_RANGE = (0.55, 0.95)

METADATA_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


@dataclass
class GraspWarning:
    message: str
    risky: bool = False


@dataclass
class GraspSlice:
    start: int | None
    end: int | None
    warnings: list[GraspWarning] = field(default_factory=list)
    skip: bool = False

    @property
    def risky(self) -> bool:
        return any(w.risky for w in self.warnings)


@dataclass
class ExtractGraspDatasetConfig:
    input_root: Path
    output_root: Path
    output_repo_id: str | None = None
    duration_s: str | None = None
    video_keys: list[str] = field(default_factory=lambda: ["observation.images.fisheye_rgb"])
    strict: bool = True
    dry_run: bool = False


def _closed_intervals(closed: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], closed, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=False))


def _parse_duration_window(duration_s: float | Sequence[float] | str | None) -> tuple[float, float] | None:
    if duration_s is None:
        return None

    if isinstance(duration_s, str):
        duration_str = duration_s.strip()
        if not duration_str:
            return None
        duration_str = duration_str.removeprefix("[").removesuffix("]")
        parts = [part.strip() for part in duration_str.replace(",", " ").split()]
        durations = [float(part) for part in parts]
    elif isinstance(duration_s, Sequence):
        durations = [float(value) for value in duration_s]
    else:
        durations = [float(duration_s)]

    if len(durations) == 1:
        before_s = 0.0
        after_s = durations[0]
    elif len(durations) == 2:
        before_s, after_s = durations
    else:
        raise ValueError("duration_s must be either one value or two values")

    if before_s > 0:
        raise ValueError(f"duration_s start offset must be <= 0, got {before_s}")
    if after_s <= 0:
        raise ValueError(f"duration_s end offset must be > 0, got {after_s}")
    if after_s <= before_s:
        raise ValueError(f"duration_s end offset must be greater than start offset, got {durations}")

    return before_s, after_s


def find_grasp_slice(
    gripper_width: np.ndarray,
    fps: int,
    duration_s: float | Sequence[float] | str | None = None,
) -> GraspSlice:
    if len(gripper_width) == 0:
        return GraspSlice(None, None, [GraspWarning("empty episode", risky=True)], skip=True)

    warnings: list[GraspWarning] = []
    min_width = float(np.min(gripper_width))
    max_width = float(np.max(gripper_width))
    gripper_range = max_width - min_width
    if gripper_range < MIN_GRIPPER_RANGE:
        return GraspSlice(
            None,
            None,
            [
                GraspWarning(
                    f"gripper range is too small ({gripper_range:.4f} < {MIN_GRIPPER_RANGE:.4f})",
                    risky=True,
                )
            ],
            skip=True,
        )

    threshold = (min_width + max_width) / 2
    intervals = _closed_intervals(gripper_width < threshold)
    if not intervals:
        return GraspSlice(None, None, [GraspWarning("no closed gripper interval found", risky=True)], skip=True)

    min_stable_frames = max(1, int(round(MIN_STABLE_CLOSED_S * fps)))
    stable_intervals = [(start, end) for start, end in intervals if end - start >= min_stable_frames]
    if not stable_intervals:
        longest_start, longest_end = max(intervals, key=lambda item: item[1] - item[0])
        return GraspSlice(
            None,
            None,
            [
                GraspWarning(
                    "closed gripper interval is too short "
                    f"({longest_end - longest_start} frames < {min_stable_frames} frames)",
                    risky=True,
                )
            ],
            skip=True,
        )

    if len(stable_intervals) > 1:
        warnings.append(
            GraspWarning(
                f"multiple stable closed intervals found ({len(stable_intervals)}); using the longest one",
                risky=True,
            )
        )

    grasp_start, default_end = max(stable_intervals, key=lambda item: item[1] - item[0])
    start = grasp_start
    end = default_end

    duration_window = _parse_duration_window(duration_s)
    if duration_window is not None:
        before_s, after_s = duration_window
        requested_start = grasp_start + int(round(before_s * fps))
        requested_end = grasp_start + int(round(after_s * fps))
        end = min(requested_end, len(gripper_width))
        if requested_start < 0:
            warnings.append(
                GraspWarning(
                    f"duration_s starts before episode beginning; clipped from frame {requested_start} to 0",
                    risky=False,
                )
            )
        start = max(requested_start, 0)
        if requested_end > len(gripper_width):
            warnings.append(
                GraspWarning(
                    f"duration_s extends past episode end; clipped from frame {requested_end} to {end}",
                    risky=False,
                )
            )

    start_fraction = grasp_start / len(gripper_width)
    end_fraction = default_end / len(gripper_width)
    if not START_FRACTION_RANGE[0] <= start_fraction <= START_FRACTION_RANGE[1]:
        warnings.append(
            GraspWarning(
                "closed interval starts outside expected middle region "
                f"({start_fraction:.3f} not in {START_FRACTION_RANGE})",
                risky=True,
            )
        )
    if not END_FRACTION_RANGE[0] <= end_fraction <= END_FRACTION_RANGE[1]:
        warnings.append(
            GraspWarning(
                "closed interval ends outside expected region "
                f"({end_fraction:.3f} not in {END_FRACTION_RANGE})",
                risky=True,
            )
        )

    if end <= start:
        return GraspSlice(None, None, [GraspWarning("computed empty grasp slice", risky=True)], skip=True)

    return GraspSlice(start, end, warnings)


def _get_gripper_width_index(dataset: LeRobotDataset) -> int:
    state_feature = dataset.meta.features.get("observation.state")
    if state_feature is None:
        raise ValueError("Dataset has no 'observation.state' feature")

    names = state_feature.get("names")
    if not names or "gripper_width" not in names:
        raise ValueError("'observation.state' feature must include a 'gripper_width' dimension name")

    return names.index("gripper_width")


def _filter_features(features: dict[str, dict], video_keys: list[str]) -> dict[str, dict]:
    for key in video_keys:
        if key not in features:
            raise ValueError(f"Requested video key '{key}' is not in dataset features")
        if features[key]["dtype"] not in {"video", "depth_video", "image", "depth_image"}:
            raise ValueError(f"Requested video key '{key}' is not a visual feature")

    visual_dtypes = {"video", "depth_video", "image", "depth_image"}
    return {
        key: feature
        for key, feature in features.items()
        if feature["dtype"] not in visual_dtypes or key in video_keys
    }


def _load_data_frame(dataset: LeRobotDataset) -> pd.DataFrame:
    frames = []
    for data_file in sorted((dataset.root / "data").glob("chunk-*/file-*.parquet")):
        frames.append(pd.read_parquet(data_file))
    if not frames:
        raise ValueError(f"No parquet data files found under {dataset.root / 'data'}")
    return pd.concat(frames, ignore_index=True)


def _frame_for_writer(dataset: LeRobotDataset, item: dict) -> dict:
    frame = {"task": item["task"]}
    for key, feature in dataset.meta.features.items():
        if key in METADATA_KEYS:
            continue
        if feature["dtype"] in {"video", "depth_video", "image", "depth_image"} or key in item:
            frame[key] = item[key]
    return frame


def extract_grasp_dataset(cfg: ExtractGraspDatasetConfig) -> None:
    input_root = Path(cfg.input_root)
    output_root = Path(cfg.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset root does not exist: {input_root}")
    if output_root.exists() and not cfg.dry_run:
        raise FileExistsError(f"Output root already exists: {output_root}")

    input_repo_id = input_root.name
    output_repo_id = cfg.output_repo_id or output_root.name

    source = LeRobotDataset(repo_id=input_repo_id, root=input_root)
    gripper_width_index = _get_gripper_width_index(source)
    data = _load_data_frame(source)
    output_features = _filter_features(source.meta.features, cfg.video_keys)
    source.meta.info["features"] = output_features

    output = None
    if not cfg.dry_run:
        output = LeRobotDataset.create(
            repo_id=output_repo_id,
            fps=source.meta.fps,
            features=output_features,
            root=output_root,
            robot_type=source.meta.robot_type,
            use_videos=len(source.meta.video_storage_keys) > 0,
        )

    total = 0
    written = 0
    skipped = 0
    warning_count = 0

    for episode_index, episode_df in data.groupby("episode_index", sort=True):
        total += 1
        gripper_width = np.stack(episode_df["observation.state"].to_numpy())[:, gripper_width_index]
        grasp = find_grasp_slice(gripper_width, fps=source.meta.fps, duration_s=cfg.duration_s)
        warning_count += len(grasp.warnings)

        prefix = f"episode {int(episode_index)}"
        for warning in grasp.warnings:
            logging.warning("%s: %s", prefix, warning.message)

        should_skip = grasp.skip or (cfg.strict and grasp.risky)
        if should_skip:
            skipped += 1
            reason = "invalid detection" if grasp.skip else "risky detection in strict mode"
            logging.warning("%s: skipped (%s)", prefix, reason)
            continue

        assert grasp.start is not None
        assert grasp.end is not None

        from_index = int(episode_df.iloc[grasp.start]["index"])
        to_index = int(episode_df.iloc[grasp.end - 1]["index"]) + 1
        logging.info(
            "%s: selected local frames [%d, %d), dataset indices [%d, %d), %d frames",
            prefix,
            grasp.start,
            grasp.end,
            from_index,
            to_index,
            grasp.end - grasp.start,
        )

        if output is None:
            continue

        for dataset_index in range(from_index, to_index):
            output.add_frame(_frame_for_writer(source, source[dataset_index]))
        output.save_episode()
        written += 1

    if output is not None:
        output.finalize()

    logging.info(
        "Finished extracting grasp dataset: total=%d written=%d skipped=%d warnings=%d dry_run=%s",
        total,
        written,
        skipped,
        warning_count,
        cfg.dry_run,
    )


@parser.wrap()
def run_extract_grasp_dataset(cfg: ExtractGraspDatasetConfig) -> None:
    extract_grasp_dataset(cfg)


def main() -> None:
    init_logging()
    run_extract_grasp_dataset()


if __name__ == "__main__":
    main()
