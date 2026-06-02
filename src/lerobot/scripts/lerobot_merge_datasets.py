#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Merge compatible local LeRobot datasets into one dataset."""

import argparse
import logging
import shutil
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.utils.utils import init_logging


def _repo_id_from_root(root: Path) -> str:
    return root.name


def _metadata_signature(meta: LeRobotDatasetMetadata) -> dict:
    return {
        "fps": meta.fps,
        "robot_type": meta.robot_type,
        "features": meta.features,
        "data_path": meta.data_path,
        "video_path": meta.video_path,
        "depth_video_path": meta.depth_video_path,
        "image_keys": meta.image_keys,
        "depth_image_keys": meta.depth_image_keys,
        "video_keys": meta.video_keys,
        "depth_video_keys": meta.depth_video_keys,
        "camera_keys": meta.camera_keys,
        "video_storage_keys": meta.video_storage_keys,
    }


def _format_difference(name: str, expected, actual) -> str:
    return f"{name}: expected {expected!r}, got {actual!r}"


def validate_compatible(metadata: list[LeRobotDatasetMetadata]) -> None:
    if len(metadata) < 2:
        raise ValueError("At least two datasets are required for merge.")

    expected = _metadata_signature(metadata[0])
    errors: list[str] = []

    for meta in metadata[1:]:
        actual = _metadata_signature(meta)
        for key, expected_value in expected.items():
            if actual[key] != expected_value:
                errors.append(f"{meta.root}: {_format_difference(key, expected_value, actual[key])}")

    if errors:
        for error in errors:
            logging.warning(error)
        raise ValueError("Datasets are not compatible; aborting merge.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="+",
        type=Path,
        help="Local dataset roots to merge. Each root must contain meta/, data/, and optional videos/.",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="Repo id written to the merged metadata. Defaults to output root directory name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove output-root first if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = parse_args()

    roots = [root.expanduser().resolve() for root in args.datasets]
    output_root = args.output_root.expanduser().resolve()
    output_repo_id = args.output_repo_id or output_root.name

    if len(roots) < 2:
        raise ValueError("At least two datasets are required for merge.")
    if len(set(roots)) != len(roots):
        raise ValueError("Input datasets must be unique.")
    for root in roots:
        if not (root / "meta").is_dir() or not (root / "data").is_dir():
            raise FileNotFoundError(f"{root} is not a local LeRobot dataset root.")

    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"{output_root} already exists. Use --force to overwrite it.")
        shutil.rmtree(output_root)

    repo_ids = [_repo_id_from_root(root) for root in roots]
    metadata = [
        LeRobotDatasetMetadata(repo_id=repo_id, root=root)
        for repo_id, root in zip(repo_ids, roots, strict=True)
    ]

    validate_compatible(metadata)

    logging.info("Merging %d datasets into %s", len(roots), output_root)
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=output_repo_id,
        roots=roots,
        aggr_root=output_root,
    )
    merged = LeRobotDatasetMetadata(output_repo_id, root=output_root)
    logging.info(
        "Merged dataset saved to %s (%d episodes, %d frames)",
        output_root,
        merged.total_episodes,
        merged.total_frames,
    )


if __name__ == "__main__":
    main()
