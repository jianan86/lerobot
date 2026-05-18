#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Profile LeRobotDataset read-side latency by pipeline stage.

This benchmark is intended for diagnosing GPU-waits-for-CPU training runs. It
does not modify the dataset on disk. Feature variants are applied in memory so
the same local dataset can be compared as metadata/parquet-only, RGB-only,
depth-only, and full visual loading.

Example:

python benchmarks/dataset/run_dataset_loading_benchmark.py \
    --repo-id local/lerobot_0511_depth_video \
    --root /home/jianan/workspace/data/lerobot_0511_depth_video \
    --tolerance-s 0.02 \
    --num-samples 8 \
    --num-workers 0 4
"""

import argparse
import csv
import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import (
    _default_decoder_cache,
    _default_depth_decoder_cache,
    decode_depth_video_frames,
    decode_video_frames,
)

DEFAULT_VARIANTS = ["metadata", "rgb", "depth", "full"]


def parse_int_list(values: list[str] | None) -> list[int] | None:
    if values is None:
        return None
    return [int(value) for value in values]


def summarize_seconds(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}

    ordered = sorted(values)

    def percentile(q: float) -> float:
        idx = min(len(ordered) - 1, round((len(ordered) - 1) * q))
        return ordered[idx] * 1000

    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values) * 1000,
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "max_ms": max(values) * 1000,
    }


def format_ms(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def visual_storage_keys(dataset: LeRobotDataset) -> list[str]:
    return dataset.meta.video_keys + dataset.meta.depth_video_keys


def filter_features_for_variant(dataset: LeRobotDataset, variant: str) -> None:
    """Apply a feature subset to dataset metadata in memory."""
    if variant == "full":
        return

    def keep(key: str, feature: dict[str, Any]) -> bool:
        dtype = feature["dtype"]
        if variant == "metadata":
            return dtype not in {"image", "video", "depth_image", "depth_video"}
        if variant == "rgb":
            return dtype != "depth_video" and not key.startswith("observation.depth.")
        if variant == "depth":
            return dtype != "video"
        raise ValueError(f"Unsupported variant: {variant}")

    dataset.meta.info["features"] = {
        key: feature for key, feature in dataset.meta.info["features"].items() if keep(key, feature)
    }


def make_delta_timestamps(dataset: LeRobotDataset, observation_delta_indices: list[int] | None) -> dict:
    if observation_delta_indices is None:
        return {}
    return {
        key: [index / dataset.meta.fps for index in observation_delta_indices]
        for key in dataset.meta.features
        if key.startswith("observation.")
    }


def make_dataset(args: argparse.Namespace, variant: str, delta_timestamps: dict | None = None) -> LeRobotDataset:
    _default_decoder_cache.clear()
    _default_depth_decoder_cache.clear()
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        episodes=args.episodes,
        delta_timestamps=delta_timestamps,
        tolerance_s=args.tolerance_s,
        revision=args.revision,
        video_backend=args.video_backend,
        return_uint8=args.return_uint8,
    )
    filter_features_for_variant(dataset, variant)
    return dataset


def choose_indices(dataset: LeRobotDataset, num_samples: int, seed: int, mode: str) -> list[int]:
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if mode == "head":
        return list(range(min(num_samples, len(dataset))))
    if mode == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(range(len(dataset)), k=min(num_samples, len(dataset))))
    raise ValueError(f"Unsupported sample mode: {mode}")


def time_stage(rows: list[dict], variant: str, stage: str, seconds: list[float], feature_key: str = "") -> None:
    summary = summarize_seconds(seconds)
    rows.append(
        {
            "kind": "stage",
            "variant": variant,
            "feature_key": feature_key,
            "stage": stage,
            "count": summary["count"],
            "mean_ms": summary["mean_ms"],
            "p50_ms": summary["p50_ms"],
            "p95_ms": summary["p95_ms"],
            "max_ms": summary["max_ms"],
            "num_workers": "",
            "batch_size": "",
            "frames": "",
            "seconds": "",
            "fps": "",
        }
    )


def profile_getitem_stages(dataset: LeRobotDataset, variant: str, indices: Iterable[int]) -> list[dict]:
    reader = dataset.reader
    if reader is None:
        raise RuntimeError("Dataset reader is not initialized.")

    timings: dict[str, list[float]] = {
        "current_row": [],
        "delta_indices": [],
        "delta_hf_dataset": [],
        "query_timestamps": [],
        "video_decode_total": [],
        "depth_image_load": [],
        "image_transforms": [],
        "task_lookup": [],
        "getitem_total": [],
    }
    per_key_decode: dict[str, list[float]] = {key: [] for key in visual_storage_keys(dataset)}

    for idx in indices:
        total_start = time.perf_counter()

        start = time.perf_counter()
        item = reader._get_current_item(idx)
        timings["current_row"].append(time.perf_counter() - start)

        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()
        query_indices = None

        start = time.perf_counter()
        if reader.delta_indices is not None:
            query_indices, padding = reader._get_query_indices(abs_idx, ep_idx)
            item = {**item, **padding}
        timings["delta_indices"].append(time.perf_counter() - start)

        start = time.perf_counter()
        if query_indices is not None:
            item.update(reader._query_hf_dataset(query_indices))
        timings["delta_hf_dataset"].append(time.perf_counter() - start)

        start = time.perf_counter()
        current_ts = item["timestamp"].item()
        query_timestamps = reader._get_query_timestamps(current_ts, query_indices)
        timings["query_timestamps"].append(time.perf_counter() - start)

        start_videos = time.perf_counter()
        ep = reader._meta.episodes[ep_idx]
        for key, timestamps in query_timestamps.items():
            start = time.perf_counter()
            from_timestamp = ep[f"videos/{key}/from_timestamp"]
            shifted_timestamps = [from_timestamp + ts for ts in timestamps]
            video_path = reader.root / reader._meta.get_video_file_path(ep_idx, key)
            if key in reader._meta.depth_video_keys:
                frames = decode_depth_video_frames(video_path, shifted_timestamps, reader._tolerance_s)
            else:
                frames = decode_video_frames(
                    video_path,
                    shifted_timestamps,
                    reader._tolerance_s,
                    reader._video_backend,
                    return_uint8=reader._return_uint8,
                )
            item[key] = frames.squeeze(0)
            per_key_decode.setdefault(key, []).append(time.perf_counter() - start)
        timings["video_decode_total"].append(time.perf_counter() - start_videos)

        start = time.perf_counter()
        for key in reader._meta.depth_image_keys:
            if key in item and isinstance(item[key], str):
                item[key] = reader._load_depth_image(item[key])
        timings["depth_image_load"].append(time.perf_counter() - start)

        start = time.perf_counter()
        if reader._image_transforms is not None:
            for cam in reader._meta.camera_keys:
                item[cam] = reader._image_transforms(item[cam])
        timings["image_transforms"].append(time.perf_counter() - start)

        start = time.perf_counter()
        task_idx = item["task_index"].item()
        item["task"] = reader._meta.tasks.iloc[task_idx].name
        timings["task_lookup"].append(time.perf_counter() - start)

        timings["getitem_total"].append(time.perf_counter() - total_start)

    rows = []
    for stage, seconds in timings.items():
        time_stage(rows, variant, stage, seconds)
    for key, seconds in per_key_decode.items():
        time_stage(rows, variant, "video_decode", seconds, feature_key=key)
    return rows


def profile_dataloader(dataset: LeRobotDataset, variant: str, args: argparse.Namespace, num_workers: int) -> dict:
    _default_decoder_cache.clear()
    _default_depth_decoder_cache.clear()
    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "shuffle": False,
        "pin_memory": args.pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = args.persistent_workers

    dataloader = torch.utils.data.DataLoader(dataset, **kwargs)
    iterator = iter(dataloader)

    for _ in range(args.warmup_batches):
        next(iterator)

    start = time.perf_counter()
    frames = 0
    for _ in range(args.measure_batches):
        batch = next(iterator)
        frames += len(batch["index"])
    seconds = time.perf_counter() - start
    fps = frames / seconds if seconds > 0 else float("inf")

    return {
        "kind": "dataloader",
        "variant": variant,
        "feature_key": "",
        "stage": "dataloader",
        "count": "",
        "mean_ms": "",
        "p50_ms": "",
        "p95_ms": "",
        "max_ms": "",
        "num_workers": num_workers,
        "batch_size": args.batch_size,
        "frames": frames,
        "seconds": seconds,
        "fps": fps,
    }


def print_rows(rows: list[dict]) -> None:
    print("\nStage timings")
    header = ("variant", "stage", "feature_key", "count", "mean_ms", "p50_ms", "p95_ms", "max_ms")
    print(" | ".join(header))
    print(" | ".join(["-" * len(item) for item in header]))
    for row in rows:
        if row["kind"] != "stage":
            continue
        print(
            " | ".join(
                [
                    str(row["variant"]),
                    str(row["stage"]),
                    str(row["feature_key"]),
                    str(row["count"]),
                    format_ms(row["mean_ms"]),
                    format_ms(row["p50_ms"]),
                    format_ms(row["p95_ms"]),
                    format_ms(row["max_ms"]),
                ]
            )
        )

    dataloader_rows = [row for row in rows if row["kind"] == "dataloader"]
    if not dataloader_rows:
        return

    print("\nDataLoader throughput")
    header = ("variant", "num_workers", "batch_size", "frames", "seconds", "fps")
    print(" | ".join(header))
    print(" | ".join(["-" * len(item) for item in header]))
    for row in dataloader_rows:
        print(
            " | ".join(
                [
                    str(row["variant"]),
                    str(row["num_workers"]),
                    str(row["batch_size"]),
                    str(row["frames"]),
                    f"{row['seconds']:.3f}",
                    f"{row['fps']:.2f}",
                ]
            )
        )


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "variant",
        "feature_key",
        "stage",
        "count",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "max_ms",
        "num_workers",
        "batch_size",
        "frames",
        "seconds",
        "fps",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Dataset repo id passed to LeRobotDataset.")
    parser.add_argument("--root", type=Path, default=None, help="Local dataset root.")
    parser.add_argument("--revision", default=None, help="Dataset revision.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Optional episode indices.")
    parser.add_argument("--video-backend", default=None, help="Video backend passed to LeRobotDataset.")
    parser.add_argument("--tolerance-s", type=float, default=1e-4, help="Timestamp tolerance in seconds.")
    parser.add_argument("--return-uint8", action="store_true", help="Return RGB video frames as uint8.")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS, choices=DEFAULT_VARIANTS)
    parser.add_argument("--num-samples", type=int, default=8, help="Number of __getitem__ samples.")
    parser.add_argument("--sample-mode", choices=["head", "random"], default="head")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--observation-delta-indices",
        nargs="*",
        default=None,
        help="Optional observation delta indices, e.g. --observation-delta-indices -2 -1 0.",
    )
    parser.add_argument("--skip-stages", action="store_true", help="Skip per-stage __getitem__ profiling.")
    parser.add_argument("--skip-dataloader", action="store_true", help="Skip DataLoader throughput profiling.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, nargs="*", default=[0, 4])
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=0)
    parser.add_argument("--measure-batches", type=int, default=4)
    parser.add_argument("--csv", type=Path, default=None, help="Optional path to write raw benchmark rows.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.observation_delta_indices = parse_int_list(args.observation_delta_indices)

    rows = []
    for variant in args.variants:
        base_dataset = make_dataset(args, variant)
        delta_timestamps = make_delta_timestamps(base_dataset, args.observation_delta_indices) or None

        if not args.skip_stages:
            dataset = make_dataset(args, variant, delta_timestamps=delta_timestamps)
            indices = choose_indices(dataset, args.num_samples, args.seed, args.sample_mode)
            print(
                f"Profiling stages: variant={variant}, samples={len(indices)}, "
                f"video_keys={visual_storage_keys(dataset)}"
            )
            rows.extend(profile_getitem_stages(dataset, variant, indices))

        if not args.skip_dataloader:
            for num_workers in args.num_workers:
                dataset = make_dataset(args, variant, delta_timestamps=delta_timestamps)
                print(
                    f"Profiling DataLoader: variant={variant}, num_workers={num_workers}, "
                    f"video_keys={visual_storage_keys(dataset)}"
                )
                rows.append(profile_dataloader(dataset, variant, args, num_workers))

    print_rows(rows)
    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
