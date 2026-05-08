"""Analyze pose trajectory smoothness in a local LeRobot dataset.

The script reads numeric pose features from ``data/*/*.parquet`` under a
LeRobot dataset root and prints first/second-difference metrics per episode.
It does not generate plots.

Example:
    python tools/analyze_pose_dataset_jitter.py --dataset_root /home/jianan/workspace/data/lerobot_0429
    python tools/analyze_pose_dataset_jitter.py --dataset_root /home/jianan/workspace/data/lerobot_0429 --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


POSE_PARTS = ("xyz", "rpy", "gripper")
DEFAULT_FEATURES = ("observation.state", "action")


def _unwrap_rpy(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Unwrap rpy in ``curr`` so each component stays near ``prev``."""
    delta = curr - prev
    delta = delta - 2 * np.pi * np.floor((delta + np.pi) / (2 * np.pi))
    return prev + delta


def _unwrap_sequence_rpy(rpy: np.ndarray) -> np.ndarray:
    if rpy.shape[0] < 2:
        return rpy.copy()
    out = rpy.copy()
    for i in range(1, out.shape[0]):
        out[i] = _unwrap_rpy(out[i - 1], out[i])
    return out


def _summarize(diffs: np.ndarray) -> dict[str, float | int]:
    if diffs.size == 0:
        return {"count": 0, "mean": 0.0, "max": 0.0, "rms": 0.0, "p95": 0.0}
    norms = np.linalg.norm(diffs, axis=1)
    return {
        "count": int(diffs.shape[0]),
        "mean": float(norms.mean()),
        "max": float(norms.max()),
        "rms": float(np.sqrt(np.mean(norms**2))),
        "p95": float(np.quantile(norms, 0.95)),
    }


def _empty_part_summary() -> dict[str, dict[str, float | int]]:
    return {part: _summarize(np.zeros((0, 1))) for part in POSE_PARTS}


def summarize_pose_sequence(poses: np.ndarray) -> dict[str, Any]:
    """Return first/second-difference summaries for one [T, 7] pose sequence."""
    if poses.ndim != 2 or poses.shape[1] < 7:
        raise ValueError(f"expected poses with shape [T, >=7], got {poses.shape}")

    first: dict[str, dict[str, float | int]] = _empty_part_summary()
    second: dict[str, dict[str, float | int]] = _empty_part_summary()

    if poses.shape[0] >= 2:
        rpy = _unwrap_sequence_rpy(poses[:, 3:6])
        first = {
            "xyz": _summarize(np.diff(poses[:, :3], axis=0)),
            "rpy": _summarize(np.diff(rpy, axis=0)),
            "gripper": _summarize(np.diff(poses[:, 6:7], axis=0)),
        }

    if poses.shape[0] >= 3:
        rpy = _unwrap_sequence_rpy(poses[:, 3:6])
        second = {
            "xyz": _summarize(np.diff(poses[:, :3], n=2, axis=0)),
            "rpy": _summarize(np.diff(rpy, n=2, axis=0)),
            "gripper": _summarize(np.diff(poses[:, 6:7], n=2, axis=0)),
        }

    return {"first_difference": first, "second_difference": second}


def _merge_summaries(rows: list[np.ndarray], width: int) -> dict[str, float | int]:
    arr = np.concatenate(rows, axis=0) if rows else np.zeros((0, width), dtype=np.float64)
    return _summarize(arr)


def summarize_feature_by_episode(episodes: dict[int, np.ndarray]) -> dict[str, Any]:
    first_rows: dict[str, list[np.ndarray]] = {part: [] for part in POSE_PARTS}
    second_rows: dict[str, list[np.ndarray]] = {part: [] for part in POSE_PARTS}
    per_episode: list[dict[str, Any]] = []

    for episode_index in sorted(episodes):
        poses = episodes[episode_index]
        if poses.shape[0] >= 2:
            rpy = _unwrap_sequence_rpy(poses[:, 3:6])
            first_rows["xyz"].append(np.diff(poses[:, :3], axis=0))
            first_rows["rpy"].append(np.diff(rpy, axis=0))
            first_rows["gripper"].append(np.diff(poses[:, 6:7], axis=0))
        if poses.shape[0] >= 3:
            rpy = _unwrap_sequence_rpy(poses[:, 3:6])
            second_rows["xyz"].append(np.diff(poses[:, :3], n=2, axis=0))
            second_rows["rpy"].append(np.diff(rpy, n=2, axis=0))
            second_rows["gripper"].append(np.diff(poses[:, 6:7], n=2, axis=0))

        episode_summary = summarize_pose_sequence(poses)
        per_episode.append(
            {
                "episode_index": int(episode_index),
                "frames": int(poses.shape[0]),
                **episode_summary,
            }
        )

    overall = {
        "first_difference": {
            "xyz": _merge_summaries(first_rows["xyz"], 3),
            "rpy": _merge_summaries(first_rows["rpy"], 3),
            "gripper": _merge_summaries(first_rows["gripper"], 1),
        },
        "second_difference": {
            "xyz": _merge_summaries(second_rows["xyz"], 3),
            "rpy": _merge_summaries(second_rows["rpy"], 3),
            "gripper": _merge_summaries(second_rows["gripper"], 1),
        },
    }
    worst = sorted(
        per_episode,
        key=lambda item: item["second_difference"]["xyz"]["rms"],
        reverse=True,
    )[:10]
    return {"overall": overall, "worst_episodes": worst}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_feature_episodes(
    dataset_root: Path,
    feature: str,
    episodes: set[int] | None,
) -> dict[int, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to read LeRobot parquet data; install lerobot[dataset]") from exc

    rows: list[Any] = []
    columns = ["episode_index", "frame_index", feature]
    for path in sorted((dataset_root / "data").glob("*/*.parquet")):
        df = pd.read_parquet(path, columns=columns)
        if episodes is not None:
            df = df[df["episode_index"].isin(episodes)]
        if not df.empty:
            rows.append(df)

    if not rows:
        return {}

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"])

    out: dict[int, np.ndarray] = {}
    for episode_index, group in df.groupby("episode_index", sort=True):
        poses = np.asarray(group[feature].to_list(), dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] < 7:
            raise ValueError(f"feature {feature!r} must contain [N, >=7] pose rows, got {poses.shape}")
        out[int(episode_index)] = poses
    return out


def analyze_dataset(
    dataset_root: Path,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    episodes: set[int] | None = None,
) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing dataset metadata: {info_path}")

    info = _load_json(info_path)
    available = info.get("features", {})
    result: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "fps": info.get("fps"),
        "features": {},
    }

    for feature in features:
        if feature not in available:
            raise ValueError(f"feature {feature!r} not found in meta/info.json")
        shape = available[feature].get("shape", [])
        if not shape or int(shape[0]) < 7:
            raise ValueError(f"feature {feature!r} must have width >= 7, got shape={shape}")

        feature_episodes = _load_feature_episodes(dataset_root, feature, episodes)
        result["features"][feature] = {
            "frames": int(sum(arr.shape[0] for arr in feature_episodes.values())),
            "episodes": int(len(feature_episodes)),
            "names": available[feature].get("names"),
            **summarize_feature_by_episode(feature_episodes),
        }

    result["total_frames"] = int(max((item["frames"] for item in result["features"].values()), default=0))
    result["total_episodes"] = int(max((item["episodes"] for item in result["features"].values()), default=0))
    return result


def _print_summary(result: dict[str, Any]) -> None:
    print(
        f"loaded dataset_root={result['dataset_root']} "
        f"episodes={result['total_episodes']} frames={result['total_frames']} fps={result['fps']}"
    )
    for feature, payload in result["features"].items():
        print()
        print(f"=== {feature} trajectory jitter ===")
        print(f"frames={payload['frames']} episodes={payload['episodes']} names={payload['names']}")
        print(json.dumps(payload["overall"], indent=2))
        print("worst_episodes_by_xyz_second_diff_rms:")
        for episode in payload["worst_episodes"][:5]:
            rms = episode["second_difference"]["xyz"]["rms"]
            p95 = episode["second_difference"]["xyz"]["p95"]
            print(
                f"  episode={episode['episode_index']} frames={episode['frames']} "
                f"xyz_second_rms={rms:.8f} xyz_second_p95={p95:.8f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", required=True, help="Local LeRobot dataset root")
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES), help="Pose feature columns to analyze")
    parser.add_argument("--episodes", nargs="*", type=int, default=None, help="Optional episode indices to analyze")
    parser.add_argument("--json", action="store_true", help="Print only JSON")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"dataset_root does not exist: {dataset_root}")
        return 1

    episodes = set(args.episodes) if args.episodes is not None else None
    result = analyze_dataset(dataset_root, tuple(args.features), episodes)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
