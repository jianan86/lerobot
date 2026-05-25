#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

DEFAULT_RUNS = {
    "0520_orange": {
        "dataset_root": "/home/jianan/workspace/data/lerobot_0520_orange",
        "checkpoint": "outputs/train/pose_act_0520orange_mid_enlarge/checkpoints/200000/pretrained_model",
    },
    "0429": {
        "dataset_root": "/home/jianan/workspace/data/lerobot_0429",
        "checkpoint": "outputs/train/pose_act_0429/checkpoints/020000/pretrained_model",
    },
}

GRIPPER_7D_INDEX = 6
GRIPPER_10D_INDEX = 9


@dataclass(frozen=True)
class RunSpec:
    label: str
    dataset_root: Path
    checkpoint: Path


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_int(value: Any) -> int:
    array = _to_numpy(value)
    return int(array.item() if array.shape == () else array.reshape(-1)[0])


def _to_float(value: Any) -> float:
    array = _to_numpy(value)
    return float(array.item() if array.shape == () else array.reshape(-1)[0])


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    return value


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.quantile(values, q))


def _basic_stats(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "q10": _quantile(values, 0.10),
        "q50": _quantile(values, 0.50),
        "q90": _quantile(values, 0.90),
        "std": float(values.std()),
        "range": float(values.max() - values.min()),
    }


def load_metadata_dataset(root: Path, episodes: list[int] | None = None) -> LeRobotDataset:
    return LeRobotDataset(
        repo_id=root.name,
        root=root,
        episodes=episodes,
        download_videos=False,
    )


def iter_column_rows(dataset: LeRobotDataset, columns: list[str]):
    narrow = dataset.select_columns(columns)
    for idx in range(len(narrow)):
        yield narrow[idx]


def audit_dataset_distribution(dataset: LeRobotDataset, close_eps: float) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    grippers: list[float] = []
    for row in iter_column_rows(dataset, ["episode_index", "frame_index", ACTION, OBS_STATE]):
        action = _to_numpy(row[ACTION]).astype(np.float64)
        state = _to_numpy(row[OBS_STATE]).astype(np.float64)
        rows.append(
            {
                "episode_index": _to_int(row["episode_index"]),
                "frame_index": _to_int(row["frame_index"]),
                "action_gripper": float(action[GRIPPER_7D_INDEX]),
                "state_gripper": float(state[GRIPPER_7D_INDEX]),
            }
        )
        grippers.append(float(action[GRIPPER_7D_INDEX]))

    frames = pd.DataFrame(rows)
    action_gripper = np.asarray(grippers, dtype=np.float64)
    episode_rows = []
    for ep_idx, group in frames.groupby("episode_index", sort=True):
        values = group["action_gripper"].to_numpy(dtype=np.float64)
        deltas = np.diff(values)
        open_level = float(np.quantile(values, 0.90))
        target_level = float(np.quantile(values, 0.10))
        denom = open_level - target_level
        progress = np.zeros_like(values) if abs(denom) < 1e-9 else (open_level - values) / denom
        close_frames = int(np.count_nonzero(progress >= 1.0 - close_eps))
        episode_rows.append(
            {
                "episode_index": int(ep_idx),
                "num_frames": int(len(values)),
                "open_level": open_level,
                "target_level": target_level,
                "gripper_range": float(values.max() - values.min()),
                "gripper_total_abs_delta": float(np.abs(deltas).sum()) if deltas.size else 0.0,
                "gripper_max_abs_step": float(np.abs(deltas).max()) if deltas.size else 0.0,
                "close_frames": close_frames,
                "close_frame_fraction": close_frames / len(values),
            }
        )

    episode_df = pd.DataFrame(episode_rows)
    summary = {
        "num_frames": int(len(frames)),
        "num_episodes": int(frames["episode_index"].nunique()),
        "gripper": _basic_stats(action_gripper),
        "episode_gripper_total_abs_delta": _basic_stats(
            episode_df["gripper_total_abs_delta"].to_numpy(dtype=np.float64)
        ),
        "episode_gripper_max_abs_step": _basic_stats(
            episode_df["gripper_max_abs_step"].to_numpy(dtype=np.float64)
        ),
        "episode_close_frames": _basic_stats(episode_df["close_frames"].to_numpy(dtype=np.float64)),
    }
    return summary, episode_df


def choose_replay_episodes(episode_df: pd.DataFrame, max_episodes: int) -> list[int]:
    if max_episodes <= 0 or len(episode_df) == 0:
        return []
    ranked = episode_df.sort_values("gripper_total_abs_delta").reset_index(drop=True)
    positions = np.linspace(0, len(ranked) - 1, num=min(max_episodes, len(ranked)))
    episodes = [int(ranked.iloc[int(round(pos))]["episode_index"]) for pos in positions]
    return list(dict.fromkeys(episodes))


def load_policy_and_processors(checkpoint: Path, device: str):
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = device
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(checkpoint, config=config, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "pose_act_umi_normalizer": {"device": device},
        },
        postprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "pose_act_umi_unnormalizer": {"device": "cpu"},
        },
    )
    return config, policy, preprocessor, postprocessor


def load_replay_dataset(root: Path, config: PreTrainedConfig, episodes: list[int]) -> LeRobotDataset:
    image_key = getattr(config, "fisheye_rgb_key", "observation.images.fisheye_rgb")
    delta_timestamps = {
        OBS_STATE: [idx / 30 for idx in config.observation_delta_indices],
        image_key: [idx / 30 for idx in config.observation_delta_indices],
    }
    return LeRobotDataset(
        repo_id=root.name,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )


def build_observation(item: dict[str, Any], input_keys: list[str]) -> dict[str, torch.Tensor]:
    observation = {}
    for key in input_keys:
        value = item[key]
        observation[key] = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return observation


def progress_from_levels(values: np.ndarray, open_level: float, target_level: float) -> np.ndarray:
    denom = open_level - target_level
    if abs(denom) < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return (open_level - values) / denom


def first_crossing(values: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(values >= threshold)
    return int(hits[0]) if hits.size else None


def replay_policy(
    root: Path,
    config: PreTrainedConfig,
    policy,
    preprocessor,
    postprocessor,
    episode_df: pd.DataFrame,
    episodes: list[int],
    max_frames_per_episode: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not episodes:
        return pd.DataFrame(), {}

    dataset = load_replay_dataset(root, config, episodes)
    input_keys = list(config.input_features)
    episode_levels = episode_df.set_index("episode_index")[["open_level", "target_level"]].to_dict("index")
    records: list[dict[str, Any]] = []
    previous_episode = None
    frames_seen_in_episode: dict[int, int] = {}

    for rel_idx in range(len(dataset)):
        item = dataset[rel_idx]
        ep_idx = _to_int(item["episode_index"])
        seen = frames_seen_in_episode.get(ep_idx, 0)
        if seen >= max_frames_per_episode:
            continue
        frames_seen_in_episode[ep_idx] = seen + 1

        if previous_episode != ep_idx:
            policy.reset()
            previous_episode = ep_idx

        observation = build_observation(item, input_keys)
        processed_observation = preprocessor(observation)
        raw_action = policy.select_action(processed_observation)
        post_action = postprocessor(raw_action)

        gt_gripper = float(_to_numpy(item[ACTION])[GRIPPER_7D_INDEX])
        pred_gripper = float(post_action.detach().cpu().reshape(-1, 10)[0, GRIPPER_10D_INDEX])
        raw_gripper = float(raw_action.detach().cpu().reshape(-1, 10)[0, GRIPPER_10D_INDEX])
        levels = episode_levels[ep_idx]
        gt_progress = progress_from_levels(
            np.asarray([gt_gripper]), levels["open_level"], levels["target_level"]
        )[0]
        pred_progress = progress_from_levels(
            np.asarray([pred_gripper]), levels["open_level"], levels["target_level"]
        )[0]
        records.append(
            {
                "episode_index": ep_idx,
                "frame_index": _to_int(item["frame_index"]),
                "timestamp": _to_float(item["timestamp"]),
                "gt_gripper": gt_gripper,
                "pred_gripper": pred_gripper,
                "raw_normalized_gripper10d": raw_gripper,
                "gt_progress": float(gt_progress),
                "pred_progress": float(pred_progress),
            }
        )

    replay_df = pd.DataFrame(records)
    if replay_df.empty:
        return replay_df, {}

    replay_df = replay_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    replay_df["gt_delta"] = replay_df.groupby("episode_index")["gt_gripper"].diff().fillna(0.0)
    replay_df["pred_delta"] = replay_df.groupby("episode_index")["pred_gripper"].diff().fillna(0.0)
    replay_df["abs_delta_error"] = (replay_df["pred_delta"] - replay_df["gt_delta"]).abs()
    replay_df["abs_gripper_error"] = (replay_df["pred_gripper"] - replay_df["gt_gripper"]).abs()
    replay_df["abs_progress_error"] = (replay_df["pred_progress"] - replay_df["gt_progress"]).abs()

    lags = []
    jitter = []
    slope_ratios = []
    for _ep_idx, group in replay_df.groupby("episode_index", sort=True):
        gt_progress = group["gt_progress"].to_numpy(dtype=np.float64)
        pred_progress = group["pred_progress"].to_numpy(dtype=np.float64)
        gt_cross = first_crossing(gt_progress, 0.5)
        pred_cross = first_crossing(pred_progress, 0.5)
        if gt_cross is not None and pred_cross is not None:
            lags.append(pred_cross - gt_cross)
        pred_delta = group["pred_delta"].to_numpy(dtype=np.float64)
        gt_delta = group["gt_delta"].to_numpy(dtype=np.float64)
        closing = gt_delta < 0
        if np.count_nonzero(closing) > 1:
            jitter.append(float(np.std(pred_delta[closing])))
            gt_sum = float(gt_delta[closing].sum())
            if abs(gt_sum) > 1e-9:
                slope_ratios.append(float(pred_delta[closing].sum() / gt_sum))

    summary = {
        "episodes": episodes,
        "num_frames": int(len(replay_df)),
        "gripper_mae": float(replay_df["abs_gripper_error"].mean()),
        "progress_mae": float(replay_df["abs_progress_error"].mean()),
        "delta_mae": float(replay_df["abs_delta_error"].mean()),
        "mean_closing_start_lag_frames": float(np.mean(lags)) if lags else math.nan,
        "mean_closing_pred_delta_jitter": float(np.mean(jitter)) if jitter else math.nan,
        "mean_closing_slope_ratio": float(np.mean(slope_ratios)) if slope_ratios else math.nan,
    }
    summary["delta_buckets"] = delta_bucket_summary(replay_df)
    return replay_df, summary


def delta_bucket_summary(replay_df: pd.DataFrame) -> dict[str, Any]:
    values = replay_df["gt_delta"].abs().to_numpy(dtype=np.float64)
    nonzero = values[values > 1e-9]
    if nonzero.size < 3:
        return {}
    q1, q2 = np.quantile(nonzero, [1 / 3, 2 / 3])
    buckets = {
        "low": replay_df[replay_df["gt_delta"].abs() <= q1],
        "mid": replay_df[(replay_df["gt_delta"].abs() > q1) & (replay_df["gt_delta"].abs() <= q2)],
        "high": replay_df[replay_df["gt_delta"].abs() > q2],
    }
    result = {"thresholds": {"low_max": float(q1), "mid_max": float(q2)}}
    for name, group in buckets.items():
        gt = group["gt_delta"].to_numpy(dtype=np.float64)
        pred = group["pred_delta"].to_numpy(dtype=np.float64)
        moving = np.abs(gt) > 1e-9
        same_direction = np.sign(gt[moving]) == np.sign(pred[moving])
        result[name] = {
            "num_frames": int(len(group)),
            "delta_mae": float(np.mean(np.abs(pred - gt))) if len(group) else math.nan,
            "direction_accuracy": float(np.mean(same_direction)) if same_direction.size else math.nan,
            "slope_ratio": float(pred[moving].sum() / gt[moving].sum())
            if moving.any() and abs(float(gt[moving].sum())) > 1e-9
            else math.nan,
        }
    return result


def build_loss_samples(
    root: Path,
    config: PreTrainedConfig,
    preprocessor,
    episode_df: pd.DataFrame,
    episodes: list[int],
    batch_size: int,
):
    if batch_size <= 0 or not episodes:
        return None

    dataset = load_replay_dataset(root, config, episodes)
    raw = load_metadata_dataset(root, episodes=episodes)
    action_rows = list(iter_column_rows(raw, ["episode_index", "frame_index", ACTION]))
    by_episode: dict[int, list[np.ndarray]] = {}
    for row in action_rows:
        by_episode.setdefault(_to_int(row["episode_index"]), []).append(_to_numpy(row[ACTION]).astype(np.float32))

    candidate_indices = np.linspace(0, max(0, len(dataset) - 1), num=min(batch_size, len(dataset)), dtype=int)
    processed_samples = []
    for rel_idx in candidate_indices:
        item = dataset[int(rel_idx)]
        ep_idx = _to_int(item["episode_index"])
        frame_idx = _to_int(item["frame_index"])
        actions = by_episode[ep_idx]
        chunk = []
        pad = []
        for offset in range(config.chunk_size):
            action_idx = frame_idx + offset
            if action_idx < len(actions):
                chunk.append(actions[action_idx])
                pad.append(False)
            else:
                chunk.append(actions[-1])
                pad.append(True)
        sample = build_observation(item, list(config.input_features))
        sample[ACTION] = torch.as_tensor(np.stack(chunk), dtype=torch.float32)
        sample["action_is_pad"] = torch.as_tensor(pad, dtype=torch.bool)
        processed_samples.append(preprocessor(sample))

    if not processed_samples:
        return None

    batch: dict[str, torch.Tensor] = {}
    for key in processed_samples[0]:
        if key in {"reward", "done", "truncated", "info", "task"}:
            continue
        values = [sample[key] for sample in processed_samples if isinstance(sample.get(key), torch.Tensor)]
        if len(values) == len(processed_samples):
            batch[key] = torch.stack(values, dim=0)
    if ACTION in batch and batch[ACTION].ndim == 4 and batch[ACTION].shape[1] == 1:
        batch[ACTION] = batch[ACTION].squeeze(1)
    if "action_is_pad" in batch and batch["action_is_pad"].ndim == 3 and batch["action_is_pad"].shape[1] == 1:
        batch["action_is_pad"] = batch["action_is_pad"].squeeze(1)
    return batch


@torch.no_grad()
def audit_loss(policy, batch: dict[str, torch.Tensor] | None) -> dict[str, Any]:
    if batch is None:
        return {}
    device = next(policy.model.parameters()).device
    model_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()
    }
    prepared = policy._prepare_batch(model_batch)
    predictions = policy.model(prepared)[0]
    target = prepared[ACTION]
    valid = ~prepared["action_is_pad"].unsqueeze(-1)
    per_entry = torch.abs(predictions - target) * valid
    valid_count = valid.sum().clamp_min(1)
    per_dim = per_entry.sum(dim=(0, 1)) / valid_count
    scalar_l1 = per_entry.mean()
    gripper_l1 = per_dim[GRIPPER_10D_INDEX]
    xyz_l1_mean = per_dim[:3].mean()
    rot6d_l1_mean = per_dim[3:9].mean()
    per_dim_sum = per_dim.sum().clamp_min(torch.finfo(per_dim.dtype).eps)
    gripper_share = gripper_l1 / per_dim_sum

    target_gripper = target[..., GRIPPER_10D_INDEX]
    target_gripper_delta = target_gripper[:, 1:] - target_gripper[:, :-1]
    changing_steps = (target_gripper_delta.abs() > 1e-4).sum()
    total_delta_steps = torch.numel(target_gripper_delta)
    weights = {}
    for weight in (3, 5, 10):
        weighted_per_dim = per_dim.clone()
        weighted_per_dim[GRIPPER_10D_INDEX] *= weight
        weights[f"{weight}x"] = {
            "scalar_l1_mean_if_weighted": float(weighted_per_dim.mean().detach().cpu()),
            "gripper_loss_share_if_weighted": float(
                (weighted_per_dim[GRIPPER_10D_INDEX] / weighted_per_dim.sum().clamp_min(1e-12))
                .detach()
                .cpu()
            ),
        }

    return {
        "batch_size": int(target.shape[0]),
        "chunk_size": int(target.shape[1]),
        "valid_action_steps": int((~prepared["action_is_pad"]).sum().detach().cpu()),
        "scalar_l1_mean_current": float(scalar_l1.detach().cpu()),
        "per_dim_l1": [float(x) for x in per_dim.detach().cpu()],
        "xyz_l1_mean": float(xyz_l1_mean.detach().cpu()),
        "rot6d_l1_mean": float(rot6d_l1_mean.detach().cpu()),
        "gripper_l1": float(gripper_l1.detach().cpu()),
        "gripper_loss_share": float(gripper_share.detach().cpu()),
        "gripper_changing_delta_step_fraction": float(changing_steps.detach().cpu()) / max(total_delta_steps, 1),
        "counterfactual_gripper_weights": weights,
    }


def write_plots(replay_df: pd.DataFrame, output_dir: Path, label: str, max_plots: int) -> list[str]:
    if replay_df.empty or max_plots <= 0:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths = []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ranked = (
        replay_df.groupby("episode_index")["gt_delta"]
        .apply(lambda s: float(s.abs().sum()))
        .sort_values(ascending=False)
    )
    for ep_idx in ranked.head(max_plots).index:
        group = replay_df[replay_df["episode_index"] == ep_idx]
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        axes[0].plot(group["frame_index"], group["gt_gripper"], label="gt")
        axes[0].plot(group["frame_index"], group["pred_gripper"], label="pred")
        axes[0].set_ylabel("gripper")
        axes[0].legend()
        axes[1].plot(group["frame_index"], group["gt_delta"], label="gt_delta")
        axes[1].plot(group["frame_index"], group["pred_delta"], label="pred_delta")
        axes[1].set_ylabel("delta")
        axes[1].set_xlabel("frame")
        axes[1].legend()
        fig.tight_layout()
        path = plot_dir / f"{label}_episode_{int(ep_idx):04d}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PoseACT gripper distribution, replay behavior, and 10D loss dilution."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pose_act_gripper_audit"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-episodes", type=int, default=6)
    parser.add_argument("--max-frames-per-episode", type=int, default=180)
    parser.add_argument("--loss-batch-size", type=int, default=12)
    parser.add_argument("--close-progress-eps", type=float, default=0.05)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-loss", action="store_true")
    parser.add_argument("--plots", type=int, default=3)
    for label, defaults in DEFAULT_RUNS.items():
        parser.add_argument(f"--{label}-dataset-root", type=Path, default=Path(defaults["dataset_root"]))
        parser.add_argument(f"--{label}-checkpoint", type=Path, default=Path(defaults["checkpoint"]))
    return parser.parse_args()


def run_one(spec: RunSpec, args: argparse.Namespace) -> dict[str, Any]:
    print(f"[{spec.label}] dataset audit: {spec.dataset_root}")
    dataset = load_metadata_dataset(spec.dataset_root)
    dataset_summary, episode_df = audit_dataset_distribution(dataset, args.close_progress_eps)
    selected_episodes = choose_replay_episodes(episode_df, args.max_episodes)

    run_dir = args.output_dir / spec.label
    run_dir.mkdir(parents=True, exist_ok=True)
    episode_df.to_parquet(run_dir / "episode_gripper_summary.parquet", index=False)

    summary: dict[str, Any] = {
        "dataset_root": spec.dataset_root,
        "checkpoint": spec.checkpoint,
        "dataset": dataset_summary,
        "selected_episodes": selected_episodes,
    }

    if not args.skip_inference or not args.skip_loss:
        print(f"[{spec.label}] loading policy: {spec.checkpoint}")
        config, policy, preprocessor, postprocessor = load_policy_and_processors(spec.checkpoint, args.device)
        summary["policy"] = {
            "type": config.type,
            "device": config.device,
            "chunk_size": config.chunk_size,
            "n_action_steps": config.n_action_steps,
            "input_features": list(config.input_features),
            "output_features": list(config.output_features),
        }

        if not args.skip_inference:
            print(f"[{spec.label}] replay inference episodes={selected_episodes}")
            replay_df, replay_summary = replay_policy(
                spec.dataset_root,
                config,
                policy,
                preprocessor,
                postprocessor,
                episode_df,
                selected_episodes,
                args.max_frames_per_episode,
            )
            if not replay_df.empty:
                replay_df.to_parquet(run_dir / "frame_replay_gripper.parquet", index=False)
                summary["plots"] = write_plots(replay_df, run_dir, spec.label, args.plots)
            summary["replay"] = replay_summary

        if not args.skip_loss:
            print(f"[{spec.label}] loss dilution audit")
            loss_batch = build_loss_samples(
                spec.dataset_root,
                config,
                preprocessor,
                episode_df,
                selected_episodes,
                args.loss_batch_size,
            )
            summary["loss"] = audit_loss(policy, loss_batch)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    print(f"[{spec.label}] wrote {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = []
    for label in DEFAULT_RUNS:
        specs.append(
            RunSpec(
                label=label,
                dataset_root=getattr(args, f"{label}_dataset_root"),
                checkpoint=getattr(args, f"{label}_checkpoint"),
            )
        )

    all_summary = {spec.label: run_one(spec, args) for spec in specs}
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(all_summary), indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
