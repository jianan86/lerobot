"""Offline analyzer for pose_act jitter dumps.

Reads three pose_act diagnostic files produced by the async-inference stack when
``diagnostics_dump_dir`` is enabled and prints the A/B/C metrics that separate
intra-chunk smoothness, cross-chunk fusion jumps, and the executed action
sequence. The verdict at the bottom points to the dominant jitter source.

Inputs (under ``--dump_dir``):

- ``pose_act_chunks.jsonl``           : server-side raw absolute pose7d chunks
- ``pose_act_fusion_events.jsonl``    : client-side fusion events at overlapping timesteps
- ``pose_act_executed_actions.csv``   : client-side pose7d sent to the robot

The old filenames ``chunk_dump.jsonl``, ``aggregate_events.jsonl``, and
``executed.csv`` are still accepted for compatibility.

Optional plotting (``--plot``) requires matplotlib.

Example:
    python tools/analyze_pose_act_jitter.py --dump_dir runs/jitter_demo
    python tools/analyze_pose_act_jitter.py --dump_dir runs/jitter_demo --plot --out_dir runs/jitter_demo/plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Chunk:
    obs_step: int
    first_action_step: int
    obs_timestamp: float
    actions: np.ndarray  # [N, 7]


def _unwrap_rpy(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Unwrap rpy in ``curr`` so each component is on the same +/- pi sheet as prev."""
    delta = curr - prev
    delta = delta - 2 * np.pi * np.floor((delta + np.pi) / (2 * np.pi))
    return prev + delta


def _unwrap_chunk_rpy(rpy: np.ndarray) -> np.ndarray:
    """Sequential rpy unwrap along axis=0 for an [N, 3] array."""
    if rpy.shape[0] < 2:
        return rpy.copy()
    out = rpy.copy()
    for i in range(1, out.shape[0]):
        out[i] = _unwrap_rpy(out[i - 1], out[i])
    return out


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not path.exists():
        return chunks
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            actions = np.asarray(data["chunk"], dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] != 7:
                continue
            chunks.append(
                Chunk(
                    obs_step=int(data["obs_step"]),
                    first_action_step=int(data["first_action_step"]),
                    obs_timestamp=float(data.get("obs_timestamp", 0.0)),
                    actions=actions,
                )
            )
    chunks.sort(key=lambda c: c.first_action_step)
    return chunks


def load_aggregate_events(path: Path) -> list[dict]:
    events: list[dict] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def load_executed(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (timesteps[K], pre_pose[K, 7], post_pose[K, 7]). May be empty."""
    if not path.exists():
        return (np.zeros(0, dtype=np.int64), np.zeros((0, 7)), np.zeros((0, 7)))
    timesteps: list[int] = []
    pre_rows: list[list[float]] = []
    post_rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                timesteps.append(int(r["timestep"]))
            except (KeyError, ValueError):
                continue
            pre_rows.append(
                [float(r.get(k, "nan")) for k in (
                    "pre_x",
                    "pre_y",
                    "pre_z",
                    "pre_roll",
                    "pre_pitch",
                    "pre_yaw",
                    "pre_gripper",
                )]
            )
            post_rows.append(
                [float(r.get(k, "nan")) for k in (
                    "post_x",
                    "post_y",
                    "post_z",
                    "post_roll",
                    "post_pitch",
                    "post_yaw",
                    "post_gripper",
                )]
            )
    order = np.argsort(np.asarray(timesteps))
    timesteps_arr = np.asarray(timesteps)[order]
    pre = np.asarray(pre_rows)[order] if pre_rows else np.zeros((0, 7))
    post = np.asarray(post_rows)[order] if post_rows else np.zeros((0, 7))
    return timesteps_arr, pre, post


def _first_existing(dump_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = dump_dir / name
        if path.exists():
            return path
    return dump_dir / names[0]


def _summarize(diffs: np.ndarray) -> dict:
    """diffs: [K, D]; returns scalar stats over the L2 norm of each row."""
    if diffs.size == 0:
        return {"count": 0, "max": 0.0, "rms": 0.0, "p95": 0.0}
    norms = np.linalg.norm(diffs, axis=1)
    return {
        "count": int(diffs.shape[0]),
        "max": float(norms.max()),
        "rms": float(np.sqrt(np.mean(norms ** 2))),
        "p95": float(np.quantile(norms, 0.95)),
    }


def metric_a_intra_chunk(chunks: list[Chunk]) -> dict:
    """Per-chunk first/second-difference of (xyz) and (rpy with unwrap)."""
    xyz_diffs = []
    rpy_diffs = []
    grip_diffs = []
    xyz_second_diffs = []
    rpy_second_diffs = []
    grip_second_diffs = []
    for c in chunks:
        if c.actions.shape[0] < 2:
            continue
        xyz_diffs.append(np.diff(c.actions[:, :3], axis=0))
        rpy_unwrapped = _unwrap_chunk_rpy(c.actions[:, 3:6])
        rpy_diffs.append(np.diff(rpy_unwrapped, axis=0))
        grip_diffs.append(np.diff(c.actions[:, 6:7], axis=0))
        if c.actions.shape[0] >= 3:
            xyz_second_diffs.append(np.diff(c.actions[:, :3], n=2, axis=0))
            rpy_second_diffs.append(np.diff(rpy_unwrapped, n=2, axis=0))
            grip_second_diffs.append(np.diff(c.actions[:, 6:7], n=2, axis=0))
    xyz = np.concatenate(xyz_diffs, axis=0) if xyz_diffs else np.zeros((0, 3))
    rpy = np.concatenate(rpy_diffs, axis=0) if rpy_diffs else np.zeros((0, 3))
    grip = np.concatenate(grip_diffs, axis=0) if grip_diffs else np.zeros((0, 1))
    xyz_second = np.concatenate(xyz_second_diffs, axis=0) if xyz_second_diffs else np.zeros((0, 3))
    rpy_second = np.concatenate(rpy_second_diffs, axis=0) if rpy_second_diffs else np.zeros((0, 3))
    grip_second = np.concatenate(grip_second_diffs, axis=0) if grip_second_diffs else np.zeros((0, 1))
    return {
        "xyz": _summarize(xyz),
        "rpy": _summarize(rpy),
        "gripper": _summarize(grip),
        "second_difference": {
            "xyz": _summarize(xyz_second),
            "rpy": _summarize(rpy_second),
            "gripper": _summarize(grip_second),
        },
    }


def metric_b_cross_chunk(chunks: list[Chunk]) -> dict:
    """For each adjacent chunk pair, diff at overlapping timesteps."""
    by_step: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for c in chunks:
        for i in range(c.actions.shape[0]):
            t = c.first_action_step + i
            by_step[t].append((c.obs_step, c.actions[i]))

    xyz_diffs = []
    rpy_diffs = []
    grip_diffs = []
    intra_position_bins: dict[int, list[float]] = defaultdict(list)

    for c_idx in range(len(chunks) - 1):
        a = chunks[c_idx]
        b = chunks[c_idx + 1]
        a_steps = {a.first_action_step + i: a.actions[i] for i in range(a.actions.shape[0])}
        b_steps = {b.first_action_step + i: b.actions[i] for i in range(b.actions.shape[0])}
        overlap = sorted(set(a_steps).intersection(b_steps))
        for t in overlap:
            va = a_steps[t]
            vb = b_steps[t]
            xyz_diffs.append((vb[:3] - va[:3])[None, :])
            unwrapped_b = _unwrap_rpy(va[3:6], vb[3:6])
            rpy_diffs.append((unwrapped_b - va[3:6])[None, :])
            grip_diffs.append((vb[6:7] - va[6:7])[None, :])
            pos_in_new_chunk = t - b.first_action_step
            intra_position_bins[pos_in_new_chunk].append(
                float(np.linalg.norm(vb[:3] - va[:3]))
            )

    xyz = np.concatenate(xyz_diffs, axis=0) if xyz_diffs else np.zeros((0, 3))
    rpy = np.concatenate(rpy_diffs, axis=0) if rpy_diffs else np.zeros((0, 3))
    grip = np.concatenate(grip_diffs, axis=0) if grip_diffs else np.zeros((0, 1))

    bins_summary: dict[int, dict] = {}
    for k in sorted(intra_position_bins):
        arr = np.asarray(intra_position_bins[k])
        bins_summary[k] = {
            "count": int(arr.size),
            "mean_xyz_jump": float(arr.mean()),
            "p95_xyz_jump": float(np.quantile(arr, 0.95)) if arr.size > 0 else 0.0,
        }

    return {
        "xyz": _summarize(xyz),
        "rpy": _summarize(rpy),
        "gripper": _summarize(grip),
        "by_position_in_new_chunk": bins_summary,
    }


def metric_c_executed(executed: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    timesteps, pre, post = executed
    out: dict = {}
    for tag, arr in (("pre", pre), ("post", post)):
        if arr.shape[0] < 2:
            out[tag] = {
                "xyz": _summarize(np.zeros((0, 3))),
                "rpy": _summarize(np.zeros((0, 3))),
                "second_difference": {
                    "xyz": _summarize(np.zeros((0, 3))),
                    "rpy": _summarize(np.zeros((0, 3))),
                },
            }
            continue
        rpy_unwrapped = _unwrap_chunk_rpy(arr[:, 3:6])
        diff_xyz = np.diff(arr[:, :3], axis=0)
        diff_rpy = np.diff(rpy_unwrapped, axis=0)
        second_diff_xyz = np.diff(arr[:, :3], n=2, axis=0) if arr.shape[0] >= 3 else np.zeros((0, 3))
        second_diff_rpy = np.diff(rpy_unwrapped, n=2, axis=0) if arr.shape[0] >= 3 else np.zeros((0, 3))
        out[tag] = {
            "xyz": _summarize(diff_xyz),
            "rpy": _summarize(diff_rpy),
            "second_difference": {
                "xyz": _summarize(second_diff_xyz),
                "rpy": _summarize(second_diff_rpy),
            },
        }
    return out


def render_verdict(a: dict, b: dict, c: dict) -> str:
    a_rms = a["xyz"]["rms"]
    b_rms = b["xyz"]["rms"]
    c_rms = c.get("post", {}).get("xyz", {}).get("rms", 0.0)
    if b_rms == 0.0 and a_rms == 0.0:
        return "no overlap and no chunks observed; check dump files"

    lines = [
        f"  A_rms (intra-chunk xyz step)        = {a_rms:.5f} m",
        f"  B_rms (cross-chunk xyz jump)        = {b_rms:.5f} m",
        f"  C_rms (executed post-adapter step)  = {c_rms:.5f} m",
    ]
    if a_rms > 0 and b_rms > 0:
        ratio = b_rms / max(a_rms, 1e-9)
        lines.append(f"  ratio B/A                           = {ratio:.2f}x")
        if ratio > 2.0:
            lines.append(
                "  verdict: cross-chunk fusion dominates. Try aggregate_fn=weighted_average "
                "or a linear blend across the overlap window; consider RTC-style smoothing."
            )
        elif ratio < 0.5:
            lines.append(
                "  verdict: model output itself is rough. Add an action-smoothness loss, "
                "enable temporal_ensemble_coeff, or reduce n_action_steps."
            )
        else:
            lines.append(
                "  verdict: comparable contribution from intra-chunk and cross-chunk. "
                "Address both: temporal ensembling on the policy + softer aggregate_fn."
            )
    return "\n".join(lines)


def maybe_plot(out_dir: Path, chunks: list[Chunk], events: list[dict], executed) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    if chunks:
        fig, ax = plt.subplots(figsize=(10, 4))
        for c in chunks[: min(20, len(chunks))]:
            t = np.arange(c.actions.shape[0])
            ax.plot(t, c.actions[:, 0], alpha=0.5)
        ax.set_title("Per-chunk x trajectory (first 20 chunks)")
        ax.set_xlabel("position in chunk")
        ax.set_ylabel("x [m]")
        fig.tight_layout()
        fig.savefig(out_dir / "A_chunks_x.png", dpi=120)
        plt.close(fig)

    if events:
        deltas = np.asarray([e["delta"][:3] for e in events], dtype=np.float64)
        norms = np.linalg.norm(deltas, axis=1)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(norms, bins=40)
        ax.set_title("B: cross-chunk xyz jump magnitude (overlap timesteps)")
        ax.set_xlabel("||delta_xyz|| [m]")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "B_overlap_jump_hist.png", dpi=120)
        plt.close(fig)

    timesteps, pre, post = executed
    if post.shape[0] >= 2:
        fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for i, name in enumerate(("x", "y", "z")):
            ax[i].plot(timesteps, post[:, i])
            ax[i].set_ylabel(f"post_{name}")
        ax[-1].set_xlabel("timestep")
        fig.suptitle("C: executed (post-adapter) trajectory")
        fig.tight_layout()
        fig.savefig(out_dir / "C_executed_xyz.png", dpi=120)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump_dir", required=True, help="Directory containing the three dump files")
    parser.add_argument("--plot", action="store_true", help="Render diagnostic plots (requires matplotlib)")
    parser.add_argument("--out_dir", default=None, help="Output dir for plots (default: <dump_dir>/plots)")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    if not dump_dir.exists():
        print(f"dump_dir does not exist: {dump_dir}")
        return 1

    chunks = load_chunks(_first_existing(dump_dir, ("pose_act_chunks.jsonl", "chunk_dump.jsonl")))
    events = load_aggregate_events(
        _first_existing(dump_dir, ("pose_act_fusion_events.jsonl", "aggregate_events.jsonl"))
    )
    executed = load_executed(
        _first_existing(dump_dir, ("pose_act_executed_actions.csv", "executed.csv"))
    )

    print(f"loaded {len(chunks)} chunks | {len(events)} aggregate events | {executed[0].shape[0]} executed actions")
    if not chunks:
        print("no chunks loaded; cannot compute metrics")
        return 1

    a = metric_a_intra_chunk(chunks)
    b = metric_b_cross_chunk(chunks)
    c = metric_c_executed(executed)

    print()
    print("=== A: intra-chunk first/second-difference ===")
    print(json.dumps(a, indent=2))
    print()
    print("=== B: cross-chunk overlap (chunk_k vs chunk_{k+1}) ===")
    print(json.dumps(b, indent=2))
    print()
    print("=== C: executed trajectory first/second-difference ===")
    print(json.dumps(c, indent=2))
    print()
    print("=== Verdict ===")
    print(render_verdict(a, b, c))

    if args.plot:
        out_dir = Path(args.out_dir) if args.out_dir else dump_dir / "plots"
        maybe_plot(out_dir, chunks, events, executed)
        print(f"plots written to {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
