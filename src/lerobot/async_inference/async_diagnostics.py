"""Asynchronous file writers for async-inference diagnostics.

Two diagnostic groups may be produced under a configured diagnostics dump
directory:

- pose_act jitter diagnostics:
  - ``pose_act_chunks.jsonl``: one line per inference, raw absolute pose7d chunk.
  - ``pose_act_fusion_events.jsonl``: one line per overlapping-timestep fusion event.
  - ``pose_act_executed_actions.csv``: one line per action popped from the local queue.
- async loop timing diagnostics:
  - ``async_loop_events.jsonl``: request, server, receive, queue, and control-loop timing events.

All writers run in a daemon background thread so the inference / control loops are
never blocked by disk I/O. Writers are crash-safe in the sense that the process can be
killed and the file will simply contain whatever was flushed up to that point.

Helpers also provide rpy-unwrapped diff summaries for inline log lines, sharing the
same convention as the offline analysis script.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# pose_act jitter diagnostics
POSE_ACT_CHUNKS_NAME = "pose_act_chunks.jsonl"
POSE_ACT_FUSION_EVENTS_NAME = "pose_act_fusion_events.jsonl"
POSE_ACT_EXECUTED_ACTIONS_NAME = "pose_act_executed_actions.csv"

# async loop timing diagnostics
ASYNC_LOOP_EVENTS_NAME = "async_loop_events.jsonl"


def _to_list(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def unwrap_rpy(prev: Sequence[float] | None, curr: Sequence[float]) -> list[float]:
    """Unwrap roll/pitch/yaw to be in the same +/- pi sheet as ``prev``.

    When ``prev`` is None, returns ``curr`` unchanged. Used so that intra-chunk and
    cross-chunk differences are not contaminated by +/- pi jumps.
    """
    if prev is None:
        return [float(v) for v in curr]
    out = []
    for p, c in zip(prev, curr):
        c = float(c)
        p = float(p)
        delta = c - p
        delta -= 2 * math.pi * math.floor((delta + math.pi) / (2 * math.pi))
        out.append(p + delta)
    return out


def chunk_intra_diff_stats(chunk: np.ndarray) -> dict[str, float]:
    """Compute intra-chunk first-difference summary on an [N, 7] absolute pose7d chunk.

    Returns max/RMS for translation (xyz), rotation (rpy with unwrap), and gripper.
    """
    if chunk.ndim != 2 or chunk.shape[1] != 7:
        raise ValueError(f"expected [N, 7] pose7d chunk, got shape {tuple(chunk.shape)}")
    if chunk.shape[0] < 2:
        return {
            "xyz_step_max": 0.0,
            "xyz_step_rms": 0.0,
            "rpy_step_max": 0.0,
            "rpy_step_rms": 0.0,
            "gripper_step_max": 0.0,
        }

    rpy = chunk[:, 3:6].copy()
    for i in range(1, rpy.shape[0]):
        rpy[i] = unwrap_rpy(rpy[i - 1].tolist(), rpy[i].tolist())

    diff_xyz = np.diff(chunk[:, :3], axis=0)
    diff_rpy = np.diff(rpy, axis=0)
    diff_grip = np.diff(chunk[:, 6])

    return {
        "xyz_step_max": float(np.linalg.norm(diff_xyz, axis=1).max()),
        "xyz_step_rms": float(np.sqrt(np.mean((diff_xyz ** 2).sum(axis=1)))),
        "rpy_step_max": float(np.linalg.norm(diff_rpy, axis=1).max()),
        "rpy_step_rms": float(np.sqrt(np.mean((diff_rpy ** 2).sum(axis=1)))),
        "gripper_step_max": float(np.abs(diff_grip).max()),
    }


@dataclass(frozen=True)
class _JsonlRecord:
    path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class _CsvRecord:
    path: Path
    header: tuple[str, ...]
    row: tuple[Any, ...]


class _AsyncWriter:
    """Single background thread serving all jitter dump files."""

    def __init__(self) -> None:
        self._queue: Queue[_JsonlRecord | _CsvRecord | None] = Queue(maxsize=4096)
        self._thread: threading.Thread | None = None
        self._headers_seen: set[Path] = set()
        self._lock = threading.Lock()

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                t = threading.Thread(target=self._loop, name="jitter-dump-writer", daemon=True)
                t.start()
                self._thread = t

    def submit_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        self._ensure_thread()
        try:
            self._queue.put_nowait(_JsonlRecord(path=path, payload=payload))
        except Exception as e:
            logger.warning(f"jitter dump queue full, dropping record for {path.name}: {e}")

    def submit_csv(self, path: Path, header: Sequence[str], row: Sequence[Any]) -> None:
        self._ensure_thread()
        try:
            self._queue.put_nowait(_CsvRecord(path=path, header=tuple(header), row=tuple(row)))
        except Exception as e:
            logger.warning(f"jitter dump queue full, dropping CSV row for {path.name}: {e}")

    def _loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if item is None:
                return
            try:
                item.path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(item, _JsonlRecord):
                    with item.path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(item.payload, ensure_ascii=False))
                        f.write("\n")
                else:
                    needs_header = item.path not in self._headers_seen and not item.path.exists()
                    with item.path.open("a", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        if needs_header:
                            writer.writerow(item.header)
                        writer.writerow(item.row)
                    self._headers_seen.add(item.path)
            except Exception as e:
                logger.warning(f"jitter dump write failed for {item.path}: {e}")


_writer = _AsyncWriter()


class AsyncDiagnosticsWriter:
    """Thin facade resolving paths under ``dump_dir`` and submitting to the writer.

    Construct with ``dump_dir=None`` to disable; methods become no-ops.
    """

    def __init__(self, dump_dir: str | Path | None) -> None:
        self.dump_dir: Path | None = Path(dump_dir) if dump_dir else None
        if self.dump_dir is not None:
            self.dump_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.dump_dir is not None

    def write_chunk(
        self,
        *,
        obs_step: int,
        first_action_step: int,
        obs_timestamp: float,
        chunk: np.ndarray,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.dump_dir is None:
            return
        payload: dict[str, Any] = {
            "wallclock": time.time(),
            "obs_step": int(obs_step),
            "first_action_step": int(first_action_step),
            "obs_timestamp": float(obs_timestamp),
            "chunk_size": int(chunk.shape[0]),
            "chunk": [_to_list(row) for row in np.asarray(chunk)],
            "intra_diff": chunk_intra_diff_stats(np.asarray(chunk)),
        }
        if extra:
            payload.update(extra)
        _writer.submit_jsonl(self.dump_dir / POSE_ACT_CHUNKS_NAME, payload)

    def write_aggregate_event(
        self,
        *,
        timestep: int,
        old: Sequence[float],
        new: Sequence[float],
        agg: Sequence[float],
        fn_name: str,
        incoming_first_step: int,
        incoming_last_step: int,
    ) -> None:
        if self.dump_dir is None:
            return
        old_l = _to_list(old)
        new_l = _to_list(new)
        delta = [n - o for n, o in zip(new_l, old_l)]
        if len(delta) >= 6:
            unwrapped_new_rpy = unwrap_rpy(old_l[3:6], new_l[3:6])
            delta[3:6] = [unwrapped_new_rpy[i] - old_l[3 + i] for i in range(3)]
        payload = {
            "wallclock": time.time(),
            "timestep": int(timestep),
            "old": old_l,
            "new": new_l,
            "agg": _to_list(agg),
            "delta": delta,
            "fn": str(fn_name),
            "incoming_first_step": int(incoming_first_step),
            "incoming_last_step": int(incoming_last_step),
        }
        _writer.submit_jsonl(self.dump_dir / POSE_ACT_FUSION_EVENTS_NAME, payload)

    def write_executed(
        self,
        *,
        timestep: int,
        action_timestamp: float,
        pose7d_pre_adapter: Sequence[float] | None,
        pose7d_post_adapter: Sequence[float] | None,
    ) -> None:
        if self.dump_dir is None:
            return
        pre = _to_list(pose7d_pre_adapter) if pose7d_pre_adapter is not None else [float("nan")] * 7
        post = _to_list(pose7d_post_adapter) if pose7d_post_adapter is not None else [float("nan")] * 7
        header = (
            "wallclock",
            "timestep",
            "action_timestamp",
            "pre_x",
            "pre_y",
            "pre_z",
            "pre_roll",
            "pre_pitch",
            "pre_yaw",
            "pre_gripper",
            "post_x",
            "post_y",
            "post_z",
            "post_roll",
            "post_pitch",
            "post_yaw",
            "post_gripper",
        )
        row = (
            f"{time.time():.6f}",
            int(timestep),
            f"{float(action_timestamp):.6f}",
            *(f"{v:.6f}" for v in pre),
            *(f"{v:.6f}" for v in post),
        )
        _writer.submit_csv(self.dump_dir / POSE_ACT_EXECUTED_ACTIONS_NAME, header, row)

    def write_async_loop_event(self, event: str, **payload: Any) -> None:
        if self.dump_dir is None:
            return
        record = {
            "event": str(event),
            "wallclock": time.time(),
            **payload,
        }
        _writer.submit_jsonl(self.dump_dir / ASYNC_LOOP_EVENTS_NAME, record)


JitterDumpWriter = AsyncDiagnosticsWriter

__all__ = [
    "ASYNC_LOOP_EVENTS_NAME",
    "POSE_ACT_CHUNKS_NAME",
    "POSE_ACT_EXECUTED_ACTIONS_NAME",
    "POSE_ACT_FUSION_EVENTS_NAME",
    "AsyncDiagnosticsWriter",
    "JitterDumpWriter",
    "chunk_intra_diff_stats",
    "unwrap_rpy",
]
