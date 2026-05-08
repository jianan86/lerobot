"""Simulate the pose_act Piper async-inference closed loop.

This is a discrete-event timing model for the current RobotClient/PolicyServer
queue semantics. It does not simulate pose values or robot physics.

Example:
    uv run python tools/simulate_pose_act_async_loop.py --fps 30 --actions-per-chunk 50 \
        --chunk-size-threshold 0.5 --server-processing-ms 300 --duration-s 20
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimConfig:
    fps: float = 30.0
    actions_per_chunk: int = 50
    chunk_size_threshold: float = 0.5
    duration_s: float = 20.0
    warmup_s: float = 0.0
    server_processing_ms: float = 300.0
    client_to_server_ms: float = 0.0
    server_to_client_ms: float = 0.0
    client_receive_ms: float = 0.0
    queue_update_ms: float = 0.0

    @property
    def dt_s(self) -> float:
        return 1.0 / self.fps

    @property
    def server_processing_s(self) -> float:
        return self.server_processing_ms / 1000.0

    @property
    def client_to_server_s(self) -> float:
        return self.client_to_server_ms / 1000.0

    @property
    def server_to_client_s(self) -> float:
        return self.server_to_client_ms / 1000.0

    @property
    def client_receive_s(self) -> float:
        return self.client_receive_ms / 1000.0

    @property
    def queue_update_s(self) -> float:
        return self.queue_update_ms / 1000.0


@dataclass
class RequestRecord:
    request_id: int
    send_time_s: float
    observation_timestep: int
    latest_action_at_send: int
    queue_size_at_send: int
    queue_ratio_at_send: float


@dataclass
class ChunkRecord:
    request_id: int
    request_send_time_s: float
    receive_time_s: float
    observation_timestep: int
    latest_action_at_receive: int
    queue_size_before: int
    queue_size_after: int
    stale_actions: int
    overlap_actions: int
    new_actions: int
    dropped_old_actions: int

    @property
    def round_trip_ms(self) -> float:
        return (self.receive_time_s - self.request_send_time_s) * 1000.0

    @property
    def action_steps_processed(self) -> int:
        return max(0, self.latest_action_at_receive - self.observation_timestep)


@dataclass
class SimulationResult:
    config: SimConfig
    requests: list[RequestRecord]
    chunks: list[ChunkRecord]
    control_steps: int
    executed_actions: int
    underrun_steps: int


@dataclass(frozen=True)
class MetricSummary:
    requests_sent: int
    chunks_received: int
    request_interval_mean_ms: float
    request_interval_p95_ms: float
    request_interval_max_ms: float
    request_to_chunk_mean_ms: float
    request_to_chunk_p95_ms: float
    request_to_chunk_max_ms: float
    steps_processed_mean: float
    steps_processed_p95: float
    steps_processed_max: float
    overlap_actions_mean: float
    overlap_actions_p95: float
    overlap_actions_max: float
    underrun_steps: int
    underrun_ratio: float


def should_send_request(queue_size: int, action_chunk_size: int, threshold: float) -> bool:
    if action_chunk_size <= 0:
        raise ValueError(f"action_chunk_size must be positive, got {action_chunk_size}")
    return queue_size / action_chunk_size <= threshold


def merge_chunk_queue(
    queue: list[int], latest_action: int, incoming_timesteps: list[int]
) -> tuple[list[int], dict[str, int]]:
    current = set(queue)
    future: list[int] = []
    stale = 0
    overlap = 0
    new = 0

    for timestep in incoming_timesteps:
        if timestep <= latest_action:
            stale += 1
        elif timestep in current:
            overlap += 1
            future.append(timestep)
        else:
            new += 1
            future.append(timestep)

    return future, {
        "stale": stale,
        "overlap": overlap,
        "new": new,
        "dropped_old": max(0, len(queue) - overlap),
    }


def _validate_config(cfg: SimConfig) -> None:
    if cfg.fps <= 0:
        raise ValueError(f"fps must be positive, got {cfg.fps}")
    if cfg.actions_per_chunk <= 0:
        raise ValueError(f"actions_per_chunk must be positive, got {cfg.actions_per_chunk}")
    if not 0 <= cfg.chunk_size_threshold <= 1:
        raise ValueError(f"chunk_size_threshold must be in [0, 1], got {cfg.chunk_size_threshold}")
    if cfg.duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {cfg.duration_s}")
    for name in (
        "warmup_s",
        "server_processing_ms",
        "client_to_server_ms",
        "server_to_client_ms",
        "client_receive_ms",
        "queue_update_ms",
    ):
        value = getattr(cfg, name)
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")


def run_simulation(cfg: SimConfig) -> SimulationResult:
    _validate_config(cfg)

    queue: list[int] = []
    latest_action = -1
    requests: list[RequestRecord] = []
    chunks: list[ChunkRecord] = []
    events: list[tuple[float, int, str, Any]] = []
    event_seq = 0
    pending_observation: RequestRecord | None = None
    server_busy = False
    control_steps = 0
    executed_actions = 0
    underrun_steps = 0
    request_id = 0

    def push_event(time_s: float, event_type: str, payload: Any) -> None:
        nonlocal event_seq
        heapq.heappush(events, (time_s, event_seq, event_type, payload))
        event_seq += 1

    def start_server(request: RequestRecord, start_time_s: float) -> None:
        nonlocal server_busy
        server_busy = True
        push_event(start_time_s + cfg.server_processing_s, "server_complete", request)

    def handle_event(time_s: float, event_type: str, payload: Any) -> None:
        nonlocal latest_action, pending_observation, queue, server_busy

        if event_type == "obs_arrive":
            request = payload
            if server_busy:
                pending_observation = request
            else:
                start_server(request, time_s)
            return

        if event_type == "server_complete":
            request = payload
            receive_time_s = (
                time_s + cfg.server_to_client_s + cfg.client_receive_s + cfg.queue_update_s
            )
            push_event(receive_time_s, "chunk_receive", request)
            server_busy = False
            if pending_observation is not None:
                next_request = pending_observation
                pending_observation = None
                start_server(next_request, time_s)
            return

        if event_type == "chunk_receive":
            request = payload
            incoming = [
                request.observation_timestep + i
                for i in range(cfg.actions_per_chunk)
            ]
            queue_before = len(queue)
            new_queue, counts = merge_chunk_queue(queue, latest_action, incoming)
            queue = new_queue
            chunks.append(
                ChunkRecord(
                    request_id=request.request_id,
                    request_send_time_s=request.send_time_s,
                    receive_time_s=time_s,
                    observation_timestep=request.observation_timestep,
                    latest_action_at_receive=latest_action,
                    queue_size_before=queue_before,
                    queue_size_after=len(queue),
                    stale_actions=counts["stale"],
                    overlap_actions=counts["overlap"],
                    new_actions=counts["new"],
                    dropped_old_actions=counts["dropped_old"],
                )
            )
            return

        raise ValueError(f"unknown event type: {event_type}")

    n_ticks = int(math.ceil(cfg.duration_s / cfg.dt_s))
    for tick in range(n_ticks):
        now_s = tick * cfg.dt_s
        while events and events[0][0] <= now_s:
            event_time_s, _, event_type, payload = heapq.heappop(events)
            handle_event(event_time_s, event_type, payload)

        control_steps += 1
        if queue:
            latest_action = queue.pop(0)
            executed_actions += 1
        else:
            underrun_steps += 1

        if should_send_request(len(queue), cfg.actions_per_chunk, cfg.chunk_size_threshold):
            ratio = len(queue) / cfg.actions_per_chunk
            request = RequestRecord(
                request_id=request_id,
                send_time_s=now_s,
                observation_timestep=max(latest_action, 0),
                latest_action_at_send=latest_action,
                queue_size_at_send=len(queue),
                queue_ratio_at_send=ratio,
            )
            request_id += 1
            requests.append(request)
            push_event(now_s + cfg.client_to_server_s, "obs_arrive", request)

    while events:
        event_time_s, _, event_type, payload = heapq.heappop(events)
        if event_time_s > cfg.duration_s:
            break
        handle_event(event_time_s, event_type, payload)

    return SimulationResult(
        config=cfg,
        requests=requests,
        chunks=chunks,
        control_steps=control_steps,
        executed_actions=executed_actions,
        underrun_steps=underrun_steps,
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_async_loop_events(actual_dir: Path) -> list[dict[str, Any]]:
    path = actual_dir / "async_loop_events.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing async loop events file: {path}")
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _events_by_name(events: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event") == name]


def summarize_simulation(result: SimulationResult) -> MetricSummary:
    cfg = result.config
    requests = [r for r in result.requests if r.send_time_s >= cfg.warmup_s]
    chunks = [c for c in result.chunks if c.receive_time_s >= cfg.warmup_s]
    request_intervals = [
        (b.send_time_s - a.send_time_s) * 1000.0
        for a, b in zip(requests, requests[1:], strict=False)
    ]
    round_trip = [c.round_trip_ms for c in chunks]
    processed_steps = [float(c.action_steps_processed) for c in chunks]
    overlaps = [float(c.overlap_actions) for c in chunks]
    return MetricSummary(
        requests_sent=len(requests),
        chunks_received=len(chunks),
        request_interval_mean_ms=_mean(request_intervals),
        request_interval_p95_ms=_percentile(request_intervals, 0.95),
        request_interval_max_ms=max(request_intervals) if request_intervals else 0.0,
        request_to_chunk_mean_ms=_mean(round_trip),
        request_to_chunk_p95_ms=_percentile(round_trip, 0.95),
        request_to_chunk_max_ms=max(round_trip) if round_trip else 0.0,
        steps_processed_mean=_mean(processed_steps),
        steps_processed_p95=_percentile(processed_steps, 0.95),
        steps_processed_max=max(processed_steps) if processed_steps else 0.0,
        overlap_actions_mean=_mean(overlaps),
        overlap_actions_p95=_percentile(overlaps, 0.95),
        overlap_actions_max=max(overlaps) if overlaps else 0.0,
        underrun_steps=result.underrun_steps,
        underrun_ratio=result.underrun_steps / max(1, result.control_steps),
    )


def summarize_actual_events(events: list[dict[str, Any]], warmup_s: float = 0.0) -> MetricSummary:
    if not events:
        return MetricSummary(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

    first_wallclock = min(float(event["wallclock"]) for event in events if "wallclock" in event)
    min_wallclock = first_wallclock + warmup_s
    requests = [
        event for event in _events_by_name(events, "client_request_sent")
        if float(event.get("send_wallclock", event.get("wallclock", 0.0))) >= min_wallclock
    ]
    merges = [
        event for event in _events_by_name(events, "client_chunk_merge")
        if float(event.get("receive_wallclock", event.get("wallclock", 0.0))) >= min_wallclock
    ]
    underruns = [
        event for event in _events_by_name(events, "client_control_underrun")
        if float(event.get("wallclock", 0.0)) >= min_wallclock
    ]
    executed = [
        event for event in _events_by_name(events, "client_action_executed")
        if float(event.get("wallclock", 0.0)) >= min_wallclock
    ]
    request_by_id = {event.get("request_id"): event for event in requests}
    request_times = sorted(float(event.get("send_wallclock", event.get("wallclock", 0.0))) for event in requests)
    request_intervals = [
        (b - a) * 1000.0 for a, b in zip(request_times, request_times[1:], strict=False)
    ]
    round_trip = []
    for merge in merges:
        request = request_by_id.get(merge.get("request_id"))
        if request is None:
            continue
        recv = float(merge.get("receive_wallclock", merge.get("wallclock", 0.0)))
        sent = float(request.get("send_wallclock", request.get("wallclock", 0.0)))
        round_trip.append((recv - sent) * 1000.0)
    processed_steps = [
        float(int(event.get("latest_action_at_receive", 0)) - int(event.get("incoming_first_step", 0)))
        for event in merges
    ]
    overlaps = [float(event.get("overlap_actions", 0)) for event in merges]
    control_steps = len(executed) + len(underruns)
    return MetricSummary(
        requests_sent=len(requests),
        chunks_received=len(merges),
        request_interval_mean_ms=_mean(request_intervals),
        request_interval_p95_ms=_percentile(request_intervals, 0.95),
        request_interval_max_ms=max(request_intervals) if request_intervals else 0.0,
        request_to_chunk_mean_ms=_mean(round_trip),
        request_to_chunk_p95_ms=_percentile(round_trip, 0.95),
        request_to_chunk_max_ms=max(round_trip) if round_trip else 0.0,
        steps_processed_mean=_mean(processed_steps),
        steps_processed_p95=_percentile(processed_steps, 0.95),
        steps_processed_max=max(processed_steps) if processed_steps else 0.0,
        overlap_actions_mean=_mean(overlaps),
        overlap_actions_p95=_percentile(overlaps, 0.95),
        overlap_actions_max=max(overlaps) if overlaps else 0.0,
        underrun_steps=len(underruns),
        underrun_ratio=len(underruns) / max(1, control_steps),
    )


def derive_config_from_actual(events: list[dict[str, Any]], base: SimConfig) -> SimConfig:
    requests = _events_by_name(events, "client_request_sent")
    merges = _events_by_name(events, "client_chunk_merge")
    server_ready = _events_by_name(events, "server_actions_ready")
    server_recv = _events_by_name(events, "server_observation_received")

    wallclocks = [float(event["wallclock"]) for event in events if "wallclock" in event]
    duration_s = max(wallclocks) - min(wallclocks) if len(wallclocks) >= 2 else base.duration_s

    fps_values = [float(event["fps"]) for event in requests if "fps" in event]
    chunk_values = [int(event["actions_per_chunk"]) for event in requests if "actions_per_chunk" in event]
    threshold_values = [
        float(event["chunk_size_threshold"]) for event in requests if "chunk_size_threshold" in event
    ]
    server_processing = [
        float(event["server_processing_ms"]) for event in server_ready if "server_processing_ms" in event
    ]
    client_to_server = [
        float(event["client_to_server_ms"]) for event in server_recv if "client_to_server_ms" in event
    ]
    queue_update = [float(event["queue_update_ms"]) for event in merges if "queue_update_ms" in event]

    actual_summary = summarize_actual_events(events, warmup_s=base.warmup_s)
    downstream_ms = max(
        0.0,
        actual_summary.request_to_chunk_mean_ms
        - _mean(client_to_server)
        - _mean(server_processing)
        - _mean(queue_update)
    )

    return SimConfig(
        fps=_mean(fps_values) if fps_values else base.fps,
        actions_per_chunk=round(_mean([float(v) for v in chunk_values])) if chunk_values else base.actions_per_chunk,
        chunk_size_threshold=_mean(threshold_values) if threshold_values else base.chunk_size_threshold,
        duration_s=duration_s if duration_s > 0 else base.duration_s,
        warmup_s=base.warmup_s,
        server_processing_ms=_mean(server_processing) if server_processing else base.server_processing_ms,
        client_to_server_ms=_mean(client_to_server) if client_to_server else base.client_to_server_ms,
        server_to_client_ms=0.0,
        client_receive_ms=downstream_ms,
        queue_update_ms=_mean(queue_update) if queue_update else base.queue_update_ms,
    )


def print_summary_table(title: str, summary: MetricSummary) -> None:
    print(title)
    print(f"requests_sent={summary.requests_sent} chunks_received={summary.chunks_received}")
    print(
        "request_interval_ms "
        f"mean={summary.request_interval_mean_ms:.2f} "
        f"p95={summary.request_interval_p95_ms:.2f} "
        f"max={summary.request_interval_max_ms:.2f}"
    )
    print(
        "request_to_chunk_ms "
        f"mean={summary.request_to_chunk_mean_ms:.2f} "
        f"p95={summary.request_to_chunk_p95_ms:.2f} "
        f"max={summary.request_to_chunk_max_ms:.2f}"
    )
    print(
        "steps_processed_before_chunk "
        f"mean={summary.steps_processed_mean:.2f} "
        f"p95={summary.steps_processed_p95:.2f} "
        f"max={summary.steps_processed_max:.0f}"
    )
    print(
        "overlap_actions_per_chunk "
        f"mean={summary.overlap_actions_mean:.2f} "
        f"p95={summary.overlap_actions_p95:.2f} "
        f"max={summary.overlap_actions_max:.0f}"
    )
    print(
        f"underrun_steps={summary.underrun_steps} "
        f"underrun_ratio={summary.underrun_ratio:.3f}"
    )


def print_comparison(actual: MetricSummary, simulated: MetricSummary) -> None:
    print("=== actual vs simulation delta ===")
    rows = [
        ("requests_sent", actual.requests_sent, simulated.requests_sent),
        ("chunks_received", actual.chunks_received, simulated.chunks_received),
        ("request_interval_mean_ms", actual.request_interval_mean_ms, simulated.request_interval_mean_ms),
        ("request_to_chunk_mean_ms", actual.request_to_chunk_mean_ms, simulated.request_to_chunk_mean_ms),
        ("steps_processed_mean", actual.steps_processed_mean, simulated.steps_processed_mean),
        ("overlap_actions_mean", actual.overlap_actions_mean, simulated.overlap_actions_mean),
        ("underrun_ratio", actual.underrun_ratio, simulated.underrun_ratio),
    ]
    for name, actual_value, simulated_value in rows:
        print(f"{name}: actual={actual_value:.3f} sim={simulated_value:.3f} delta={simulated_value - actual_value:.3f}")


def print_report(result: SimulationResult, max_events: int) -> None:
    cfg = result.config
    warm_chunks = [c for c in result.chunks if c.receive_time_s >= cfg.warmup_s]
    summary = summarize_simulation(result)

    print("=== pose_act async loop simulation ===")
    print(
        f"fps={cfg.fps:g} dt={cfg.dt_s * 1000:.2f}ms "
        f"actions_per_chunk={cfg.actions_per_chunk} threshold={cfg.chunk_size_threshold:g}"
    )
    print(
        f"server_processing={cfg.server_processing_ms:.2f}ms "
        f"client_to_server={cfg.client_to_server_ms:.2f}ms "
        f"server_to_client={cfg.server_to_client_ms:.2f}ms "
        f"client_receive={cfg.client_receive_ms:.2f}ms "
        f"queue_update={cfg.queue_update_ms:.2f}ms"
    )
    print()
    print_summary_table("=== summary ===", summary)

    if max_events <= 0:
        return

    print()
    print("=== first chunk events ===")
    header = (
        "id send_s recv_s obs_step latest_recv rt_ms steps_done "
        "q_before stale overlap new dropped_old q_after"
    )
    print(header)
    for chunk in warm_chunks[:max_events]:
        print(
            f"{chunk.request_id} "
            f"{chunk.request_send_time_s:.3f} "
            f"{chunk.receive_time_s:.3f} "
            f"{chunk.observation_timestep} "
            f"{chunk.latest_action_at_receive} "
            f"{chunk.round_trip_ms:.1f} "
            f"{chunk.action_steps_processed} "
            f"{chunk.queue_size_before} "
            f"{chunk.stale_actions} "
            f"{chunk.overlap_actions} "
            f"{chunk.new_actions} "
            f"{chunk.dropped_old_actions} "
            f"{chunk.queue_size_after}"
        )


def write_csv(result: SimulationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "request_id",
                "request_send_time_s",
                "chunk_receive_time_s",
                "round_trip_ms",
                "observation_timestep",
                "latest_action_at_receive",
                "action_steps_processed",
                "queue_size_before",
                "stale_actions",
                "overlap_actions",
                "new_actions",
                "dropped_old_actions",
                "queue_size_after",
            ],
        )
        writer.writeheader()
        for chunk in result.chunks:
            writer.writerow(
                {
                    "request_id": chunk.request_id,
                    "request_send_time_s": f"{chunk.request_send_time_s:.9f}",
                    "chunk_receive_time_s": f"{chunk.receive_time_s:.9f}",
                    "round_trip_ms": f"{chunk.round_trip_ms:.6f}",
                    "observation_timestep": chunk.observation_timestep,
                    "latest_action_at_receive": chunk.latest_action_at_receive,
                    "action_steps_processed": chunk.action_steps_processed,
                    "queue_size_before": chunk.queue_size_before,
                    "stale_actions": chunk.stale_actions,
                    "overlap_actions": chunk.overlap_actions,
                    "new_actions": chunk.new_actions,
                    "dropped_old_actions": chunk.dropped_old_actions,
                    "queue_size_after": chunk.queue_size_after,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.5)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--warmup-s", type=float, default=0.0)
    parser.add_argument("--server-processing-ms", type=float, default=300.0)
    parser.add_argument("--client-to-server-ms", type=float, default=0.0)
    parser.add_argument("--server-to-client-ms", type=float, default=0.0)
    parser.add_argument("--client-receive-ms", type=float, default=0.0)
    parser.add_argument("--queue-update-ms", type=float, default=0.0)
    parser.add_argument("--actual-dir", type=Path, default=None)
    parser.add_argument("--derive-from-actual", action="store_true")
    parser.add_argument("--compare-actual", action="store_true")
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(
        fps=args.fps,
        actions_per_chunk=args.actions_per_chunk,
        chunk_size_threshold=args.chunk_size_threshold,
        duration_s=args.duration_s,
        warmup_s=args.warmup_s,
        server_processing_ms=args.server_processing_ms,
        client_to_server_ms=args.client_to_server_ms,
        server_to_client_ms=args.server_to_client_ms,
        client_receive_ms=args.client_receive_ms,
        queue_update_ms=args.queue_update_ms,
    )
    actual_events = _load_async_loop_events(args.actual_dir) if args.actual_dir is not None else None
    if args.derive_from_actual:
        if actual_events is None:
            raise ValueError("--derive-from-actual requires --actual-dir")
        cfg = derive_config_from_actual(actual_events, cfg)

    result = run_simulation(cfg)
    print_report(result, max_events=args.max_events)
    if args.compare_actual:
        if actual_events is None:
            raise ValueError("--compare-actual requires --actual-dir")
        print()
        actual_summary = summarize_actual_events(actual_events, warmup_s=cfg.warmup_s)
        simulated_summary = summarize_simulation(result)
        print_summary_table("=== actual summary ===", actual_summary)
        print()
        print_comparison(actual_summary, simulated_summary)
    if args.csv is not None:
        write_csv(result, args.csv)
        print(f"\nwrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
