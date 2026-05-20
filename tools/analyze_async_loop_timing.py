"""Summarize async-inference timing diagnostics.

Example:
    python tools/analyze_async_loop_timing.py --dump_dir runs/pose_act_diag/server
    python tools/analyze_async_loop_timing.py --dump_dir runs/pose_act_diag/client
    python tools/analyze_async_loop_timing.py --dump_dir runs/pose_act_diag/merged
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

TIMING_FIELDS_BY_EVENT = {
    "client_observation_sent": ("serialize_ms", "send_rpc_ms", "payload_bytes"),
    "server_observation_received": ("client_to_server_ms", "deserialize_ms"),
    "server_pose_act_timing": (
        "prepare_ms",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "pose_convert_ms",
        "total_ms",
    ),
    "server_actions_ready": ("server_processing_ms", "model_path_ms", "serialize_ms", "chunk_size"),
    "client_chunk_merge": ("deserialize_ms", "queue_update_ms"),
}


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": mean(values),
        "p50": median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _request_id(event: dict[str, Any]) -> str | None:
    request_id = event.get("request_id")
    if request_id is None:
        return None
    return str(request_id)


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_request: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in events:
        event_name = str(event.get("event", ""))
        by_event[event_name].append(event)
        request_id = _request_id(event)
        if request_id is not None:
            by_request[request_id][event_name] = event

    event_summary: dict[str, Any] = {}
    for event_name, fields in TIMING_FIELDS_BY_EVENT.items():
        records = by_event.get(event_name, [])
        event_summary[event_name] = {
            field: _summarize([float(r[field]) for r in records if field in r]) for field in fields
        }

    client_to_server = []
    server_to_client = []
    round_trip = []
    for request_events in by_request.values():
        sent = request_events.get("client_observation_sent")
        received = request_events.get("server_observation_received")
        ready = request_events.get("server_actions_ready")
        merged = request_events.get("client_chunk_merge")
        if sent is not None and received is not None:
            client_to_server.append(
                (float(received["receive_wallclock"]) - float(sent["send_wallclock"])) * 1000
            )
        if ready is not None and merged is not None:
            server_to_client.append(
                (float(merged["receive_wallclock"]) - float(ready["response_wallclock"])) * 1000
            )
        if sent is not None and merged is not None:
            round_trip.append((float(merged["receive_wallclock"]) - float(sent["send_wallclock"])) * 1000)

    return {
        "events": {name: len(records) for name, records in sorted(by_event.items())},
        "timing": event_summary,
        "derived": {
            "client_to_server_wallclock_ms": _summarize(client_to_server),
            "server_to_client_wallclock_ms": _summarize(server_to_client),
            "client_round_trip_ms": _summarize(round_trip),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump_dir", required=True, help="Directory containing async_loop_events.jsonl")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    events = _load_events(dump_dir / "async_loop_events.jsonl")
    if not events:
        print(f"no async_loop_events.jsonl records found under {dump_dir}")
        return 1
    print(json.dumps(summarize_events(events), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
