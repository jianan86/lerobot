from __future__ import annotations

import pytest

from tools.simulate_pose_act_async_loop import (
    SimConfig,
    derive_config_from_actual,
    merge_chunk_queue,
    run_simulation,
    summarize_actual_events,
    should_send_request,
)


@pytest.mark.parametrize(
    "queue_size, chunk_size, threshold, expected",
    [
        (8, 20, 0.5, True),
        (10, 20, 0.5, True),
        (11, 20, 0.5, False),
        (0, 20, 0.0, True),
    ],
)
def test_should_send_request_uses_queue_ratio(queue_size, chunk_size, threshold, expected):
    assert should_send_request(queue_size, chunk_size, threshold) is expected


def test_merge_chunk_queue_counts_stale_overlap_and_new_actions():
    queue = [5, 6, 7, 8]
    incoming = [3, 4, 5, 6, 7, 8, 9]

    new_queue, counts = merge_chunk_queue(queue, latest_action=4, incoming_timesteps=incoming)

    assert new_queue == [5, 6, 7, 8, 9]
    assert counts == {
        "stale": 2,
        "overlap": 4,
        "new": 1,
        "dropped_old": 0,
    }


def test_merge_chunk_queue_drops_old_non_overlapping_actions_like_client():
    new_queue, counts = merge_chunk_queue(
        queue=[20, 21, 22],
        latest_action=10,
        incoming_timesteps=[11, 12],
    )

    assert new_queue == [11, 12]
    assert counts["dropped_old"] == 3


def test_simulation_round_trip_includes_processing_and_transport_times():
    result = run_simulation(
        SimConfig(
            fps=10,
            actions_per_chunk=5,
            chunk_size_threshold=0.5,
            duration_s=1.0,
            server_processing_ms=200,
            client_to_server_ms=50,
            server_to_client_ms=50,
            client_receive_ms=20,
        )
    )

    first_chunk = result.chunks[0]

    assert first_chunk.round_trip_ms == pytest.approx(320.0)
    assert first_chunk.observation_timestep == 0


def test_simulation_round_trip_includes_queue_update_time():
    result = run_simulation(
        SimConfig(
            fps=10,
            actions_per_chunk=5,
            chunk_size_threshold=0.5,
            duration_s=1.0,
            server_processing_ms=200,
            queue_update_ms=30,
        )
    )

    assert result.chunks[0].round_trip_ms == pytest.approx(230.0)


def test_simulation_reports_queue_underruns_with_slow_server_and_small_chunks():
    fast_enough = run_simulation(
        SimConfig(
            fps=10,
            actions_per_chunk=8,
            chunk_size_threshold=0.5,
            duration_s=2.0,
            server_processing_ms=100,
        )
    )
    too_slow = run_simulation(
        SimConfig(
            fps=10,
            actions_per_chunk=2,
            chunk_size_threshold=0.5,
            duration_s=2.0,
            server_processing_ms=600,
        )
    )

    assert too_slow.underrun_steps > fast_enough.underrun_steps


def test_actual_events_can_derive_config_and_summary():
    events = [
        {
            "event": "client_request_sent",
            "wallclock": 10.0,
            "request_id": 1,
            "send_wallclock": 10.0,
            "fps": 10,
            "actions_per_chunk": 5,
            "chunk_size_threshold": 0.4,
        },
        {
            "event": "server_observation_received",
            "wallclock": 10.05,
            "request_id": 1,
            "client_to_server_ms": 50.0,
        },
        {
            "event": "server_actions_ready",
            "wallclock": 10.25,
            "request_id": 1,
            "server_processing_ms": 200.0,
            "chunk_size": 5,
        },
        {
            "event": "client_chunk_merge",
            "wallclock": 10.32,
            "request_id": 1,
            "receive_wallclock": 10.32,
            "latest_action_at_receive": 3,
            "incoming_first_step": 1,
            "overlap_actions": 2,
            "queue_update_ms": 10.0,
            "deserialize_ms": 5.0,
        },
        {"event": "client_action_executed", "wallclock": 10.4, "request_id": 1},
        {"event": "client_control_underrun", "wallclock": 10.5},
    ]

    summary = summarize_actual_events(events)
    cfg = derive_config_from_actual(events, SimConfig())

    assert summary.requests_sent == 1
    assert summary.chunks_received == 1
    assert summary.request_to_chunk_mean_ms == pytest.approx(320.0)
    assert summary.steps_processed_mean == 2
    assert summary.overlap_actions_mean == 2
    assert summary.underrun_ratio == pytest.approx(0.5)
    assert cfg.fps == 10
    assert cfg.actions_per_chunk == 5
    assert cfg.chunk_size_threshold == pytest.approx(0.4)
    assert cfg.server_processing_ms == pytest.approx(200.0)
    assert cfg.client_to_server_ms == pytest.approx(50.0)
