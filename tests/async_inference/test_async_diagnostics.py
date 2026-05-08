from __future__ import annotations


def test_async_diagnostics_exports_new_names_and_compat_alias():
    from lerobot.async_inference.async_diagnostics import (
        ASYNC_LOOP_EVENTS_NAME,
        POSE_ACT_CHUNKS_NAME,
        POSE_ACT_EXECUTED_ACTIONS_NAME,
        POSE_ACT_FUSION_EVENTS_NAME,
        AsyncDiagnosticsWriter,
        JitterDumpWriter,
    )

    assert POSE_ACT_CHUNKS_NAME == "pose_act_chunks.jsonl"
    assert POSE_ACT_FUSION_EVENTS_NAME == "pose_act_fusion_events.jsonl"
    assert POSE_ACT_EXECUTED_ACTIONS_NAME == "pose_act_executed_actions.csv"
    assert ASYNC_LOOP_EVENTS_NAME == "async_loop_events.jsonl"
    assert JitterDumpWriter is AsyncDiagnosticsWriter
