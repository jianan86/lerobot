import numpy as np

from tools.analyze_pose_act_jitter import (
    Chunk,
    _first_existing,
    metric_a_intra_chunk,
    metric_b_cross_chunk,
    metric_c_executed,
)


def test_metric_a_includes_second_difference_without_changing_first_difference():
    chunks = [
        Chunk(
            obs_step=0,
            first_action_step=0,
            obs_timestamp=0.0,
            actions=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
        )
    ]

    summary = metric_a_intra_chunk(chunks)

    assert summary["xyz"]["count"] == 2
    assert summary["xyz"]["max"] == 2.0
    assert summary["second_difference"]["xyz"]["count"] == 1
    assert summary["second_difference"]["xyz"]["rms"] == 1.0


def test_metric_c_includes_second_difference_for_pre_and_post():
    timesteps = np.array([0, 1, 2], dtype=np.int64)
    pre = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    post = pre.copy()

    summary = metric_c_executed((timesteps, pre, post))

    assert summary["pre"]["xyz"]["count"] == 2
    assert summary["pre"]["second_difference"]["xyz"]["count"] == 1
    assert summary["pre"]["second_difference"]["xyz"]["rms"] == 1.0
    assert summary["post"]["second_difference"]["xyz"]["rms"] == 1.0


def test_metric_b_output_is_unchanged():
    chunks = [
        Chunk(
            obs_step=0,
            first_action_step=0,
            obs_timestamp=0.0,
            actions=np.zeros((3, 7), dtype=np.float64),
        ),
        Chunk(
            obs_step=1,
            first_action_step=1,
            obs_timestamp=0.1,
            actions=np.ones((3, 7), dtype=np.float64),
        ),
    ]

    summary = metric_b_cross_chunk(chunks)

    assert set(summary) == {"xyz", "rpy", "gripper", "by_position_in_new_chunk"}


def test_first_existing_prefers_new_diagnostics_filename(tmp_path):
    old_path = tmp_path / "chunk_dump.jsonl"
    new_path = tmp_path / "pose_act_chunks.jsonl"
    old_path.write_text("", encoding="utf-8")
    new_path.write_text("", encoding="utf-8")

    assert _first_existing(tmp_path, ("pose_act_chunks.jsonl", "chunk_dump.jsonl")) == new_path


def test_first_existing_falls_back_to_old_diagnostics_filename(tmp_path):
    old_path = tmp_path / "chunk_dump.jsonl"
    old_path.write_text("", encoding="utf-8")

    assert _first_existing(tmp_path, ("pose_act_chunks.jsonl", "chunk_dump.jsonl")) == old_path
