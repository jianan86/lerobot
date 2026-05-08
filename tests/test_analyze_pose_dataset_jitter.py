import json

import numpy as np

from tools import analyze_pose_dataset_jitter
from tools.analyze_pose_dataset_jitter import analyze_dataset, summarize_feature_by_episode, summarize_pose_sequence


def test_summarize_pose_sequence_unwraps_rpy():
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, np.pi - 0.01, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -np.pi + 0.01, 0.0],
        ],
        dtype=np.float64,
    )

    summary = summarize_pose_sequence(poses)

    assert summary["first_difference"]["rpy"]["max"] < 0.03


def test_summarize_feature_by_episode_does_not_cross_episode_boundaries():
    episodes = {
        0: np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        1: np.array(
            [
                [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [101.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    }

    summary = summarize_feature_by_episode(episodes)

    assert summary["overall"]["first_difference"]["xyz"]["count"] == 2
    assert summary["overall"]["first_difference"]["xyz"]["max"] == 1.0


def test_analyze_dataset_reports_dataset_frames_once(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "features": {
                    "observation.state": {"shape": [7], "names": None},
                    "action": {"shape": [7], "names": None},
                },
            }
        ),
        encoding="utf-8",
    )

    poses = {
        0: np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    }
    monkeypatch.setattr(analyze_pose_dataset_jitter, "_load_feature_episodes", lambda *args: poses)

    result = analyze_dataset(dataset_root)

    assert result["features"]["observation.state"]["frames"] == 2
    assert result["features"]["action"]["frames"] == 2
    assert result["total_frames"] == 2
