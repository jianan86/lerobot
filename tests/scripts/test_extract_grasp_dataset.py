#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from lerobot.scripts.lerobot_extract_grasp_dataset import _filter_features, find_grasp_slice


def test_find_grasp_slice_returns_single_stable_closed_interval():
    width = np.array([0.09] * 20 + [0.02] * 40 + [0.09] * 20)

    grasp = find_grasp_slice(width, fps=30)

    assert grasp.start == 20
    assert grasp.end == 60
    assert not grasp.skip
    assert not grasp.warnings


def test_find_grasp_slice_duration_overrides_release_time():
    width = np.array([0.09] * 20 + [0.02] * 40 + [0.09] * 20)

    grasp = find_grasp_slice(width, fps=30, duration_s=0.5)

    assert grasp.start == 20
    assert grasp.end == 35
    assert not grasp.skip


def test_find_grasp_slice_duration_window_keeps_frames_before_and_after_grasp():
    width = np.array([0.09] * 30 + [0.02] * 40 + [0.09] * 20)

    grasp = find_grasp_slice(width, fps=30, duration_s="-0.5,2")

    assert grasp.start == 15
    assert grasp.end == 90
    assert not grasp.skip


def test_find_grasp_slice_duration_window_accepts_sequence():
    width = np.array([0.09] * 30 + [0.02] * 40 + [0.09] * 20)

    grasp = find_grasp_slice(width, fps=30, duration_s=[-0.5, 1.0])

    assert grasp.start == 15
    assert grasp.end == 60
    assert not grasp.skip


def test_find_grasp_slice_warns_and_clips_duration_past_episode_end():
    width = np.array([0.09] * 20 + [0.02] * 40 + [0.09] * 20)

    grasp = find_grasp_slice(width, fps=30, duration_s=3.0)

    assert grasp.start == 20
    assert grasp.end == len(width)
    assert not grasp.skip
    assert any("clipped" in warning.message for warning in grasp.warnings)


def test_find_grasp_slice_warns_and_clips_duration_before_episode_start():
    width = np.array([0.09] * 10 + [0.02] * 40 + [0.09] * 20)

    grasp = find_grasp_slice(width, fps=30, duration_s="-0.5,1")

    assert grasp.start == 0
    assert grasp.end == 40
    assert not grasp.skip
    assert any("before episode beginning" in warning.message for warning in grasp.warnings)


def test_find_grasp_slice_skips_when_no_closed_interval_exists():
    width = np.array([0.09] * 60)

    grasp = find_grasp_slice(width, fps=30)

    assert grasp.skip
    assert grasp.risky
    assert any("range is too small" in warning.message for warning in grasp.warnings)


def test_find_grasp_slice_skips_when_closed_interval_is_too_short():
    width = np.array([0.09] * 20 + [0.02] * 5 + [0.09] * 50)

    grasp = find_grasp_slice(width, fps=30)

    assert grasp.skip
    assert grasp.risky
    assert any("too short" in warning.message for warning in grasp.warnings)


def test_find_grasp_slice_warns_on_multiple_stable_intervals_and_uses_longest():
    width = np.array([0.09] * 12 + [0.02] * 18 + [0.09] * 10 + [0.02] * 25 + [0.09] * 25)

    grasp = find_grasp_slice(width, fps=30)

    assert grasp.start == 40
    assert grasp.end == 65
    assert not grasp.skip
    assert grasp.risky
    assert any("multiple stable closed intervals" in warning.message for warning in grasp.warnings)


def test_find_grasp_slice_warns_when_interval_position_is_suspicious():
    width = np.array([0.02] * 25 + [0.09] * 55)

    grasp = find_grasp_slice(width, fps=30)

    assert grasp.start == 0
    assert grasp.end == 25
    assert not grasp.skip
    assert grasp.risky
    assert any("starts outside expected" in warning.message for warning in grasp.warnings)
    assert any("ends outside expected" in warning.message for warning in grasp.warnings)


def test_filter_features_keeps_only_requested_visual_features():
    features = {
        "observation.images.depth_camera_rgb": {"dtype": "video"},
        "observation.images.fisheye_rgb": {"dtype": "video"},
        "observation.depth.depth_camera": {"dtype": "depth_video"},
        "observation.state": {"dtype": "float32"},
        "action": {"dtype": "float32"},
        "timestamp": {"dtype": "float32"},
    }

    filtered = _filter_features(features, ["observation.images.fisheye_rgb"])

    assert set(filtered) == {
        "observation.images.fisheye_rgb",
        "observation.state",
        "action",
        "timestamp",
    }
