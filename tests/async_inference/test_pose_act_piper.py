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

from __future__ import annotations

import pytest
import torch

pytest.importorskip("grpc")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")


class _StubPoseActPiperRobot:
    def __init__(self):
        self._ee_state = {"x": 0.2, "y": 0.0, "z": 0.3, "rx": 0.0, "ry": 0.0, "rz": 0.0}
        self._joint_state = {"gripper.pos": 0.0}

    def _get_end_pose(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._ee_state.items()}

    def _get_motor_positions(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._joint_state.items()}


def _make_adapter():
    from lerobot.async_inference.adapters import PoseActPiperAdapter

    return PoseActPiperAdapter(_StubPoseActPiperRobot())


def test_pose_act_piper_round_trip_ee_tcp():
    adapter = _make_adapter()
    ee_pose = torch.tensor([0.35, -0.12, 0.41, 0.4, -0.3, 0.2, 0.02], dtype=torch.float32)

    tcp_pose = adapter.ee_pose7d_to_tcp_pose7d(ee_pose)
    restored = adapter.tcp_pose7d_to_ee_pose7d(tcp_pose)

    torch.testing.assert_close(restored, ee_pose, rtol=0, atol=1e-6)


def test_pose_act_piper_round_trip_tcp_ee():
    adapter = _make_adapter()
    tcp_pose = torch.tensor([0.18, -0.07, 0.29, -0.2, 0.5, -0.4, 0.01], dtype=torch.float32)

    ee_pose = adapter.tcp_pose7d_to_ee_pose7d(tcp_pose)
    restored = adapter.ee_pose7d_to_tcp_pose7d(ee_pose)

    torch.testing.assert_close(restored, tcp_pose, rtol=0, atol=1e-6)


def test_pose_act_piper_fixed_transform_matches_declared_matrix():
    from lerobot.async_inference.adapters.pose_act_piper import T_EE_TCP
    from lerobot.utils.pose_act import euler_rpy_to_matrix

    adapter = _make_adapter()
    ee_pose = torch.zeros(7, dtype=torch.float32)
    tcp_pose = adapter.ee_pose7d_to_tcp_pose7d(ee_pose)

    assert tcp_pose[0].item() == pytest.approx(0.0, abs=1e-6)
    assert tcp_pose[1].item() == pytest.approx(0.0, abs=1e-6)
    assert tcp_pose[2].item() == pytest.approx(0.1943, abs=1e-6)
    tcp_rot = euler_rpy_to_matrix(tcp_pose[3:6])
    torch.testing.assert_close(tcp_rot, T_EE_TCP[:3, :3], rtol=0, atol=1e-6)


def test_pose_act_piper_translation_uses_full_ee_pose_chain():
    from lerobot.async_inference.adapters.pose_act_piper import T_EE_TCP
    from lerobot.utils.pose_act import euler_rpy_to_matrix

    adapter = _make_adapter()
    ee_pose = torch.tensor([0.11, -0.08, 0.26, 0.4, -0.2, 0.3, 0.0], dtype=torch.float32)

    tcp_pose = adapter.ee_pose7d_to_tcp_pose7d(ee_pose)

    ee_rot = euler_rpy_to_matrix(ee_pose[3:6])
    expected_pos = ee_pose[:3] + ee_rot @ T_EE_TCP[:3, 3]
    torch.testing.assert_close(tcp_pose[:3], expected_pos, rtol=0, atol=1e-6)


def test_pose_act_piper_convert_pose7d_tcp_to_ee_action():
    adapter = _make_adapter()
    tcp_pose = torch.tensor([0.1, 0.2, 0.5, 0.0, 0.0, 0.0, 0.03], dtype=torch.float32)
    ee_pose = adapter.tcp_pose7d_to_ee_pose7d(tcp_pose)

    action = adapter.convert(tcp_pose)

    assert action["ee.abs_x"] == pytest.approx(float(ee_pose[0].item()), abs=1e-6)
    assert action["ee.abs_y"] == pytest.approx(float(ee_pose[1].item()), abs=1e-6)
    assert action["ee.abs_z"] == pytest.approx(float(ee_pose[2].item()), abs=1e-6)
    assert action["ee.abs_rx"] == pytest.approx(float(ee_pose[3].item()), abs=1e-6)
    assert action["ee.abs_ry"] == pytest.approx(float(ee_pose[4].item()), abs=1e-6)
    assert action["ee.abs_rz"] == pytest.approx(float(ee_pose[5].item()), abs=1e-6)
    assert action["gripper.pos"] == pytest.approx(0.03, abs=1e-6)
