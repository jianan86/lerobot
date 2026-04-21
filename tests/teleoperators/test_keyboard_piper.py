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

from unittest.mock import patch

import pytest

from lerobot.teleoperators.keyboard_piper import (
    KeyboardPiperEndEffectorTeleop,
    KeyboardPiperEndEffectorTeleopConfig,
    KeyboardPiperJointTeleop,
    KeyboardPiperJointTeleopConfig,
)


@pytest.fixture(autouse=True)
def _skip_pynput_requirement():
    with patch(
        "lerobot.teleoperators.keyboard_piper.keyboard_piper.require_package",
        lambda *_args, **_kw: None,
    ):
        yield


def test_joint_keyboard_maps_pressed_keys_to_deltas():
    teleop = KeyboardPiperJointTeleop(KeyboardPiperJointTeleopConfig(joint_step=0.1, gripper_step=0.01))
    teleop.current_pressed = {"q": True, "s": True, "u": True}

    action = KeyboardPiperJointTeleop.get_action.__wrapped__(teleop)

    assert action["joint_1.delta"] == pytest.approx(0.1)
    assert action["joint_2.delta"] == pytest.approx(-0.1)
    assert action["gripper.delta"] == pytest.approx(0.01)


def test_joint_keyboard_stop_key_returns_zero_deltas():
    teleop = KeyboardPiperJointTeleop(KeyboardPiperJointTeleopConfig(joint_step=0.1))
    teleop.current_pressed = {"q": True, "x": True}

    action = KeyboardPiperJointTeleop.get_action.__wrapped__(teleop)

    assert action == {}


def test_ee_keyboard_maps_pressed_keys_to_deltas():
    teleop = KeyboardPiperEndEffectorTeleop(
        KeyboardPiperEndEffectorTeleopConfig(xyz_step=0.01, rpy_step=0.1, gripper_step=0.02)
    )
    teleop.current_pressed = {"w": True, "e": True, "r": True, "j": True}

    action = KeyboardPiperEndEffectorTeleop.get_action.__wrapped__(teleop)

    assert action["ee.delta_x"] == pytest.approx(0.01)
    assert action["ee.delta_z"] == pytest.approx(-0.01)
    assert action["ee.delta_rx"] == pytest.approx(0.1)
    assert action["gripper.delta"] == pytest.approx(-0.02)
