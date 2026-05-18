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

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.piper_follower import PiperFollower, PiperFollowerConfig
from lerobot.robots.piper_follower.piper_follower import DEG_MILLI_PER_RAD, MILLI_MM_PER_METER


class _JointState:
    joint_1 = 0
    joint_2 = 0
    joint_3 = 0
    joint_4 = 0
    joint_5 = 0
    joint_6 = 0


class _JointMsgs:
    joint_state = _JointState()


class _GripperState:
    grippers_angle = 20_000


class _GripperMsgs:
    gripper_state = _GripperState()


class _EndPose:
    X_axis = 100_000
    Y_axis = 200_000
    Z_axis = 300_000
    RX_axis = 0
    RY_axis = 0
    RZ_axis = 0


class _EndPoseMsgs:
    end_pose = _EndPose()


class _FakePiper:
    instances = []

    def __init__(self, can_name):
        self.can_name = can_name
        self.calls = []
        _FakePiper.instances.append(self)

    def ConnectPort(self):
        self.calls.append(("ConnectPort",))

    def EnablePiper(self):
        self.calls.append(("EnablePiper",))
        return True

    def DisablePiper(self):
        self.calls.append(("DisablePiper",))
        return True

    def MotionCtrl_2(self, *args):
        self.calls.append(("MotionCtrl_2", *args))

    def GripperCtrl(self, *args):
        self.calls.append(("GripperCtrl", *args))

    def JointCtrl(self, *args):
        self.calls.append(("JointCtrl", *args))

    def EndPoseCtrl(self, *args):
        self.calls.append(("EndPoseCtrl", *args))

    def GetArmJointMsgs(self):
        return _JointMsgs()

    def GetArmGripperMsgs(self):
        return _GripperMsgs()

    def GetArmEndPoseMsgs(self):
        return _EndPoseMsgs()


@pytest.fixture
def fake_piper_sdk(monkeypatch):
    _FakePiper.instances.clear()
    module = types.SimpleNamespace(C_PiperInterface_V2=_FakePiper)
    monkeypatch.setitem(sys.modules, "piper_sdk", module)
    yield _FakePiper


def test_connect_configures_can_control(fake_piper_sdk):
    robot = PiperFollower(PiperFollowerConfig(can_name="can-test", move_spd_rate_ctrl=12))

    robot.connect()

    fake = fake_piper_sdk.instances[0]
    assert fake.can_name == "can-test"
    assert fake.calls[:3] == [
        ("ConnectPort",),
        ("EnablePiper",),
        ("MotionCtrl_2", 0x01, 0x01, 12, 0x00),
    ]


def test_get_observation_converts_sdk_units(fake_piper_sdk):
    robot = PiperFollower(PiperFollowerConfig())
    robot.connect()

    obs = robot.get_observation()

    assert obs["joint_1.pos"] == 0
    assert obs["gripper.pos"] == 20_000 / MILLI_MM_PER_METER


def test_get_observation_adds_depth_camera_from_depth_rgb_realsense(fake_piper_sdk, monkeypatch):
    class FakeDepthCamera:
        is_connected = True
        use_depth = True

        def read_latest(self):
            return np.zeros((4, 5, 3), dtype=np.uint8)

        def read_latest_depth(self):
            return np.full((4, 5), 1000, dtype=np.uint16)

        def connect(self):
            pass

    monkeypatch.setattr(
        "lerobot.robots.piper_follower.piper_follower.make_cameras_from_configs",
        lambda _configs: {"depth_camera_rgb": FakeDepthCamera()},
    )
    cfg = PiperFollowerConfig(
        cameras={
            "depth_camera_rgb": SimpleNamespace(height=4, width=5, fps=30, use_depth=True),
        }
    )
    robot = PiperFollower(cfg)
    robot.connect()

    obs = robot.get_observation()

    assert robot.observation_features["depth_camera_rgb"] == (4, 5, 3)
    assert robot.observation_features["depth_camera"] == (4, 5, 1)
    assert obs["depth_camera_rgb"].shape == (4, 5, 3)
    assert obs["depth_camera"].shape == (4, 5, 1)
    assert obs["depth_camera"].dtype == np.uint16


def test_send_joint_delta_converts_to_sdk_units(fake_piper_sdk):
    robot = PiperFollower(PiperFollowerConfig())
    robot.connect()

    returned = robot.send_action({"joint_1.delta": 0.1})

    fake = fake_piper_sdk.instances[0]
    assert ("JointCtrl", round(0.1 * DEG_MILLI_PER_RAD), 0, 0, 0, 0, 0) in fake.calls
    assert returned == {"joint_1.pos": pytest.approx(0.1)}


def test_send_gripper_delta_converts_to_sdk_units(fake_piper_sdk):
    robot = PiperFollower(PiperFollowerConfig())
    robot.connect()

    returned = robot.send_action({"gripper.delta": 0.01})

    fake = fake_piper_sdk.instances[0]
    assert ("GripperCtrl", round(0.03 * MILLI_MM_PER_METER), 1000, 0x01, 0) in fake.calls
    assert returned == {"gripper.pos": pytest.approx(0.03)}


def test_send_ee_delta_uses_end_pose_control(fake_piper_sdk):
    robot = PiperFollower(PiperFollowerConfig(move_spd_rate_ctrl=20))
    robot.connect()

    robot.send_action({"ee.delta_x": 0.01, "ee.delta_z": -0.02})

    fake = fake_piper_sdk.instances[0]
    assert ("MotionCtrl_2", 0x01, 0x00, 20, 0x00) in fake.calls
    assert ("EndPoseCtrl", 110_000, 200_000, 280_000, 0, 0, 0) in fake.calls
