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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_piper_follower import PiperFollowerConfig

logger = logging.getLogger(__name__)

JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
GRIPPER_NAME = "gripper"
DEG_MILLI_PER_RAD = 1000.0 * 180.0 / 3.1415926
MILLI_MM_PER_METER = 1000.0 * 1000.0
MILLI_DEG_PER_RAD = DEG_MILLI_PER_RAD


def _clip(value: float, limits: tuple[float, float]) -> float:
    return min(max(value, limits[0]), limits[1])


class PiperFollower(Robot):
    config_class = PiperFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig):
        super().__init__(config)
        self.config = config
        self.piper: Any | None = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self.logs: dict[str, float] = {}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in (*JOINT_NAMES, GRIPPER_NAME)}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features = {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
        depth_rgb_cfg = self.config.cameras.get("depth_camera_rgb")
        if depth_rgb_cfg is not None and getattr(depth_rgb_cfg, "use_depth", False):
            features["depth_camera"] = (depth_rgb_cfg.height, depth_rgb_cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        joint_deltas = {f"{motor}.delta": float for motor in (*JOINT_NAMES, GRIPPER_NAME)}
        ee_deltas = {
            "ee.delta_x": float,
            "ee.delta_y": float,
            "ee.delta_z": float,
            "ee.delta_rx": float,
            "ee.delta_ry": float,
            "ee.delta_rz": float,
            "gripper.delta": float,
        }
        ee_abs = {
            "ee.abs_x": float,
            "ee.abs_y": float,
            "ee.abs_z": float,
            "ee.abs_rx": float,
            "ee.abs_ry": float,
            "ee.abs_rz": float,
        }
        return {**self._motors_ft, **joint_deltas, **ee_deltas, **ee_abs}

    @property
    def is_connected(self) -> bool:
        return self.piper is not None and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        try:
            from piper_sdk import C_PiperInterface_V2
        except ImportError as e:
            raise ImportError(
                "'piper_sdk' is required for PiperFollower. Install it with "
                "`pip install -e ../piper/piper_sdk` or `pip install 'lerobot[piper]'`."
            ) from e

        self.piper = C_PiperInterface_V2(self.config.can_name)
        self.piper.ConnectPort()
        if self.config.enable_on_connect:
            start = time.perf_counter()
            while not self.piper.EnablePiper():
                if (
                    self.config.enable_timeout_s is not None
                    and time.perf_counter() - start > self.config.enable_timeout_s
                ):
                    self.piper = None
                    raise TimeoutError(f"Timed out enabling Piper on {self.config.can_name}.")
                time.sleep(0.01)

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        if self.piper is None:
            return
        self.piper.MotionCtrl_2(0x01, 0x01, self.config.move_spd_rate_ctrl, 0x00)

    def _get_motor_positions(self) -> dict[str, float]:
        assert self.piper is not None
        joints_msg = self.piper.GetArmJointMsgs().joint_state
        gripper_msg = self.piper.GetArmGripperMsgs().gripper_state

        obs = {
            f"{name}.pos": getattr(joints_msg, name) / DEG_MILLI_PER_RAD for name in JOINT_NAMES
        }
        obs[f"{GRIPPER_NAME}.pos"] = gripper_msg.grippers_angle / MILLI_MM_PER_METER
        return obs

    def _get_end_pose(self) -> dict[str, float]:
        assert self.piper is not None
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        return {
            "x": end_pose.X_axis / MILLI_MM_PER_METER,
            "y": end_pose.Y_axis / MILLI_MM_PER_METER,
            "z": end_pose.Z_axis / MILLI_MM_PER_METER,
            "rx": end_pose.RX_axis / MILLI_DEG_PER_RAD,
            "ry": end_pose.RY_axis / MILLI_DEG_PER_RAD,
            "rz": end_pose.RZ_axis / MILLI_DEG_PER_RAD,
        }

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        obs_dict: RobotObservation = self._get_motor_positions()
        self.logs["read_pos_dt_s"] = time.perf_counter() - start

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            self.logs[f"read_camera_{cam_key}_dt_s"] = time.perf_counter() - start
            if cam_key == "depth_camera_rgb" and getattr(cam, "use_depth", False):
                start = time.perf_counter()
                depth = cam.read_latest_depth()
                if depth.ndim == 2:
                    depth = depth[..., None]
                obs_dict["depth_camera"] = depth
                self.logs["read_camera_depth_camera_dt_s"] = time.perf_counter() - start

        return obs_dict

    def _clip_action(self, action: RobotAction) -> dict[str, float]:
        goal_pos = {
            key.removesuffix(".pos"): float(val) for key, val in action.items() if key.endswith(".pos")
        }
        if any(key.endswith(".delta") for key in action):
            present_pos = {key.removesuffix(".pos"): val for key, val in self._get_motor_positions().items()}
            for key, val in action.items():
                if key.endswith(".delta"):
                    name = key.removesuffix(".delta")
                    if name in present_pos:
                        goal_pos[name] = present_pos[name] + float(val)

        clipped = {}
        for name in JOINT_NAMES:
            if name in goal_pos:
                clipped[name] = _clip(goal_pos[name], self.config.joint_limits_rad[name])
        if GRIPPER_NAME in goal_pos:
            clipped[GRIPPER_NAME] = _clip(goal_pos[GRIPPER_NAME], self.config.gripper_limit_m)

        if self.config.max_relative_target is not None and clipped:
            present_pos = {key.removesuffix(".pos"): val for key, val in self._get_motor_positions().items()}
            goal_present_pos = {name: (value, present_pos[name]) for name, value in clipped.items()}
            clipped = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        return clipped

    def _apply_ee_safety(self, target: dict[str, float], pose: dict[str, float]) -> dict[str, float]:
        """Cap per-axis |target - pose| using ``max_relative_target`` when it is a float.

        The per-joint dict form of ``max_relative_target`` is reserved for the joint path.
        """
        cap = self.config.max_relative_target
        if not isinstance(cap, float):
            return target
        safe = {}
        for axis, val in target.items():
            base = pose[axis]
            diff = max(-cap, min(cap, val - base))
            safe[axis] = base + diff
        return safe

    def _send_ee_pose_target(self, target: dict[str, float]) -> None:
        assert self.piper is not None
        self.piper.MotionCtrl_2(0x01, 0x00, self.config.move_spd_rate_ctrl, 0x00)
        self.piper.EndPoseCtrl(
            round(target["x"] * MILLI_MM_PER_METER),
            round(target["y"] * MILLI_MM_PER_METER),
            round(target["z"] * MILLI_MM_PER_METER),
            round(target["rx"] * MILLI_DEG_PER_RAD),
            round(target["ry"] * MILLI_DEG_PER_RAD),
            round(target["rz"] * MILLI_DEG_PER_RAD),
        )

    def _send_ee_delta_action(self, action: RobotAction) -> None:
        assert self.piper is not None
        if not any(key.startswith("ee.delta_") for key in action):
            return

        pose = self._get_end_pose()
        target = {
            "x": pose["x"] + float(action.get("ee.delta_x", 0.0)),
            "y": pose["y"] + float(action.get("ee.delta_y", 0.0)),
            "z": pose["z"] + float(action.get("ee.delta_z", 0.0)),
            "rx": pose["rx"] + float(action.get("ee.delta_rx", 0.0)),
            "ry": pose["ry"] + float(action.get("ee.delta_ry", 0.0)),
            "rz": pose["rz"] + float(action.get("ee.delta_rz", 0.0)),
        }

        self._send_ee_pose_target(self._apply_ee_safety(target, pose))

    def _send_ee_abs_action(self, action: RobotAction) -> None:
        assert self.piper is not None
        if not any(key.startswith("ee.abs_") for key in action):
            return

        pose = self._get_end_pose()
        target = {
            "x": float(action.get("ee.abs_x", pose["x"])),
            "y": float(action.get("ee.abs_y", pose["y"])),
            "z": float(action.get("ee.abs_z", pose["z"])),
            "rx": float(action.get("ee.abs_rx", pose["rx"])),
            "ry": float(action.get("ee.abs_ry", pose["ry"])),
            "rz": float(action.get("ee.abs_rz", pose["rz"])),
        }
        self._send_ee_pose_target(self._apply_ee_safety(target, pose))

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        assert self.piper is not None
        goal_pos = self._clip_action(action)
        self._send_ee_delta_action(action)
        self._send_ee_abs_action(action)

        if any(name in goal_pos for name in JOINT_NAMES):
            current = self._get_motor_positions()
            joints = [
                round(goal_pos.get(name, current[f"{name}.pos"]) * DEG_MILLI_PER_RAD) for name in JOINT_NAMES
            ]
            self.piper.MotionCtrl_2(0x01, 0x01, self.config.move_spd_rate_ctrl, 0x00)
            self.piper.JointCtrl(*joints)

        if GRIPPER_NAME in goal_pos:
            gripper = round(abs(goal_pos[GRIPPER_NAME]) * MILLI_MM_PER_METER)
            self.piper.GripperCtrl(gripper, self.config.gripper_effort, 0x01, 0)

        return {f"{name}.pos": value for name, value in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        assert self.piper is not None
        if self.config.disable_on_disconnect:
            self.piper.DisablePiper()
        self.piper = None
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")
