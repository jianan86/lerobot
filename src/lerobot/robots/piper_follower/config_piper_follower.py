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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("piper_follower")
@dataclass
class PiperFollowerConfig(RobotConfig):
    can_name: str = "can0"
    move_spd_rate_ctrl: int = 30
    enable_on_connect: bool = True
    enable_timeout_s: float | None = 10.0
    disable_on_disconnect: bool = False
    gripper_effort: int = 1000

    # Caps absolute target changes against the latest feedback, in radians for joints and meters for gripper.
    max_relative_target: float | dict[str, float] | None = None

    # Absolute command limits exposed in LeRobot units.
    joint_limits_rad: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "joint_1": (-2.6179, 2.6179),
            "joint_2": (0.0, 3.14),
            "joint_3": (-2.967, 0.0),
            "joint_4": (-1.745, 1.745),
            "joint_5": (-1.22, 1.22),
            "joint_6": (-2.09439, 2.09439),
        }
    )
    gripper_limit_m: tuple[float, float] = (0.0, 0.08)

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
