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
import os
import sys
from queue import Queue
from typing import Any

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available, require_package

from ..teleoperator import Teleoperator
from .config_keyboard_piper import KeyboardPiperEndEffectorTeleopConfig, KeyboardPiperJointTeleopConfig

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
            logging.info("No DISPLAY set. Skipping pynput import.")
            PYNPUT_AVAILABLE = False
        else:
            from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info(f"Could not import pynput: {e}")


class _KeyboardPiperBase(Teleoperator):
    def __init__(self, config):
        require_package("pynput", extra="pynput-dep")
        super().__init__(config)
        self.config = config
        self.event_queue = Queue()
        self.current_pressed: dict[str, bool] = {}
        self.listener = None

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if PYNPUT_AVAILABLE:
            self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self.listener.start()
        else:
            self.listener = None

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _on_press(self, key) -> None:
        if hasattr(key, "char") and key.char is not None:
            self.event_queue.put((key.char.lower(), True))

    def _on_release(self, key) -> None:
        if hasattr(key, "char") and key.char is not None:
            self.event_queue.put((key.char.lower(), False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self) -> None:
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                self.current_pressed.pop(key_char, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()


class KeyboardPiperJointTeleop(_KeyboardPiperBase):
    config_class = KeyboardPiperJointTeleopConfig
    name = "keyboard_piper_joint"

    _KEY_BINDINGS = {
        "q": ("joint_1.delta", 1.0),
        "a": ("joint_1.delta", -1.0),
        "w": ("joint_2.delta", 1.0),
        "s": ("joint_2.delta", -1.0),
        "e": ("joint_3.delta", 1.0),
        "d": ("joint_3.delta", -1.0),
        "r": ("joint_4.delta", 1.0),
        "f": ("joint_4.delta", -1.0),
        "t": ("joint_5.delta", 1.0),
        "g": ("joint_5.delta", -1.0),
        "y": ("joint_6.delta", 1.0),
        "h": ("joint_6.delta", -1.0),
        "u": ("gripper.delta", 1.0),
        "j": ("gripper.delta", -1.0),
    }

    @property
    def action_features(self) -> dict:
        return {feature: float for feature, _ in self._KEY_BINDINGS.values()}

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        if "x" in self.current_pressed:
            return {}

        action: RobotAction = {}
        for key in self.current_pressed:
            if key not in self._KEY_BINDINGS:
                continue
            feature, direction = self._KEY_BINDINGS[key]
            step = self.config.gripper_step if feature == "gripper.delta" else self.config.joint_step
            action[feature] = action.get(feature, 0.0) + direction * step
        return action


class KeyboardPiperEndEffectorTeleop(_KeyboardPiperBase):
    config_class = KeyboardPiperEndEffectorTeleopConfig
    name = "keyboard_piper_ee"

    _KEY_BINDINGS = {
        "w": ("ee.delta_x", 1.0),
        "s": ("ee.delta_x", -1.0),
        "a": ("ee.delta_y", 1.0),
        "d": ("ee.delta_y", -1.0),
        "q": ("ee.delta_z", 1.0),
        "e": ("ee.delta_z", -1.0),
        "r": ("ee.delta_rx", 1.0),
        "f": ("ee.delta_rx", -1.0),
        "t": ("ee.delta_ry", 1.0),
        "g": ("ee.delta_ry", -1.0),
        "y": ("ee.delta_rz", 1.0),
        "h": ("ee.delta_rz", -1.0),
        "u": ("gripper.delta", 1.0),
        "j": ("gripper.delta", -1.0),
    }

    @property
    def action_features(self) -> dict:
        return {feature: float for feature, _ in self._KEY_BINDINGS.values()}

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        if "x" in self.current_pressed:
            return {}

        action: RobotAction = {}
        for key in self.current_pressed:
            if key not in self._KEY_BINDINGS:
                continue
            feature, direction = self._KEY_BINDINGS[key]
            if feature == "gripper.delta":
                step = self.config.gripper_step
            elif feature in {"ee.delta_rx", "ee.delta_ry", "ee.delta_rz"}:
                step = self.config.rpy_step
            else:
                step = self.config.xyz_step
            action[feature] = action.get(feature, 0.0) + direction * step
        return action
