import logging
import time
from functools import cached_property

import numpy as np

from lerobot.types import RobotAction, RobotObservation

from ..robot import Robot
from .config_mock_piper_follower import MockPiperFollowerConfig

logger = logging.getLogger(__name__)

JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
GRIPPER_NAME = "gripper"


class MockPiperFollower(Robot):
    """Mock robot matching PiperFollower's observation/action schema.

    Observations are random within plausible ranges; send_action logs incoming dicts
    and throttles to the configured fps. Used to validate the async-inference
    pipeline end-to-end without touching hardware.
    """

    config_class = MockPiperFollowerConfig
    name = "mock_piper_follower"

    def __init__(self, config: MockPiperFollowerConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._rng = np.random.default_rng(0)
        self._action_count = 0
        self._last_send_t: float | None = None
        self._joint_state = {f"{m}.pos": 0.0 for m in (*JOINT_NAMES, GRIPPER_NAME)}
        self._ee_state = {"x": 0.2, "y": 0.0, "z": 0.3, "rx": 0.0, "ry": 0.0, "rz": 0.0}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in (*JOINT_NAMES, GRIPPER_NAME)}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {name: (spec.height, spec.width, 3) for name, spec in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        joint_deltas = {f"{m}.delta": float for m in (*JOINT_NAMES, GRIPPER_NAME)}
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
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self._connected = True
        logger.info(f"{self} connected (mock).")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _get_motor_positions(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._joint_state.items()}

    def _get_end_pose(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._ee_state.items()}

    def get_observation(self) -> RobotObservation:
        if not self._connected:
            raise RuntimeError(f"{self} is not connected")

        obs: RobotObservation = {
            f"{m}.pos": float(self._joint_state[f"{m}.pos"] + self._rng.normal(0, 1e-3))
            for m in JOINT_NAMES
        }
        obs[f"{GRIPPER_NAME}.pos"] = float(self._joint_state[f"{GRIPPER_NAME}.pos"])
        for cam, (h, w, c) in self._cameras_ft.items():
            obs[cam] = self._rng.integers(0, 255, size=(h, w, c), dtype=np.uint8)
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._connected:
            raise RuntimeError(f"{self} is not connected")

        now = time.perf_counter()
        if self._last_send_t is not None:
            dt = 1.0 / max(1, self.config.fps)
            elapsed = now - self._last_send_t
            if elapsed < dt:
                time.sleep(dt - elapsed)
        self._last_send_t = time.perf_counter()

        self._action_count += 1
        if self.config.verbose and self._action_count % max(1, self.config.print_every_n) == 0:
            keys = ",".join(sorted(action.keys())[:6])
            sample = {k: float(v) for k, v in list(action.items())[:6]}
            logger.info(
                f"[MockPiperFollower] action#{self._action_count} keys=[{keys}...] sample={sample}"
            )

        for key, val in action.items():
            if key.endswith(".pos") and key in self._joint_state:
                self._joint_state[key] = float(val)
            elif key.endswith(".delta"):
                name = key.removesuffix(".delta")
                pos_key = f"{name}.pos"
                if pos_key in self._joint_state:
                    self._joint_state[pos_key] += float(val)
            elif key.startswith("ee.delta_"):
                axis = key.removeprefix("ee.delta_")
                if axis in self._ee_state:
                    self._ee_state[axis] += float(val)
            elif key.startswith("ee.abs_"):
                axis = key.removeprefix("ee.abs_")
                if axis in self._ee_state:
                    self._ee_state[axis] = float(val)

        return {k: float(v) for k, v in action.items() if not hasattr(v, "shape")}

    def disconnect(self) -> None:
        self._connected = False
        logger.info(f"{self} disconnected (mock).")
