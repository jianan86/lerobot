"""Client-side adapter for pose_act Piper observations/actions.

The real pose_act server path returns base-frame absolute pose7d
``[x, y, z, roll, pitch, yaw, gripper]``. The shell path still returns the
older relative pose10d chunks used for endpoint testing, so this adapter
supports both.

For shell pose10d, degenerate relative rotations (all-zero rot6d) are
substituted with the identity 6D vector ``[1, 0, 0, 0, 1, 0]`` so that the zero
action means "stay at current pose".
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.pose_act.utils import (
    absolute_pose10d,
    pose10d_to_pose7d,
    pose7d_to_pose10d,
)

logger = logging.getLogger(__name__)

_IDENTITY_ROT6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
POSE7D_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper_width")


def is_pose_act_piper(policy_type: str, robot_type: str) -> bool:
    """Adapter is used when the policy is pose_act and the robot is any Piper-like follower."""
    return policy_type == "pose_act" and robot_type in ("piper_follower", "mock_piper_follower")


class PoseActPiperAdapter:
    """Stateful adapter; holds a handle to the robot so it can read current TCP on demand."""

    def __init__(self, robot: Any):
        self.robot = robot
        self._last_log_t = 0.0

    def current_pose7d(self) -> Tensor:
        """Read Piper TCP and gripper as base-frame pose7d."""
        get_end_pose = getattr(self.robot, "_get_end_pose", None)
        if get_end_pose is None:
            raise RuntimeError(
                f"Robot {type(self.robot).__name__} does not expose _get_end_pose(); "
                f"cannot build pose7d observation for pose_act adapter."
            )
        pose = get_end_pose()
        gripper = 0.0
        get_motor_positions = getattr(self.robot, "_get_motor_positions", None)
        if get_motor_positions is not None:
            motors = get_motor_positions()
            gripper = float(motors.get("gripper.pos", 0.0))
        return torch.tensor(
            [
                float(pose["x"]),
                float(pose["y"]),
                float(pose["z"]),
                float(pose["rx"]),
                float(pose["ry"]),
                float(pose["rz"]),
                float(gripper),
            ],
            dtype=torch.float32,
        )

    def _current_base_pose10d(self) -> Tensor:
        return pose7d_to_pose10d(self.current_pose7d())

    def _pose7d_to_action_dict(self, pose7d: Tensor) -> dict[str, float]:
        return {
            "ee.abs_x": float(pose7d[0].item()),
            "ee.abs_y": float(pose7d[1].item()),
            "ee.abs_z": float(pose7d[2].item()),
            "ee.abs_rx": float(pose7d[3].item()),
            "ee.abs_ry": float(pose7d[4].item()),
            "ee.abs_rz": float(pose7d[5].item()),
            "gripper.pos": max(0.0, float(pose7d[6].item())),
        }

    def convert(self, action_tensor: Tensor) -> dict[str, float]:
        """Convert a single pose_act action into an ``ee.abs_*`` dict."""
        if action_tensor.ndim != 1 or action_tensor.shape[0] not in (7, 10):
            raise ValueError(
                f"PoseActPiperAdapter expects a (7,) pose7d or (10,) pose10d tensor, got {tuple(action_tensor.shape)}"
            )
        if action_tensor.shape[0] == 7:
            return self._pose7d_to_action_dict(action_tensor.detach().to(torch.float32).cpu())

        rel = action_tensor.detach().to(torch.float32).cpu().clone()
        rot6d = rel[3:9]
        if torch.allclose(rot6d, torch.zeros_like(rot6d), atol=1e-8):
            rel[3:9] = _IDENTITY_ROT6D
            # Assert rot6d -> matrix is valid; catch weird near-zero predictions
        try:
            pose10d_to_pose7d(rel)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Bad rot6d {rel[3:9].tolist()}: {e}; substituting identity")
            rel[3:9] = _IDENTITY_ROT6D

        base = self._current_base_pose10d()
        absolute = absolute_pose10d(rel, base)
        return self._pose7d_to_action_dict(pose10d_to_pose7d(absolute))
