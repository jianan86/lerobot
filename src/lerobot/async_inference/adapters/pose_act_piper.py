"""Client-side adapter: pose_act relative pose10d -> Piper EndPoseCtrl action dict.

The async policy server returns chunks of **relative** pose10d
(pos(3) + rot6d(6) + gripper(1)). This adapter:

1. Reads Piper's current TCP (x, y, z, rx, ry, rz in meters/rad) and current gripper.
2. Builds a base pose10d tensor.
3. Calls :func:`absolute_pose10d` to map each relative step into world-frame absolute.
4. Decomposes absolute pose10d back into (xyz, euler rx/ry/rz, gripper).
5. Returns a dict usable by ``PiperFollower.send_action`` via the new ``ee.abs_*`` keys.

Degenerate relative rotations (all-zero rot6d, produced by the shell server) are
substituted with the identity 6D vector ``[1, 0, 0, 0, 1, 0]`` so that the zero
action means "stay at current pose".
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.pose_act.utils import (
    absolute_pose10d,
    combine_pose10d,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

logger = logging.getLogger(__name__)

_IDENTITY_ROT6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def is_pose_act_piper(policy_type: str, robot_type: str) -> bool:
    """Adapter is used when the policy is pose_act and the robot is any Piper-like follower."""
    return policy_type == "pose_act" and robot_type in ("piper_follower", "mock_piper_follower")


def _euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> Tensor:
    """Euler (XYZ extrinsic / ZYX intrinsic) to rotation matrix.

    Matches Piper's convention where EndPoseCtrl consumes RX, RY, RZ as roll/pitch/yaw.
    """
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = torch.tensor([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = torch.tensor([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_m @ ry_m @ rx_m


def _matrix_to_euler_xyz(R: Tensor) -> tuple[float, float, float]:
    """Inverse of :func:`_euler_xyz_to_matrix`. Returns (rx, ry, rz) in radians."""
    r20 = float(R[2, 0].item())
    r20 = max(-1.0, min(1.0, r20))
    ry = math.asin(-r20)
    if abs(math.cos(ry)) > 1e-6:
        rx = math.atan2(float(R[2, 1].item()), float(R[2, 2].item()))
        rz = math.atan2(float(R[1, 0].item()), float(R[0, 0].item()))
    else:  # Gimbal lock; pick a consistent branch
        rx = math.atan2(-float(R[1, 2].item()), float(R[1, 1].item()))
        rz = 0.0
    return rx, ry, rz


def _base_pose10d(pose_xyz_rxryrz: dict[str, float], gripper_m: float) -> Tensor:
    rot_mat = _euler_xyz_to_matrix(
        float(pose_xyz_rxryrz["rx"]),
        float(pose_xyz_rxryrz["ry"]),
        float(pose_xyz_rxryrz["rz"]),
    )
    rot6d = matrix_to_rotation_6d(rot_mat)
    pos = torch.tensor(
        [float(pose_xyz_rxryrz["x"]), float(pose_xyz_rxryrz["y"]), float(pose_xyz_rxryrz["z"])]
    )
    grip = torch.tensor([float(gripper_m)])
    return combine_pose10d(pos, rot_mat, grip)  # uses matrix_to_rotation_6d internally
    # (rot6d variable kept for clarity; combine_pose10d reuses matrix_to_rotation_6d)


class PoseActPiperAdapter:
    """Stateful adapter; holds a handle to the robot so it can read current TCP on demand."""

    def __init__(self, robot: Any):
        self.robot = robot
        self._last_log_t = 0.0

    def _current_base_pose10d(self) -> Tensor:
        get_end_pose = getattr(self.robot, "_get_end_pose", None)
        if get_end_pose is None:
            raise RuntimeError(
                f"Robot {type(self.robot).__name__} does not expose _get_end_pose(); "
                f"cannot build pose10d base for pose_act adapter."
            )
        pose = get_end_pose()
        gripper = 0.0
        get_motor_positions = getattr(self.robot, "_get_motor_positions", None)
        if get_motor_positions is not None:
            motors = get_motor_positions()
            gripper = float(motors.get("gripper.pos", 0.0))
        return _base_pose10d(pose, gripper)

    def convert(self, action_tensor: Tensor) -> dict[str, float]:
        """Convert a single relative pose10d (shape ``(10,)``) into an ``ee.abs_*`` dict."""
        if action_tensor.ndim != 1 or action_tensor.shape[0] != 10:
            raise ValueError(
                f"PoseActPiperAdapter expects a (10,) pose10d tensor, got {tuple(action_tensor.shape)}"
            )
        rel = action_tensor.detach().to(torch.float32).cpu().clone()
        rot6d = rel[3:9]
        if torch.allclose(rot6d, torch.zeros_like(rot6d), atol=1e-8):
            rel[3:9] = _IDENTITY_ROT6D
            # Assert rot6d -> matrix is valid; catch weird near-zero predictions
        try:
            rotation_6d_to_matrix(rel[3:9].unsqueeze(0))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Bad rot6d {rel[3:9].tolist()}: {e}; substituting identity")
            rel[3:9] = _IDENTITY_ROT6D

        base = self._current_base_pose10d()
        absolute = absolute_pose10d(rel, base)
        pos = absolute[:3]
        rot_mat = rotation_6d_to_matrix(absolute[3:9].unsqueeze(0))[0]
        grip_abs = float(absolute[9].item())
        rx, ry, rz = _matrix_to_euler_xyz(rot_mat)

        return {
            "ee.abs_x": float(pos[0].item()),
            "ee.abs_y": float(pos[1].item()),
            "ee.abs_z": float(pos[2].item()),
            "ee.abs_rx": float(rx),
            "ee.abs_ry": float(ry),
            "ee.abs_rz": float(rz),
            "gripper.pos": max(0.0, grip_abs),
        }
