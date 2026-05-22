"""Client-side adapter for pose_act Piper observations/actions.

The real pose_act server path returns base-frame absolute pose7d
``[x, y, z, roll, pitch, yaw, gripper]``. The shell path still returns the
older relative pose10d chunks used for endpoint testing, so this adapter
supports both.

For shell pose10d, degenerate relative rotations (all-zero rot6d) are
substituted with the identity 6D vector ``[1, 0, 0, 0, 1, 0]`` so that the zero
action means "stay at current pose".

The fixed TCP extrinsics are encoded as a single homogeneous transform
``T_EE_TCP`` whose physical meaning is:

- ``p_ee = T_EE_TCP @ p_tcp``
- ``T_base_tcp = T_base_ee @ T_EE_TCP``

The transform captures both:

- translation: TCP origin is ``0.1943 m`` along ``+z`` of the EE frame
- rotation: ``tcp.x = ee.z``, ``tcp.y = ee.y``, ``tcp.z = -ee.x``
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from lerobot.utils.pose_act import (
    POSE7D_NAMES as POSE7D_NAMES,
    absolute_pose10d,
    euler_rpy_to_matrix,
    matrix_to_euler_rpy,
    pose7d_to_pose10d,
    pose10d_to_pose7d,
)

logger = logging.getLogger(__name__)
POSE_POLICY_TYPES = {"pose_act", "pose_smolvla"}

_IDENTITY_ROT6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
R_EE_TCP = torch.tensor(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)
t_EE_TCP = torch.tensor([0.0, 0.0, 0.1943], dtype=torch.float32)
T_EE_TCP = torch.eye(4, dtype=torch.float32)
T_EE_TCP[:3, :3] = R_EE_TCP
T_EE_TCP[:3, 3] = t_EE_TCP
T_TCP_EE = torch.linalg.inv(T_EE_TCP)


def _pose7d_to_transform(pose7d: Tensor) -> Tensor:
    if pose7d.shape != (7,):
        raise ValueError(f"Expected pose7d shape (7,), got {tuple(pose7d.shape)}")
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = euler_rpy_to_matrix(pose7d[3:6])
    transform[:3, 3] = pose7d[:3]
    return transform


def _transform_to_pose7d(transform: Tensor, gripper: float) -> Tensor:
    if transform.shape != (4, 4):
        raise ValueError(f"Expected homogeneous transform shape (4, 4), got {tuple(transform.shape)}")
    pose7d = torch.empty(7, dtype=torch.float32)
    pose7d[:3] = transform[:3, 3]
    pose7d[3:6] = matrix_to_euler_rpy(transform[:3, :3])
    pose7d[6] = gripper
    return pose7d


def is_pose_act_piper(policy_type: str, robot_type: str) -> bool:
    """Adapter is used when the policy consumes TCP pose history and the robot is Piper-like."""
    return policy_type in POSE_POLICY_TYPES and robot_type == "piper_follower"


class PoseActPiperAdapter:
    """Stateful adapter; holds a handle to the robot so it can read current TCP on demand."""

    def __init__(self, robot: Any, gripper_width_offset: float = 0.0):
        self.robot = robot
        self.gripper_width_offset = float(gripper_width_offset)
        self._last_log_t = 0.0

    def ee_pose7d_to_tcp_pose7d(self, ee_pose7d: Tensor) -> Tensor:
        pose = ee_pose7d.detach().to(torch.float32).cpu()
        base_to_ee = _pose7d_to_transform(pose)
        base_to_tcp = base_to_ee @ T_EE_TCP
        return _transform_to_pose7d(base_to_tcp, float(pose[6].item()))

    def tcp_pose7d_to_ee_pose7d(self, tcp_pose7d: Tensor) -> Tensor:
        pose = tcp_pose7d.detach().to(torch.float32).cpu()
        base_to_tcp = _pose7d_to_transform(pose)
        base_to_ee = base_to_tcp @ T_TCP_EE
        return _transform_to_pose7d(base_to_ee, float(pose[6].item()))

    def current_pose7d(self) -> Tensor:
        """Read Piper EE pose and convert it to base-frame TCP pose7d."""
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
        ee_pose7d = torch.tensor(
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
        return self.ee_pose7d_to_tcp_pose7d(ee_pose7d)

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
            "gripper.pos": max(0.0, float(pose7d[6].item()) + self.gripper_width_offset),
        }

    def convert(self, action_tensor: Tensor) -> dict[str, float]:
        """Convert a single pose_act TCP action into an ``ee.abs_*`` dict."""
        if action_tensor.ndim != 1 or action_tensor.shape[0] not in (7, 10):
            raise ValueError(
                f"PoseActPiperAdapter expects a (7,) pose7d or (10,) pose10d tensor, got {tuple(action_tensor.shape)}"
            )
        if action_tensor.shape[0] == 7:
            tcp_pose7d = action_tensor.detach().to(torch.float32).cpu()
            return self._pose7d_to_action_dict(self.tcp_pose7d_to_ee_pose7d(tcp_pose7d))

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
        tcp_pose7d = pose10d_to_pose7d(absolute)
        return self._pose7d_to_action_dict(self.tcp_pose7d_to_ee_pose7d(tcp_pose7d))
