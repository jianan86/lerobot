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

"""Pose utilities for pose_act.

This module keeps the pose representation self-contained in lerobot:
- pose10d = position(3) + rotation6d(6) + gripper(1)
- relative pose is defined in SE(3) w.r.t. a base TCP pose
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """Convert 6D rotation representation to rotation matrices."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    proj = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - proj * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(rot_mats: Tensor) -> Tensor:
    """Convert rotation matrices to the 6D rotation representation."""
    return torch.cat((rot_mats[..., :, 0], rot_mats[..., :, 1]), dim=-1)


def euler_rpy_to_matrix(rpy: Tensor) -> Tensor:
    """Convert roll/pitch/yaw radians to rotation matrices.

    Uses the same convention as Piper EndPoseCtrl: roll, pitch, yaw map to
    ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """
    if rpy.shape[-1] != 3:
        raise ValueError(f"rpy expects final dimension 3, got {rpy.shape[-1]}")

    roll, pitch, yaw = rpy.unbind(dim=-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    row0 = torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), dim=-1)
    row1 = torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), dim=-1)
    row2 = torch.stack((-sp, cp * sr, cp * cr), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def matrix_to_euler_rpy(rot_mats: Tensor) -> Tensor:
    """Inverse of :func:`euler_rpy_to_matrix`; returns roll/pitch/yaw radians."""
    if rot_mats.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices with shape (..., 3, 3), got {tuple(rot_mats.shape)}")

    r20 = rot_mats[..., 2, 0].clamp(-1.0, 1.0)
    pitch = torch.asin(-r20)
    cos_pitch = torch.cos(pitch)

    roll_regular = torch.atan2(rot_mats[..., 2, 1], rot_mats[..., 2, 2])
    yaw_regular = torch.atan2(rot_mats[..., 1, 0], rot_mats[..., 0, 0])
    roll_locked = torch.atan2(-rot_mats[..., 1, 2], rot_mats[..., 1, 1])
    yaw_locked = torch.zeros_like(yaw_regular)

    regular = torch.abs(cos_pitch) > 1e-6
    roll = torch.where(regular, roll_regular, roll_locked)
    yaw = torch.where(regular, yaw_regular, yaw_locked)
    return torch.stack((roll, pitch, yaw), dim=-1)


def pose7d_to_pose10d(pose7d: Tensor) -> Tensor:
    """Convert ``[x, y, z, roll, pitch, yaw, gripper]`` to pose10d."""
    if pose7d.shape[-1] != 7:
        raise ValueError(f"pose7d expects final dimension 7, got {pose7d.shape[-1]}")
    pos = pose7d[..., :3]
    rot = euler_rpy_to_matrix(pose7d[..., 3:6])
    grip = pose7d[..., 6:7]
    return combine_pose10d(pos, rot, grip)


def pose10d_to_pose7d(pose10d: Tensor) -> Tensor:
    """Convert pose10d to ``[x, y, z, roll, pitch, yaw, gripper]``."""
    pos, rot, grip = split_pose10d(pose10d)
    return torch.cat((pos, matrix_to_euler_rpy(rot), grip), dim=-1)


def split_pose10d(pose: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Split pose10d into position, rotation matrix, and gripper width."""
    if pose.shape[-1] != 10:
        raise ValueError(f"pose10d expects final dimension 10, got {pose.shape[-1]}")
    pos = pose[..., :3]
    rot = rotation_6d_to_matrix(pose[..., 3:9])
    grip = pose[..., 9:10]
    return pos, rot, grip


def combine_pose10d(pos: Tensor, rot: Tensor, grip: Tensor) -> Tensor:
    """Combine position, rotation matrix, and gripper width into pose10d."""
    return torch.cat((pos, matrix_to_rotation_6d(rot), grip), dim=-1)


def relative_pose10d(poses: Tensor, base_pose: Tensor) -> Tensor:
    """Express absolute pose(s) relative to a base TCP pose."""
    base_pos, base_rot, _ = split_pose10d(base_pose)
    pos, rot, grip = split_pose10d(poses)

    base_rot_t = base_rot.transpose(-1, -2)
    if poses.ndim == 3:
        pos_rel = torch.matmul(base_rot_t.unsqueeze(1), (pos - base_pos.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot_rel = torch.matmul(base_rot_t.unsqueeze(1), rot)
    else:
        pos_rel = torch.matmul(base_rot_t, (pos - base_pos).unsqueeze(-1)).squeeze(-1)
        rot_rel = torch.matmul(base_rot_t, rot)

    return combine_pose10d(pos_rel, rot_rel, grip)


def absolute_pose10d(relative_poses: Tensor, base_pose: Tensor) -> Tensor:
    """Convert relative pose(s) back to absolute TCP pose(s)."""
    base_pos, base_rot, _ = split_pose10d(base_pose)
    pos_rel, rot_rel, grip = split_pose10d(relative_poses)

    if relative_poses.ndim == 3:
        pos_abs = torch.matmul(base_rot.unsqueeze(1), pos_rel.unsqueeze(-1)).squeeze(-1) + base_pos.unsqueeze(1)
        rot_abs = torch.matmul(base_rot.unsqueeze(1), rot_rel)
    else:
        pos_abs = torch.matmul(base_rot, pos_rel.unsqueeze(-1)).squeeze(-1) + base_pos
        rot_abs = torch.matmul(base_rot, rot_rel)

    return combine_pose10d(pos_abs, rot_abs, grip)


def make_relative_state_history(states: Tensor) -> Tensor:
    """Rewrite a TCP pose history into the current TCP frame."""
    if states.ndim != 3:
        raise ValueError(f"Expected state history with shape (B, T, 10), got {tuple(states.shape)}")
    return relative_pose10d(states, states[:, -1])
