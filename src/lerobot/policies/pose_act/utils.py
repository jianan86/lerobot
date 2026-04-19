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
