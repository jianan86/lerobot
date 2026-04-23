#!/usr/bin/env python

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


POSE7D_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper_width")
POSE10D_NAMES = (
    "x",
    "y",
    "z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper_width",
)


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    proj = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - proj * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(rot_mats: Tensor) -> Tensor:
    return torch.cat((rot_mats[..., :, 0], rot_mats[..., :, 1]), dim=-1)


def euler_rpy_to_matrix(rpy: Tensor) -> Tensor:
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
    if pose7d.shape[-1] != 7:
        raise ValueError(f"pose7d expects final dimension 7, got {pose7d.shape[-1]}")
    pos = pose7d[..., :3]
    rot = euler_rpy_to_matrix(pose7d[..., 3:6])
    grip = pose7d[..., 6:7]
    return combine_pose10d(pos, rot, grip)


def pose10d_to_pose7d(pose10d: Tensor) -> Tensor:
    pos, rot, grip = split_pose10d(pose10d)
    return torch.cat((pos, matrix_to_euler_rpy(rot), grip), dim=-1)


def ensure_pose10d(pose: Tensor) -> Tensor:
    if pose.shape[-1] == 10:
        return pose
    if pose.shape[-1] == 7:
        return pose7d_to_pose10d(pose)
    raise ValueError(f"Expected pose final dimension 7 or 10, got {pose.shape[-1]}")


def split_pose10d(pose: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if pose.shape[-1] != 10:
        raise ValueError(f"pose10d expects final dimension 10, got {pose.shape[-1]}")
    pos = pose[..., :3]
    rot = rotation_6d_to_matrix(pose[..., 3:9])
    grip = pose[..., 9:10]
    return pos, rot, grip


def combine_pose10d(pos: Tensor, rot: Tensor, grip: Tensor) -> Tensor:
    return torch.cat((pos, matrix_to_rotation_6d(rot), grip), dim=-1)


def relative_pose10d(poses: Tensor, base_pose: Tensor) -> Tensor:
    base_pose = ensure_pose10d(base_pose)
    poses = ensure_pose10d(poses)
    base_pos, base_rot, _ = split_pose10d(base_pose)
    pos, rot, grip = split_pose10d(poses)

    base_rot_t = base_rot.transpose(-1, -2)
    if poses.ndim == base_pose.ndim + 1:
        pos_rel = torch.matmul(base_rot_t.unsqueeze(-3), (pos - base_pos.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        rot_rel = torch.matmul(base_rot_t.unsqueeze(-3), rot)
    else:
        pos_rel = torch.matmul(base_rot_t, (pos - base_pos).unsqueeze(-1)).squeeze(-1)
        rot_rel = torch.matmul(base_rot_t, rot)

    return combine_pose10d(pos_rel, rot_rel, grip)


def absolute_pose10d(relative_poses: Tensor, base_pose: Tensor) -> Tensor:
    base_pose = ensure_pose10d(base_pose)
    relative_poses = ensure_pose10d(relative_poses)
    base_pos, base_rot, _ = split_pose10d(base_pose)
    pos_rel, rot_rel, grip = split_pose10d(relative_poses)

    if relative_poses.ndim == base_pose.ndim + 1:
        pos_abs = torch.matmul(base_rot.unsqueeze(-3), pos_rel.unsqueeze(-1)).squeeze(-1) + base_pos.unsqueeze(-2)
        rot_abs = torch.matmul(base_rot.unsqueeze(-3), rot_rel)
    else:
        pos_abs = torch.matmul(base_rot, pos_rel.unsqueeze(-1)).squeeze(-1) + base_pos
        rot_abs = torch.matmul(base_rot, rot_rel)

    return combine_pose10d(pos_abs, rot_abs, grip)


def make_relative_state_history(states: Tensor) -> Tensor:
    if states.ndim != 3:
        raise ValueError(f"Expected state history with shape (B, T, D), got {tuple(states.shape)}")
    states = ensure_pose10d(states)
    return relative_pose10d(states, states[:, -1])


def poses_to_relative_10d(states: Tensor, actions: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
    states = ensure_pose10d(states)
    if states.ndim == 2:
        # Ambiguous case: either (B, D) or an unbatched history (T, D).
        if actions is not None and (actions.ndim == 1 or (actions.ndim >= 2 and actions.shape[0] != states.shape[0])):
            base_pose = states[-1]
            rel_states = relative_pose10d(states, base_pose)
        else:
            base_pose = states
            rel_states = relative_pose10d(states, base_pose)
    elif states.ndim == 3:
        base_pose = states[:, -1]
        rel_states = relative_pose10d(states, base_pose)
    else:
        raise ValueError(f"Expected state shape (B, D) or (B, T, D), got {tuple(states.shape)}")

    rel_actions = None
    if actions is not None:
        rel_actions = relative_pose10d(ensure_pose10d(actions), base_pose)
    return rel_states, rel_actions


def pose10d_umi_scale_offset(stats: dict[str, Tensor], eps: float = 1e-7) -> tuple[Tensor, Tensor]:
    """Build UMI-style normalizer params for pose10d: range for xyz/gripper, identity for rot6d."""
    min_v = stats.get("min")
    max_v = stats.get("max")
    if min_v is None or max_v is None:
        raise ValueError("UMI pose normalization requires min/max stats.")
    min_v = min_v.to(dtype=torch.float32)
    max_v = max_v.to(dtype=torch.float32)
    if min_v.shape[-1] == 7:
        min_pose = pose7d_to_pose10d(min_v)
        max_pose = pose7d_to_pose10d(max_v)
        min_v = torch.cat((min_pose[..., :3], min_pose[..., 9:10]), dim=-1)
        max_v = torch.cat((max_pose[..., :3], max_pose[..., 9:10]), dim=-1)
    elif min_v.shape[-1] == 10:
        min_v = torch.cat((min_v[..., :3], min_v[..., 9:10]), dim=-1)
        max_v = torch.cat((max_v[..., :3], max_v[..., 9:10]), dim=-1)
    else:
        raise ValueError(f"Expected pose stats width 7 or 10, got {min_v.shape[-1]}")

    input_range = max_v - min_v
    ignore_dim = input_range < eps
    input_range = torch.where(ignore_dim, torch.ones_like(input_range), input_range)
    scale_4d = 2.0 / input_range
    offset_4d = -scale_4d * (min_v + max_v) / 2.0
    scale_4d = torch.where(ignore_dim, torch.ones_like(scale_4d), scale_4d)
    offset_4d = torch.where(ignore_dim, torch.zeros_like(offset_4d), offset_4d)

    scale = torch.ones(10, dtype=torch.float32, device=scale_4d.device)
    offset = torch.zeros(10, dtype=torch.float32, device=offset_4d.device)
    scale[:3] = scale_4d[:3]
    offset[:3] = offset_4d[:3]
    scale[9] = scale_4d[3]
    offset[9] = offset_4d[3]
    return scale, offset


def normalize_pose10d_umi(pose: Tensor, scale: Tensor, offset: Tensor) -> Tensor:
    return pose * scale.to(device=pose.device, dtype=pose.dtype) + offset.to(device=pose.device, dtype=pose.dtype)


def unnormalize_pose10d_umi(pose: Tensor, scale: Tensor, offset: Tensor) -> Tensor:
    scale = scale.to(device=pose.device, dtype=pose.dtype)
    offset = offset.to(device=pose.device, dtype=pose.dtype)
    return (pose - offset) / scale
