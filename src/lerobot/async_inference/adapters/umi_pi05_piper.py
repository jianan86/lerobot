"""Client-side adapter for bimanual UMI PI0.5 Piper/Pika actions."""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot.async_inference.adapters.pose_act_piper import PoseActPiperAdapter
from lerobot.utils.pose_act import absolute_pose10d, pose7d_to_pose10d, pose10d_to_pose7d, relative_pose10d

_IDENTITY_ROT6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)


def replace_zero_rot6d_with_identity(pose10d: Tensor, *, atol: float = 1e-8) -> Tensor:
    pose = pose10d.detach().to(torch.float32).clone()
    if pose.shape[-1] != 10:
        raise ValueError(f"Expected pose10d final dimension 10, got {pose.shape[-1]}")
    rot6d = pose[..., 3:9]
    zero = torch.zeros_like(rot6d)
    mask = torch.all(torch.isclose(rot6d, zero, atol=atol, rtol=0), dim=-1)
    pose[..., 3:9] = torch.where(mask.unsqueeze(-1), _IDENTITY_ROT6D.to(pose.device), rot6d)
    return pose


def _pose7d_bimanual_to_pose10d(tcp_pose7_bimanual: Tensor) -> Tensor:
    pose = tcp_pose7_bimanual.detach().to(torch.float32).cpu()
    if pose.shape != (14,):
        raise ValueError(f"Expected bimanual TCP pose7 shape (14,), got {tuple(pose.shape)}")
    return torch.cat((pose7d_to_pose10d(pose[:7]), pose7d_to_pose10d(pose[7:])), dim=0)


def build_relative_state(previous_tcp_pose7: Tensor, current_tcp_pose7: Tensor) -> Tensor:
    """Build OpenPI UMI-style right-then-left relative observation.state."""
    previous = previous_tcp_pose7.detach().to(torch.float32).cpu()
    current = current_tcp_pose7.detach().to(torch.float32).cpu()
    if previous.shape != (14,) or current.shape != (14,):
        raise ValueError(
            f"Expected previous/current bimanual TCP pose7 shapes (14,), got "
            f"{tuple(previous.shape)} and {tuple(current.shape)}"
        )

    right = relative_pose10d(pose7d_to_pose10d(previous[:7]), pose7d_to_pose10d(current[:7]))
    left = relative_pose10d(pose7d_to_pose10d(previous[7:]), pose7d_to_pose10d(current[7:]))
    return torch.cat((right, left), dim=0)


def relative_actions_to_absolute_tcp(relative_actions: Tensor, base_state: Tensor) -> Tensor:
    """Convert UMI bimanual relative TCP pose10 actions to absolute TCP pose7 actions.

    Args:
        relative_actions: Tensor shaped ``(T, 20)`` as right pose10 + left pose10.
        base_state: Tensor shaped ``(14,)`` with current absolute TCP pose7 or
            ``(20,)`` with current right pose10 + left pose10.
    """
    actions = relative_actions.detach().to(torch.float32).cpu()
    state = base_state.detach().to(torch.float32).cpu()
    if actions.ndim != 2 or actions.shape[1] != 20:
        raise ValueError(f"Expected relative actions shape (T,20), got {tuple(actions.shape)}")
    if state.shape == (14,):
        state = _pose7d_bimanual_to_pose10d(state)
    elif state.shape != (20,):
        raise ValueError(f"Expected base state shape (14,) or (20,), got {tuple(state.shape)}")

    right_rel = replace_zero_rot6d_with_identity(actions[:, :10])
    left_rel = replace_zero_rot6d_with_identity(actions[:, 10:])
    right_abs = pose10d_to_pose7d(absolute_pose10d(right_rel, state[:10]))
    left_abs = pose10d_to_pose7d(absolute_pose10d(left_rel, state[10:]))
    return torch.cat((right_abs, left_abs), dim=-1)


def accelerate_gripper_closure(actions: Tensor, *, close_threshold: float = 0.04, target_width: float = 0.0) -> Tensor:
    """Push closing gripper targets to ``target_width`` for both arms."""
    adjusted = actions.detach().to(torch.float32).clone()
    if adjusted.ndim != 2 or adjusted.shape[1] != 14:
        raise ValueError(f"Expected absolute bimanual pose7 actions shape (T,14), got {tuple(adjusted.shape)}")
    for idx in (6, 13):
        adjusted[:, idx] = torch.where(
            adjusted[:, idx] < close_threshold,
            torch.full_like(adjusted[:, idx], target_width),
            adjusted[:, idx],
        )
    return adjusted


def limit_bimanual_pose7_steps(actions: Tensor, *, max_xyz_step: float = 0.03, max_rpy_step: float = 0.35) -> Tensor:
    limited = actions.detach().to(torch.float32).clone()
    if limited.ndim != 2 or limited.shape[1] != 14:
        raise ValueError(f"Expected absolute bimanual pose7 actions shape (T,14), got {tuple(limited.shape)}")
    if limited.shape[0] <= 1:
        return limited

    for t in range(1, limited.shape[0]):
        for offset in (0, 7):
            prev = limited[t - 1, offset : offset + 7]
            cur = limited[t, offset : offset + 7]
            cur[:3] = prev[:3] + (cur[:3] - prev[:3]).clamp(-max_xyz_step, max_xyz_step)
            cur[3:6] = prev[3:6] + (cur[3:6] - prev[3:6]).clamp(-max_rpy_step, max_rpy_step)
            limited[t, offset : offset + 7] = cur
    return limited


class UmiPI05PiperAdapter:
    def __init__(self, right_robot, left_robot, *, right_gripper=None, left_gripper=None):
        self.right = PoseActPiperAdapter(right_robot, gripper=right_gripper)
        self.left = PoseActPiperAdapter(left_robot, gripper=left_gripper)

    def current_tcp_pose10_state(self) -> Tensor:
        right = pose7d_to_pose10d(self.right.current_pose7d())
        left = pose7d_to_pose10d(self.left.current_pose7d())
        return torch.cat((right, left), dim=0)

    def current_tcp_pose7_state(self) -> Tensor:
        right = self.right.current_pose7d()
        left = self.left.current_pose7d()
        return torch.cat((right, left), dim=0)

    def tcp_pose7_to_ee_actions(self, tcp_pose7_bimanual: Tensor) -> tuple[dict[str, float], dict[str, float]]:
        pose = tcp_pose7_bimanual.detach().to(torch.float32).cpu()
        if pose.shape != (14,):
            raise ValueError(f"Expected bimanual TCP pose7 action shape (14,), got {tuple(pose.shape)}")
        return self.right.convert(pose[:7]), self.left.convert(pose[7:])
