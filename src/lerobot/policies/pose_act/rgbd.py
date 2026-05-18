#!/usr/bin/env python

from __future__ import annotations

import torch
from torch import Tensor

from .configuration_pose_act import PoseACTConfig


def normalize_pose_act_depth(
    depth: Tensor,
    *,
    unit_scale: float,
    min_m: float,
    max_m: float,
) -> Tensor:
    depth_m = depth.to(torch.float32) * unit_scale
    depth_m = depth_m.clamp(min=min_m, max=max_m)
    return (depth_m - min_m) / (max_m - min_m)


def normalize_pose_act_rgb(rgb: Tensor) -> Tensor:
    if rgb.dtype == torch.uint8:
        return rgb.to(torch.float32) / 255.0
    return rgb.to(torch.float32)


def fuse_pose_act_rgbd_observation(
    observation: dict[str, Tensor], config: PoseACTConfig
) -> dict[str, Tensor]:
    if not config.use_rgbd_inputs or config.rgbd_fused_key in observation:
        return observation

    fused = dict(observation)
    rgb = normalize_pose_act_rgb(fused[config.depth_camera_rgb_key])
    depth = normalize_pose_act_depth(
        fused[config.depth_key],
        unit_scale=config.depth_unit_scale,
        min_m=config.depth_min_m,
        max_m=config.depth_max_m,
    )
    fused[config.fisheye_rgb_key] = normalize_pose_act_rgb(fused[config.fisheye_rgb_key])
    fused[config.depth_camera_rgb_key] = rgb
    fused[config.rgbd_fused_key] = torch.cat([rgb, depth], dim=-3)
    return fused
