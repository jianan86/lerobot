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

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig

from ..act.configuration_act import ACTConfig

POSE_ACT_FISHEYE_RGB_KEY = "observation.images.fisheye_rgb"
POSE_ACT_DEPTH_CAMERA_RGB_KEY = "observation.images.depth_camera_rgb"
POSE_ACT_DEPTH_KEY = "observation.depth.depth_camera"
POSE_ACT_DEPTH_MASK_KEY = "observation.depth_mask.depth_camera"
POSE_ACT_RGBD_FUSED_KEY = "observation.images.depth_camera_rgbd"
POSE_ACT_DEPTH_MASK_FUSED_KEY = "observation.depth.depth_camera_with_mask"


@PreTrainedConfig.register_subclass("pose_act")
@dataclass
class PoseACTConfig(ACTConfig):
    """ACT variant that predicts relative TCP pose chunks from image and pose history."""

    n_obs_steps: int = 2
    img_obs_horizon: int = 2
    image_feature_key: str | None = None
    use_pose_normalization: bool = True
    use_rgbd_inputs: bool = False
    use_rgbd_v2_inputs: bool = False
    fisheye_rgb_key: str = POSE_ACT_FISHEYE_RGB_KEY
    depth_camera_rgb_key: str = POSE_ACT_DEPTH_CAMERA_RGB_KEY
    depth_key: str = POSE_ACT_DEPTH_KEY
    depth_mask_key: str = POSE_ACT_DEPTH_MASK_KEY
    rgbd_fused_key: str = POSE_ACT_RGBD_FUSED_KEY
    depth_mask_fused_key: str = POSE_ACT_DEPTH_MASK_FUSED_KEY
    depth_unit_scale: float = 0.001
    depth_min_m: float = 0.1
    depth_max_m: float = 5.0
    gripper_loss_weight: float = 1.0
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        # Keep a single source of truth for image/state history in v1.
        self.n_obs_steps = self.img_obs_horizon
        PreTrainedConfig.__post_init__(self)

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. "
                "This is because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps < 1:
            raise ValueError(f"`n_obs_steps` must be >= 1. Got {self.n_obs_steps}.")
        if self.depth_min_m >= self.depth_max_m:
            raise ValueError(
                "`depth_min_m` must be smaller than `depth_max_m`. "
                f"Got {self.depth_min_m} and {self.depth_max_m}."
            )
        if self.use_rgbd_inputs and self.use_rgbd_v2_inputs:
            raise ValueError("pose_act RGBD v1 and RGBD v2 input modes are mutually exclusive.")
        if self.gripper_loss_weight <= 0:
            raise ValueError(
                "`gripper_loss_weight` must be positive. "
                f"Got {self.gripper_loss_weight}."
            )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("pose_act requires at least one image input.")
        if self.use_rgbd_inputs or self.use_rgbd_v2_inputs:
            required = [self.fisheye_rgb_key, self.depth_camera_rgb_key, self.depth_key]
            mode_name = "RGBD v2" if self.use_rgbd_v2_inputs else "RGBD"
            missing = [
                key
                for key in required
                if not self.input_features or key not in self.input_features
            ]
            if missing:
                raise ValueError(f"pose_act {mode_name} mode requires input feature(s): {missing}.")
            if self.input_features[self.fisheye_rgb_key].shape[0] != 3:
                raise ValueError(f"pose_act {mode_name} mode expects fisheye RGB to have 3 channels.")
            if self.input_features[self.depth_camera_rgb_key].shape[0] != 3:
                raise ValueError(f"pose_act {mode_name} mode expects depth camera RGB to have 3 channels.")
            if self.input_features[self.depth_key].shape[0] != 1:
                raise ValueError(f"pose_act {mode_name} mode expects depth to have 1 channel.")
        if not self.robot_state_feature:
            raise ValueError("pose_act requires `observation.state` as TCP proprioception input.")
        if not self.action_feature:
            raise ValueError("pose_act requires an `action` output feature.")
        if self.robot_state_feature.shape[0] not in (7, 10):
            raise ValueError(
                "pose_act expects observation.state to be pose7d or pose10d. "
                f"Got {self.robot_state_feature.shape[0]}."
            )
        if self.action_feature.shape[0] not in (7, 10):
            raise ValueError(
                "pose_act expects action to be pose7d or pose10d. "
                f"Got {self.action_feature.shape[0]}."
            )
        if self.robot_state_feature.shape[0] != self.action_feature.shape[0]:
            raise ValueError(
                "pose_act expects observation.state and action to share the same per-step pose dimension. "
                f"Got {self.robot_state_feature.shape[0]} and {self.action_feature.shape[0]}."
            )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def model_image_feature_keys(self) -> list[str]:
        if self.use_rgbd_v2_inputs:
            return [self.fisheye_rgb_key, self.depth_camera_rgb_key, self.depth_mask_fused_key]
        if self.use_rgbd_inputs:
            return [self.fisheye_rgb_key, self.rgbd_fused_key]
        return list(self.image_features)
