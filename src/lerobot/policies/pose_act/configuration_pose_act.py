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

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("pose_act")
@dataclass
class PoseACTConfig(ACTConfig):
    """ACT variant that predicts relative TCP pose chunks from image and pose history."""

    n_obs_steps: int = 2
    img_obs_horizon: int = 2

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

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("pose_act requires at least one image input.")
        if not self.robot_state_feature:
            raise ValueError("pose_act requires `observation.state` as TCP proprioception input.")
        if not self.action_feature:
            raise ValueError("pose_act requires an `action` output feature.")
        if self.robot_state_feature.shape[0] != self.action_feature.shape[0]:
            raise ValueError(
                "pose_act expects observation.state and action to share the same per-step pose dimension. "
                f"Got {self.robot_state_feature.shape[0]} and {self.action_feature.shape[0]}."
            )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))
