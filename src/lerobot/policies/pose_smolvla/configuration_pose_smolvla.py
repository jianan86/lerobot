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

from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("pose_smolvla")
@dataclass
class PoseSmolVLAConfig(SmolVLAConfig):
    """SmolVLA variant for RGB plus TCP pose history."""

    n_obs_steps: int = 2
    image_feature_key: str | None = None
    init_smolvla_from: str | None = None
    use_pose_normalization: bool = True
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.n_obs_steps < 1:
            raise ValueError(f"`n_obs_steps` must be >= 1. Got {self.n_obs_steps}.")
        if self.n_obs_steps * 10 > self.max_state_dim:
            raise ValueError(
                "`max_state_dim` must fit flattened pose10d history. "
                f"Got n_obs_steps={self.n_obs_steps}, max_state_dim={self.max_state_dim}."
            )

    def validate_features(self) -> None:
        super().validate_features()
        if not self.image_features:
            raise ValueError("pose_smolvla requires at least one RGB image input.")
        if self.image_feature_key is not None and self.image_feature_key not in self.image_features:
            raise ValueError(
                f"pose_smolvla image_feature_key={self.image_feature_key!r} is not present in input_features. "
                f"Available image features: {sorted(self.image_features)}."
            )
        for key, feature in self.image_features.items():
            if len(feature.shape) != 3 or feature.shape[0] != 3:
                raise ValueError(
                    "pose_smolvla supports only 3-channel RGB visual inputs. "
                    f"Feature {key!r} has shape {feature.shape}."
                )
            lower_key = key.lower()
            if "depth" in lower_key or "rgbd" in lower_key:
                raise ValueError(
                    "pose_smolvla v1 supports RGB images only; depth/RGBD visual inputs are not supported. "
                    f"Got feature {key!r}."
                )
        if not self.robot_state_feature:
            raise ValueError("pose_smolvla requires `observation.state` as TCP proprioception input.")
        if not self.action_feature:
            raise ValueError("pose_smolvla requires an `action` output feature.")
        if self.robot_state_feature.shape[0] not in (7, 10):
            raise ValueError(
                "pose_smolvla expects observation.state to be pose7d or pose10d. "
                f"Got {self.robot_state_feature.shape[0]}."
            )
        if self.action_feature.shape[0] not in (7, 10):
            raise ValueError(
                "pose_smolvla expects action to be pose7d or pose10d. "
                f"Got {self.action_feature.shape[0]}."
            )
        if self.robot_state_feature.shape[0] != self.action_feature.shape[0]:
            raise ValueError(
                "pose_smolvla expects observation.state and action to share the same per-step pose dimension. "
                f"Got {self.robot_state_feature.shape[0]} and {self.action_feature.shape[0]}."
            )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def model_image_feature_keys(self) -> list[str]:
        return [key for key, ft in self.image_features.items() if ft.type is FeatureType.VISUAL]
