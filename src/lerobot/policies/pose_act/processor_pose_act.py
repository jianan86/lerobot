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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.pose_act import (
    absolute_pose10d,
    ensure_pose10d,
    normalize_pose10d_umi,
    pose10d_umi_scale_offset,
    poses_to_relative_10d,
    unnormalize_pose10d_umi,
)

from .configuration_pose_act import PoseACTConfig
from .rgbd import normalize_pose_act_depth, normalize_pose_act_rgb


def _as_pose10d_feature(feature: PolicyFeature) -> PolicyFeature:
    return PolicyFeature(type=feature.type, shape=(10,))


@ProcessorStepRegistry.register("pose_act_relative_action")
@ProcessorStepRegistry.register("pose_act_relative_pose")
@dataclass
class RelativePoseProcessorStep(ProcessorStep):
    """Convert base-frame pose7d/pose10d observations and actions to relative pose10d."""

    enabled: bool = True
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            state10 = ensure_pose10d(state)
            self._last_state = state10[:, -1] if state10.ndim == 3 else state10

        if not self.enabled:
            return transition

        if state is None:
            return transition

        action = transition.get(TransitionKey.ACTION)
        rel_state, rel_action = poses_to_relative_10d(state, action)

        new_transition = transition.copy()
        new_observation = dict(observation)
        new_observation[OBS_STATE] = rel_state
        new_transition[TransitionKey.OBSERVATION] = new_observation
        if rel_action is not None:
            new_transition[TransitionKey.ACTION] = rel_action
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if OBS_STATE in features.get(PipelineFeatureType.OBSERVATION, {}):
            features[PipelineFeatureType.OBSERVATION][OBS_STATE] = _as_pose10d_feature(
                features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            )
        if ACTION in features.get(PipelineFeatureType.ACTION, {}):
            features[PipelineFeatureType.ACTION][ACTION] = _as_pose10d_feature(
                features[PipelineFeatureType.ACTION][ACTION]
            )
        return features


RelativePoseActionProcessorStep = RelativePoseProcessorStep


@ProcessorStepRegistry.register("pose_act_rgbd_fusion")
@dataclass
class PoseActRgbdFusionProcessorStep(ProcessorStep):
    enabled: bool = True
    config: PoseACTConfig | None = None
    fisheye_rgb_key: str | None = None
    depth_camera_rgb_key: str | None = None
    depth_key: str | None = None
    rgbd_fused_key: str | None = None
    depth_unit_scale: float | None = None
    depth_min_m: float | None = None
    depth_max_m: float | None = None

    def __post_init__(self):
        if self.config is None:
            return
        self.fisheye_rgb_key = self.config.fisheye_rgb_key
        self.depth_camera_rgb_key = self.config.depth_camera_rgb_key
        self.depth_key = self.config.depth_key
        self.rgbd_fused_key = self.config.rgbd_fused_key
        self.depth_unit_scale = self.config.depth_unit_scale
        self.depth_min_m = self.config.depth_min_m
        self.depth_max_m = self.config.depth_max_m

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if (
            not self.enabled
            or self.fisheye_rgb_key is None
            or self.depth_camera_rgb_key is None
            or self.depth_key is None
            or self.rgbd_fused_key is None
            or self.depth_unit_scale is None
            or self.depth_min_m is None
            or self.depth_max_m is None
        ):
            return transition

        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            return transition

        fused = dict(observation)
        if self.rgbd_fused_key not in fused:
            rgb = normalize_pose_act_rgb(fused[self.depth_camera_rgb_key])
            depth = normalize_pose_act_depth(
                fused[self.depth_key],
                unit_scale=self.depth_unit_scale,
                min_m=self.depth_min_m,
                max_m=self.depth_max_m,
            )
            fused[self.fisheye_rgb_key] = normalize_pose_act_rgb(fused[self.fisheye_rgb_key])
            fused[self.depth_camera_rgb_key] = rgb
            fused[self.rgbd_fused_key] = torch.cat([rgb, depth], dim=-3)

        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = fused
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "fisheye_rgb_key": self.fisheye_rgb_key,
            "depth_camera_rgb_key": self.depth_camera_rgb_key,
            "depth_key": self.depth_key,
            "rgbd_fused_key": self.rgbd_fused_key,
            "depth_unit_scale": self.depth_unit_scale,
            "depth_min_m": self.depth_min_m,
            "depth_max_m": self.depth_max_m,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if (
            not self.enabled
            or self.depth_camera_rgb_key is None
            or self.rgbd_fused_key is None
        ):
            return features
        observation_features = features.get(PipelineFeatureType.OBSERVATION)
        if observation_features is not None and self.depth_camera_rgb_key in observation_features:
            rgb_shape = observation_features[self.depth_camera_rgb_key].shape
            observation_features[self.rgbd_fused_key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(4, *rgb_shape[1:]),
            )
        return features


@ProcessorStepRegistry.register("pose_act_umi_normalizer")
@dataclass
class PoseActUmiNormalizerProcessorStep(ProcessorStep):
    enabled: bool = True
    stats: dict[str, dict[str, torch.Tensor]] | None = None
    device: str | torch.device | None = None
    _scale: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _offset: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.stats = self.stats or {}
        for key in (OBS_STATE, ACTION):
            if key in self.stats:
                stat = {name: torch.as_tensor(value, dtype=torch.float32) for name, value in self.stats[key].items()}
                if "min" in stat and "max" in stat:
                    self._scale[key], self._offset[key] = pose10d_umi_scale_offset(stat)
        if self.device is not None:
            self.to(self.device)

    def to(self, device: torch.device | str):
        self.device = device
        self._scale = {key: value.to(device) for key, value in self._scale.items()}
        self._offset = {key: value.to(device) for key, value in self._offset.items()}
        return self

    def _normalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if not self.enabled or key not in self._scale:
            return value
        return normalize_pose10d_umi(value, self._scale[key], self._offset[key])

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        new_transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict) and OBS_STATE in observation:
            new_observation = dict(observation)
            new_observation[OBS_STATE] = self._normalize(OBS_STATE, observation[OBS_STATE])
            new_transition[TransitionKey.OBSERVATION] = new_observation
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            new_transition[TransitionKey.ACTION] = self._normalize(ACTION, action)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "device": str(self.device) if self.device is not None else None}

    def state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for key, scale in self._scale.items():
            state[f"{key}.scale"] = scale.cpu()
            state[f"{key}.offset"] = self._offset[key].cpu()
        return state

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._scale.clear()
        self._offset.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            if stat_name == "scale":
                self._scale[key] = tensor.to(self.device)
            elif stat_name == "offset":
                self._offset[key] = tensor.to(self.device)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("pose_act_umi_unnormalizer")
@dataclass
class PoseActUmiUnnormalizerProcessorStep(PoseActUmiNormalizerProcessorStep):
    def _normalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if not self.enabled or key not in self._scale:
            return value
        return unnormalize_pose10d_umi(value, self._scale[key], self._offset[key])


@ProcessorStepRegistry.register("pose_act_absolute_action")
@dataclass
class AbsolutePoseActionProcessorStep(ProcessorStep):
    """Restore relative pose actions to absolute TCP pose actions."""

    enabled: bool = True
    relative_step: RelativePoseProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if self.relative_step is None or self.relative_step._last_state is None:
            raise RuntimeError("pose_act absolute action restore requires a cached observation.state base pose.")

        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        new_transition = transition.copy()
        base_pose = self.relative_step._last_state
        if isinstance(base_pose, torch.Tensor) and isinstance(action, torch.Tensor) and base_pose.device != action.device:
            base_pose = base_pose.to(device=action.device, dtype=action.dtype)
        new_transition[TransitionKey.ACTION] = absolute_pose10d(action, base_pose)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_pose_act_pre_post_processors(
    config: PoseACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    relative_step = RelativePoseProcessorStep(enabled=True)
    absolute_step = AbsolutePoseActionProcessorStep(enabled=True, relative_step=relative_step)
    if config.use_rgbd_inputs:
        visual_features = {
            config.fisheye_rgb_key: config.input_features[config.fisheye_rgb_key],
            config.rgbd_fused_key: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(4, *config.input_features[config.depth_camera_rgb_key].shape[1:]),
            ),
        }
    else:
        visual_features = {
            key: feature for key, feature in config.input_features.items() if key != OBS_STATE
        }

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        DeviceProcessorStep(device=config.device),
        PoseActRgbdFusionProcessorStep(enabled=config.use_rgbd_inputs, config=config),
        PoseActUmiNormalizerProcessorStep(
            enabled=config.use_pose_normalization,
            stats=dataset_stats,
            device=config.device,
        ),
        NormalizerProcessorStep(
            features=visual_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        PoseActUmiUnnormalizerProcessorStep(
            enabled=config.use_pose_normalization,
            stats=dataset_stats,
        ),
        UnnormalizerProcessorStep(
            features={},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        absolute_step,
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
