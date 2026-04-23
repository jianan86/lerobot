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

from lerobot.configs import PipelineFeatureType, PolicyFeature
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
from lerobot.utils.constants import ACTION, OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.utils.pose_act import (
    absolute_pose10d,
    ensure_pose10d,
    normalize_pose10d_umi,
    pose10d_umi_scale_offset,
    poses_to_relative_10d,
    unnormalize_pose10d_umi,
)

from .configuration_pose_act import PoseACTConfig


def _as_pose10d_feature(feature: PolicyFeature) -> PolicyFeature:
    return PolicyFeature(type=feature.type, shape=(10,))


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
        new_transition[TransitionKey.ACTION] = absolute_pose10d(action, self.relative_step._last_state)
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
    visual_features = {
        key: feature for key, feature in config.input_features.items() if key != OBS_STATE
    }

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        DeviceProcessorStep(device=config.device),
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
