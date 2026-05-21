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

from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from ..pose_act.processor_pose_act import (
    AbsolutePoseActionProcessorStep,
    PoseActUmiNormalizerProcessorStep,
    PoseActUmiUnnormalizerProcessorStep,
    RelativePoseProcessorStep,
)
from .configuration_pose_smolvla import PoseSmolVLAConfig


def make_pose_smolvla_pre_post_processors(
    config: PoseSmolVLAConfig,
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
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
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
