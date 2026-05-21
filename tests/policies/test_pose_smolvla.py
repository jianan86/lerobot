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

import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.pose_act.processor_pose_act import (
    AbsolutePoseActionProcessorStep,
    RelativePoseActionProcessorStep,
)
from lerobot.policies.pose_smolvla.configuration_pose_smolvla import PoseSmolVLAConfig
from lerobot.policies.pose_smolvla.modeling_pose_smolvla import PoseSmolVLAPolicy
from lerobot.policies.pose_smolvla.processor_pose_smolvla import make_pose_smolvla_pre_post_processors
from lerobot.processor import EnvTransition, ProcessorStep, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.feature_utils import infer_policy_feature_sets


class MockTokenizerProcessorStep(ProcessorStep):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return transition

    def transform_features(self, features):
        return features


class TokenizerLikeProcessorStep(ProcessorStep):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = dict(transition[TransitionKey.OBSERVATION])
        observation[OBS_LANGUAGE_TOKENS] = torch.ones(1, 4, dtype=torch.long)
        observation[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, 4, dtype=torch.bool)
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(self, features):
        return features


class FakeVLAFlowMatching(nn.Module):
    def __init__(self, config, rtc_processor=None):
        super().__init__()
        self.config = config
        self.seen_images = None
        self.seen_state = None

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        self.seen_images = images
        self.seen_state = state
        return torch.zeros(state.shape[0], self.config.chunk_size, self.config.max_action_dim)

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        self.seen_images = images
        self.seen_state = state
        return torch.zeros_like(actions)


def _config() -> PoseSmolVLAConfig:
    return PoseSmolVLAConfig(
        device="cpu",
        input_features={
            OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
        resize_imgs_with_padding=None,
    )


def test_pose_smolvla_factory_and_processors(monkeypatch):
    monkeypatch.setattr(
        "lerobot.policies.pose_smolvla.processor_pose_smolvla.TokenizerProcessorStep",
        MockTokenizerProcessorStep,
    )

    config = make_policy_config("pose_smolvla", device="cpu")
    assert isinstance(config, PoseSmolVLAConfig)
    assert get_policy_class("pose_smolvla") is PoseSmolVLAPolicy

    config = _config()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=None)
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"
    absolute_step = next(step for step in postprocessor.steps if isinstance(step, AbsolutePoseActionProcessorStep))
    assert isinstance(absolute_step.relative_step, RelativePoseActionProcessorStep)

    direct_preprocessor, direct_postprocessor = make_pose_smolvla_pre_post_processors(config, dataset_stats=None)
    assert direct_preprocessor.name == preprocessor.name
    assert direct_postprocessor.name == postprocessor.name


def test_pose_smolvla_rejects_non_rgb_visual_inputs():
    config = PoseSmolVLAConfig(
        device="cpu",
        input_features={
            "observation.images.depth": PolicyFeature(type=FeatureType.VISUAL, shape=(1, 16, 16)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )

    try:
        config.validate_features()
    except ValueError as exc:
        assert "3-channel RGB" in str(exc) or "depth/RGBD" in str(exc)
    else:
        raise AssertionError("Expected pose_smolvla to reject non-RGB visual input")


def test_pose_smolvla_image_feature_key_filters_to_fisheye_rgb():
    config = PoseSmolVLAConfig(
        device="cpu",
        image_feature_key="observation.images.fisheye_rgb",
    )

    input_features, output_features = infer_policy_feature_sets(
        config,
        {
            "observation.images.fisheye_rgb": {
                "dtype": "video",
                "shape": (3, 480, 640),
                "names": None,
            },
            "observation.images.depth_camera_rgb": {
                "dtype": "video",
                "shape": (3, 480, 640),
                "names": None,
            },
            "observation.depth.depth_camera": {
                "dtype": "depth_video",
                "shape": (1, 480, 640),
                "names": None,
            },
            OBS_STATE: {"dtype": "float32", "shape": (7,), "names": None},
            ACTION: {"dtype": "float32", "shape": (7,), "names": None},
        },
    )

    assert set(input_features) == {OBS_STATE, "observation.images.fisheye_rgb"}
    assert set(output_features) == {ACTION}


def test_pose_smolvla_rejects_depth_image_feature_key():
    config = PoseSmolVLAConfig(
        device="cpu",
        image_feature_key="observation.depth.depth_camera",
    )

    try:
        infer_policy_feature_sets(
            config,
            {
                "observation.images.fisheye_rgb": {
                    "dtype": "video",
                    "shape": (3, 480, 640),
                    "names": None,
                },
                "observation.depth.depth_camera": {
                    "dtype": "depth_video",
                    "shape": (1, 480, 640),
                    "names": None,
                },
                OBS_STATE: {"dtype": "float32", "shape": (7,), "names": None},
                ACTION: {"dtype": "float32", "shape": (7,), "names": None},
            },
        )
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("Expected pose_smolvla to reject a depth image_feature_key")


def test_pose_smolvla_processor_tokenizes_task(monkeypatch):
    monkeypatch.setattr(
        "lerobot.policies.pose_smolvla.processor_pose_smolvla.TokenizerProcessorStep",
        TokenizerLikeProcessorStep,
    )

    config = _config()
    preprocessor, _ = make_pose_smolvla_pre_post_processors(config, dataset_stats=None)
    observation = {
        OBS_STATE: torch.tensor(
            [
                [0.40, -0.10, 0.45, 0.00, 0.10, 0.00, 0.02],
                [0.405, -0.10, 0.45, 0.00, 0.10, 0.01, 0.02],
            ],
            dtype=torch.float32,
        ),
        OBS_IMAGE: torch.rand(2, 3, 16, 16),
        "task": "umi episode",
    }

    batch = preprocessor(observation)

    assert OBS_LANGUAGE_TOKENS in batch
    assert OBS_LANGUAGE_ATTENTION_MASK in batch
    assert batch[OBS_STATE].shape[-1] == 10


def test_pose_smolvla_uses_full_pose_and_image_history(monkeypatch):
    monkeypatch.setattr("lerobot.policies.smolvla.modeling_smolvla.require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr("lerobot.policies.smolvla.modeling_smolvla.VLAFlowMatching", FakeVLAFlowMatching)

    config = _config()
    policy = PoseSmolVLAPolicy(config)
    batch = {
        OBS_STATE: torch.arange(20, dtype=torch.float32).reshape(1, 2, 10),
        OBS_IMAGE: torch.rand(1, 2, 3, 16, 16),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.bool),
    }

    chunk = policy.predict_action_chunk(batch)

    assert chunk.shape == (1, config.chunk_size, 10)
    assert len(policy.model.seen_images) == 2
    assert policy.model.seen_state.shape == (1, config.max_state_dim)
    torch.testing.assert_close(policy.model.seen_state[:, :20], batch[OBS_STATE].flatten(start_dim=1))
    torch.testing.assert_close(policy.model.seen_state[:, 20:], torch.zeros(1, config.max_state_dim - 20))


def test_pose_smolvla_nonstrict_checkpoint_load_skips_shape_mismatches(tmp_path):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.same = nn.Parameter(torch.zeros(2))
            self.mismatch = nn.Parameter(torch.zeros(2))

    checkpoint_path = tmp_path / "model.safetensors"
    save_file(
        {
            "same": torch.ones(2),
            "mismatch": torch.ones(3),
            "unexpected": torch.ones(1),
        },
        checkpoint_path,
    )

    model = TinyModel()
    PoseSmolVLAPolicy._load_as_safetensor(model, str(checkpoint_path), "cpu", strict=False)

    torch.testing.assert_close(model.same, torch.ones(2))
    torch.testing.assert_close(model.mismatch, torch.zeros(2))
