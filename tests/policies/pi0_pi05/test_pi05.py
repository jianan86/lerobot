#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Test script to verify PI0.5 (pi05) support in PI0 policy"""

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.factory import make_policy_config  # noqa: E402
from lerobot.policies.pi05 import (  # noqa: E402
    PI05Config,
    PI05Policy,
    make_pi05_pre_post_processors,  # noqa: E402
)
from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel  # noqa: E402
from lerobot.processor import TokenizerProcessorStep  # noqa: E402
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, require_hf_token  # noqa: E402


@pytest.mark.parametrize(
    ("tokenizer_name_or_path", "expected_name"),
    [
        ("/data/jianan/weight/paligemma-3b-pt-224", "/data/jianan/weight/paligemma-3b-pt-224"),
        (None, "google/paligemma-3b-pt-224"),
    ],
)
def test_pi05_processor_uses_configured_tokenizer_name(monkeypatch, tokenizer_name_or_path, expected_name):
    from lerobot.processor.tokenizer_processor import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())

    config = PI05Config()
    if tokenizer_name_or_path is not None:
        config.tokenizer_name_or_path = tokenizer_name_or_path

    preprocessor, _ = make_pi05_pre_post_processors(config=config)
    tokenizer_step = next(step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep))

    assert tokenizer_step.tokenizer_name == expected_name


class _DummyPaliGemma(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Linear(2, 2)
        self.model.vision_tower = nn.Linear(2, 2)
        self.multi_modal_projector = nn.Linear(2, 2)
        self.lm_head = nn.Linear(2, 2, bias=False)


def test_pi05_freeze_language_model_keeps_vision_trainable():
    model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(model)
    model.freeze_vision_encoder = False
    model.freeze_language_model = True
    model.train_expert_only = False
    model.paligemma = _DummyPaliGemma()
    model.gemma_expert = nn.Linear(2, 2)

    model._set_requires_grad()

    assert not any(param.requires_grad for param in model.paligemma.model.language_model.parameters())
    assert not any(param.requires_grad for param in model.paligemma.lm_head.parameters())
    assert any(param.requires_grad for param in model.paligemma.model.vision_tower.parameters())
    assert any(param.requires_grad for param in model.paligemma.multi_modal_projector.parameters())
    assert any(param.requires_grad for param in model.gemma_expert.parameters())

    model.train()

    assert not model.paligemma.model.language_model.training
    assert not model.paligemma.lm_head.training
    assert model.paligemma.model.vision_tower.training
    assert model.paligemma.multi_modal_projector.training
    assert model.gemma_expert.training


@require_cuda
@require_hf_token
def test_policy_instantiation():
    # Create config
    set_seed(42)
    config = PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32")

    # Set up input_features and output_features in the config
    from lerobot.configs.types import FeatureType, PolicyFeature

    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(14,),
        ),
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
    }

    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }

    assert config.tokenizer_max_length == 200, (
        f"Expected tokenizer_max_length=200 for pi05, got {config.tokenizer_max_length}"
    )

    # Create dummy dataset stats
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "min": torch.zeros(14),
            "max": torch.ones(14),
            "q01": torch.zeros(14),
            "q99": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "min": torch.zeros(7),
            "max": torch.ones(7),
            "q01": torch.zeros(7),
            "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    }

    # Instantiate policy
    policy = PI05Policy(config)
    # Test forward pass with dummy data
    batch_size = 1
    preprocessor, postprocessor = make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
    device = config.device
    batch = {
        "observation.state": torch.randn(batch_size, 14, dtype=torch.float32, device=device),
        "action": torch.randn(batch_size, config.chunk_size, 7, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),  # Use rand for [0,1] range
        "task": ["Pick up the object"] * batch_size,
    }
    batch = preprocessor(batch)
    try:
        loss, loss_dict = policy.forward(batch)
        print(f"Forward pass successful. Loss: {loss_dict['loss']:.4f}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise
    try:
        with torch.no_grad():
            action = policy.select_action(batch)
            action = postprocessor(action)
            print(f"Action: {action}")
        print(f"Action prediction successful. Action shape: {action.shape}")
    except Exception as e:
        print(f"Action prediction failed: {e}")
        raise

    # Verify pi05 model components exist
    # Check that time_mlp layers exist (for AdaRMS conditioning)
    assert hasattr(policy.model, "time_mlp_in"), "Missing time_mlp_in layer for pi05"
    assert hasattr(policy.model, "time_mlp_out"), "Missing time_mlp_out layer for pi05"

    # Check that action_time_mlp layers don't exist (pi0 only)
    assert not hasattr(policy.model, "action_time_mlp_in"), "action_time_mlp_in should not exist in pi05 mode"
    assert not hasattr(policy.model, "action_time_mlp_out"), (
        "action_time_mlp_out should not exist in pi05 mode"
    )

    # Check that state_proj doesn't exist in pi05 mode
    assert not hasattr(policy.model, "state_proj"), "state_proj should not exist in pi05 mode"

    # Check AdaRMS configuration in the underlying model
    adarms_config = policy.model.paligemma_with_expert.paligemma.config.text_config.use_adarms
    assert adarms_config == False, f"PaliGemma should not use AdaRMS, got {adarms_config}"  # noqa: E712

    adarms_expert_config = policy.model.paligemma_with_expert.gemma_expert.config.use_adarms
    assert adarms_expert_config == True, (  # noqa: E712
        f"Action expert should use AdaRMS in pi05, got {adarms_expert_config}"
    )


@require_cuda
@require_hf_token
def test_config_creation():
    """Test policy config creation through factory."""
    try:
        config = make_policy_config(
            policy_type="pi0",
            max_action_dim=7,
            max_state_dim=14,
        )
        print("Config created successfully through factory")
        print(f"  Config type: {type(config).__name__}")
        print(f"  PaliGemma variant: {config.paligemma_variant}")
        print(f"  Action expert variant: {config.action_expert_variant}")
    except Exception as e:
        print(f"Config creation failed: {e}")
        raise
