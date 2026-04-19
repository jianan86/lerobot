#!/usr/bin/env python

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pose_act.configuration_pose_act import PoseACTConfig
from lerobot.policies.pose_act.processor_pose_act import make_pose_act_pre_post_processors
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_STATE


def create_pose_act_config():
    config = PoseACTConfig(
        use_vae=False,
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )
    config.device = "cpu"
    return config


def create_pose_act_stats():
    return {
        OBS_STATE: {"mean": torch.zeros(10), "std": torch.ones(10)},
        ACTION: {"mean": torch.zeros(10), "std": torch.ones(10)},
        "observation.images.front": {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
    }


def test_pose_act_processor_relative_absolute_roundtrip():
    config = create_pose_act_config()
    preprocessor, postprocessor = make_pose_act_pre_post_processors(config, create_pose_act_stats())

    observation = {
        OBS_STATE: torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3],
            ]
        ),
        "observation.images.front": torch.randn(2, 3, 32, 32),
    }
    action = torch.tensor(
        [
            [1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.35],
            [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.40],
        ]
    )

    batch = transition_to_batch(create_transition(observation, action))
    processed = preprocessor(batch)
    restored = postprocessor(processed[ACTION])

    torch.testing.assert_close(restored.squeeze(0), action, rtol=0, atol=1e-5)
