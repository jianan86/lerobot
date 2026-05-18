#!/usr/bin/env python

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pose_act.configuration_pose_act import PoseACTConfig
from lerobot.policies.pose_act.processor_pose_act import make_pose_act_pre_post_processors
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.pose_act import pose7d_to_pose10d


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


def test_pose_act_processor_converts_pose7d_to_relative_pose10d():
    config = create_pose_act_config()
    config.use_pose_normalization = False
    preprocessor, _ = make_pose_act_pre_post_processors(config, create_pose_act_stats())

    observation = {
        OBS_STATE: torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
            ]
        ),
        "observation.images.front": torch.randn(2, 3, 32, 32),
    }
    action = torch.tensor([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35])

    processed = preprocessor(transition_to_batch(create_transition(observation, action)))

    assert processed[OBS_STATE].shape == (2, 10)
    assert processed[ACTION].shape == (1, 10)
    torch.testing.assert_close(
        processed[OBS_STATE][-1],
        pose7d_to_pose10d(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3])),
        rtol=0,
        atol=1e-6,
    )


def test_pose_act_processor_fuses_rgbd_inputs():
    config = PoseACTConfig(
        device="cpu",
        use_vae=False,
        use_rgbd_inputs=True,
        use_pose_normalization=False,
        input_features={
            "observation.images.fisheye_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4)),
            "observation.images.depth_camera_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4)),
            "observation.depth.depth_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(1, 4, 4)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )
    preprocessor, _ = make_pose_act_pre_post_processors(config, dataset_stats=None)

    observation = {
        OBS_STATE: torch.zeros(2, 10),
        "observation.images.fisheye_rgb": torch.full((2, 3, 4, 4), 255, dtype=torch.uint8),
        "observation.images.depth_camera_rgb": torch.full((2, 3, 4, 4), 128, dtype=torch.uint8),
        "observation.depth.depth_camera": torch.full((2, 1, 4, 4), 1000, dtype=torch.uint16),
    }

    processed = preprocessor(transition_to_batch(create_transition(observation, None)))

    assert processed["observation.images.fisheye_rgb"].dtype == torch.float32
    torch.testing.assert_close(processed["observation.images.fisheye_rgb"], torch.ones(2, 3, 4, 4))
    assert processed["observation.images.depth_camera_rgbd"].shape == (2, 4, 4, 4)
    torch.testing.assert_close(
        processed["observation.images.depth_camera_rgbd"][:, :3],
        torch.full((2, 3, 4, 4), 128 / 255),
    )
    expected_depth = torch.full(
        (2, 1, 4, 4), (1.0 - config.depth_min_m) / (config.depth_max_m - config.depth_min_m)
    )
    torch.testing.assert_close(processed["observation.images.depth_camera_rgbd"][:, 3:], expected_depth)
