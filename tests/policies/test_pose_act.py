#!/usr/bin/env python

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pose_act.configuration_pose_act import PoseACTConfig
from lerobot.policies.pose_act.modeling_pose_act import PoseACTPolicy
from lerobot.policies.pose_act.processor_pose_act import (
    AbsolutePoseActionProcessorStep,
    RelativePoseActionProcessorStep,
)
from lerobot.policies.pose_act.utils import pose10d_to_pose7d, pose7d_to_pose10d
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.feature_utils import infer_policy_feature_sets


def test_pose_act_forward_and_select_action():
    config = PoseACTConfig(
        device="cpu",
        use_vae=False,
        chunk_size=4,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )
    policy = PoseACTPolicy(config)

    train_batch = {
        "observation.images.front": torch.randn(2, 2, 3, 64, 64),
        OBS_STATE: torch.randn(2, 2, 10),
        ACTION: torch.randn(2, 4, 10),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }

    loss, loss_dict = policy.forward(train_batch)
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict

    policy.reset()
    infer_batch = {
        "observation.images.front": torch.randn(1, 3, 64, 64),
        OBS_STATE: torch.randn(1, 10),
    }
    action = policy.select_action(infer_batch)
    assert action.shape == (1, 10)


def test_pose_act_pose7d_pose10d_roundtrip():
    pose7d = torch.tensor(
        [
            [0.2, -0.1, 0.3, 0.1, -0.2, 0.3, 0.04],
            [0.0, 0.2, -0.4, -0.3, 0.2, -0.1, 0.02],
        ],
        dtype=torch.float32,
    )
    restored = pose10d_to_pose7d(pose7d_to_pose10d(pose7d))
    torch.testing.assert_close(restored, pose7d, rtol=0, atol=1e-6)


def test_pose_act_factory_uses_pose_processors(tmp_path):
    config = PoseACTConfig(
        device="cpu",
        use_vae=False,
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=None)
    absolute_step = next(step for step in postprocessor.steps if isinstance(step, AbsolutePoseActionProcessorStep))
    assert isinstance(absolute_step.relative_step, RelativePoseActionProcessorStep)

    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    _, loaded_postprocessor = make_pre_post_processors(config, pretrained_path=tmp_path)
    loaded_absolute_step = next(
        step for step in loaded_postprocessor.steps if isinstance(step, AbsolutePoseActionProcessorStep)
    )
    assert isinstance(loaded_absolute_step.relative_step, RelativePoseActionProcessorStep)


def test_pose_act_image_feature_key_filters_visual_inputs():
    config = PoseACTConfig(
        device="cpu",
        use_vae=False,
        image_feature_key="observation.images.cam_fish",
    )

    input_features, output_features = infer_policy_feature_sets(
        config,
        {
            "observation.images.cam_high": {"dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channel"]},
            "observation.images.cam_fish": {"dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channel"]},
            OBS_STATE: {"dtype": "float32", "shape": (10,), "names": None},
            ACTION: {"dtype": "float32", "shape": (10,), "names": None},
        },
    )

    assert set(input_features) == {OBS_STATE, "observation.images.cam_fish"}
    assert set(output_features) == {ACTION}


def test_pose_act_image_feature_key_accepts_channel_first_images_without_names():
    config = PoseACTConfig(
        device="cpu",
        use_vae=False,
        image_feature_key="observation.images.fisheye_rgb",
    )

    input_features, output_features = infer_policy_feature_sets(
        config,
        {
            "observation.images.depth_camera_rgb": {"dtype": "image", "shape": (3, 480, 640), "names": None},
            "observation.images.fisheye_rgb": {"dtype": "image", "shape": (3, 480, 640), "names": None},
            OBS_STATE: {"dtype": "float32", "shape": (10,), "names": None},
            ACTION: {"dtype": "float32", "shape": (10,), "names": None},
        },
    )

    assert input_features["observation.images.fisheye_rgb"].shape == (3, 480, 640)
    assert set(input_features) == {OBS_STATE, "observation.images.fisheye_rgb"}
    assert set(output_features) == {ACTION}
