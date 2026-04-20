#!/usr/bin/env python

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pose_act.configuration_pose_act import PoseACTConfig
from lerobot.policies.pose_act.modeling_pose_act import PoseACTPolicy
from lerobot.utils.feature_utils import infer_policy_feature_sets
from lerobot.utils.constants import ACTION, OBS_STATE


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
