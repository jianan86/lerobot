#!/usr/bin/env python

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pose_act.configuration_pose_act import PoseACTConfig
from lerobot.policies.pose_act.modeling_pose_act import PoseACTPolicy, weighted_pose_act_l1_loss
from lerobot.policies.pose_act.processor_pose_act import (
    AbsolutePoseActionProcessorStep,
    RelativePoseActionProcessorStep,
)
from lerobot.policies.pose_act.utils import pose7d_to_pose10d, pose10d_to_pose7d
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.feature_utils import infer_policy_feature_sets


def test_pose_act_gripper_loss_weight_defaults_to_unweighted_l1():
    target = torch.zeros(2, 4, 10)
    pred = torch.arange(80, dtype=torch.float32).reshape(2, 4, 10) / 100
    action_is_pad = torch.zeros(2, 4, dtype=torch.bool)

    loss, loss_dict = weighted_pose_act_l1_loss(target, pred, action_is_pad, gripper_loss_weight=1.0)
    expected = torch.nn.functional.l1_loss(target, pred, reduction="none").mean()

    torch.testing.assert_close(loss, expected)
    assert loss_dict["gripper_loss_weight"] == 1.0
    assert "gripper_l1_loss" in loss_dict


def test_pose_act_gripper_loss_weight_scales_gripper_and_respects_padding():
    target = torch.zeros(1, 2, 10)
    pred = torch.zeros(1, 2, 10)
    pred[0, 0, 9] = 2.0
    pred[0, 1, 9] = 100.0
    action_is_pad = torch.tensor([[False, True]])

    loss, loss_dict = weighted_pose_act_l1_loss(target, pred, action_is_pad, gripper_loss_weight=5.0)

    torch.testing.assert_close(loss, torch.tensor(0.5))
    assert loss_dict["gripper_l1_loss"] == pytest.approx(2.0)
    assert loss_dict["weighted_gripper_l1_loss"] == pytest.approx(10.0)


def test_pose_act_rejects_non_positive_gripper_loss_weight():
    with pytest.raises(ValueError, match="gripper_loss_weight"):
        PoseACTConfig(device="cpu", gripper_loss_weight=0.0)


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


def test_pose_act_rgbd_forward_and_select_action():
    config = PoseACTConfig(
        device="cpu",
        use_vae=False,
        use_rgbd_inputs=True,
        chunk_size=4,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        input_features={
            "observation.images.fisheye_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.depth_camera_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 64, 64)
            ),
            "observation.depth.depth_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(1, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )
    policy = PoseACTPolicy(config)

    train_batch = {
        "observation.images.fisheye_rgb": torch.randint(0, 255, (2, 2, 3, 64, 64), dtype=torch.uint8),
        "observation.images.depth_camera_rgb": torch.randint(0, 255, (2, 2, 3, 64, 64), dtype=torch.uint8),
        "observation.depth.depth_camera": torch.full((2, 2, 1, 64, 64), 1000, dtype=torch.uint16),
        OBS_STATE: torch.randn(2, 2, 10),
        ACTION: torch.randn(2, 4, 10),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }

    loss, loss_dict = policy.forward(train_batch)
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict

    policy.reset()
    infer_batch = {
        "observation.images.fisheye_rgb": torch.zeros(1, 3, 64, 64, dtype=torch.uint8),
        "observation.images.depth_camera_rgb": torch.zeros(1, 3, 64, 64, dtype=torch.uint8),
        "observation.depth.depth_camera": torch.full((1, 1, 64, 64), 1000, dtype=torch.uint16),
        OBS_STATE: torch.randn(1, 10),
    }
    action = policy.select_action(infer_batch)
    assert action.shape == (1, 10)


def test_pose_act_rgbd_v2_forward_and_select_action():
    config = PoseACTConfig(
        device="cpu",
        use_vae=False,
        use_rgbd_v2_inputs=True,
        chunk_size=4,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        input_features={
            "observation.images.fisheye_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.depth_camera_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 64, 64)
            ),
            "observation.depth.depth_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(1, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )
    policy = PoseACTPolicy(config)

    train_batch = {
        "observation.images.fisheye_rgb": torch.randint(0, 255, (2, 2, 3, 64, 64), dtype=torch.uint8),
        "observation.images.depth_camera_rgb": torch.randint(0, 255, (2, 2, 3, 64, 64), dtype=torch.uint8),
        "observation.depth.depth_camera": torch.full((2, 2, 1, 64, 64), 1000, dtype=torch.uint16),
        OBS_STATE: torch.randn(2, 2, 10),
        ACTION: torch.randn(2, 4, 10),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }

    loss, loss_dict = policy.forward(train_batch)
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict

    policy.reset()
    infer_batch = {
        "observation.images.fisheye_rgb": torch.zeros(1, 3, 64, 64, dtype=torch.uint8),
        "observation.images.depth_camera_rgb": torch.zeros(1, 3, 64, 64, dtype=torch.uint8),
        "observation.depth.depth_camera": torch.full((1, 1, 64, 64), 1000, dtype=torch.uint16),
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


def test_pose_act_accepts_pose7d_features():
    config = PoseACTConfig(
        device="cpu",
        use_vae=False,
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = PoseACTPolicy(config)
    infer_batch = {
        "observation.images.front": torch.randn(1, 3, 32, 32),
        OBS_STATE: torch.randn(1, 7),
    }
    action = policy.select_action(infer_batch)
    assert action.shape == (1, 10)


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


def test_pose_act_rgbd_feature_inference_includes_depth():
    config = PoseACTConfig(device="cpu", use_vae=False, use_rgbd_inputs=True)

    input_features, output_features = infer_policy_feature_sets(
        config,
        {
            "observation.images.fisheye_rgb": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.depth_camera_rgb": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.depth.depth_camera": {
                "dtype": "depth_image",
                "shape": (240, 320, 1),
                "names": ["height", "width", "channel"],
            },
            OBS_STATE: {"dtype": "float32", "shape": (10,), "names": None},
            ACTION: {"dtype": "float32", "shape": (10,), "names": None},
        },
    )

    assert input_features["observation.depth.depth_camera"].shape == (1, 240, 320)
    assert set(input_features) == {
        OBS_STATE,
        "observation.images.fisheye_rgb",
        "observation.images.depth_camera_rgb",
        "observation.depth.depth_camera",
    }
    assert set(output_features) == {ACTION}


def test_pose_act_rgbd_v2_feature_inference_includes_depth_without_depth_mask():
    config = PoseACTConfig(device="cpu", use_vae=False, use_rgbd_v2_inputs=True)

    input_features, output_features = infer_policy_feature_sets(
        config,
        {
            "observation.images.fisheye_rgb": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.depth_camera_rgb": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.depth.depth_camera": {
                "dtype": "depth_image",
                "shape": (240, 320, 1),
                "names": ["height", "width", "channel"],
            },
            OBS_STATE: {"dtype": "float32", "shape": (10,), "names": None},
            ACTION: {"dtype": "float32", "shape": (10,), "names": None},
        },
    )

    assert input_features["observation.depth.depth_camera"].shape == (1, 240, 320)
    assert set(input_features) == {
        OBS_STATE,
        "observation.images.fisheye_rgb",
        "observation.images.depth_camera_rgb",
        "observation.depth.depth_camera",
    }
    assert set(output_features) == {ACTION}


def test_pose_act_rgbd_v2_feature_inference_requires_depth():
    config = PoseACTConfig(device="cpu", use_vae=False, use_rgbd_v2_inputs=True)

    with pytest.raises(ValueError, match="depth_camera"):
        infer_policy_feature_sets(
            config,
            {
                "observation.images.fisheye_rgb": {
                    "dtype": "image",
                    "shape": (240, 320, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.depth_camera_rgb": {
                    "dtype": "image",
                    "shape": (240, 320, 3),
                    "names": ["height", "width", "channel"],
                },
                OBS_STATE: {"dtype": "float32", "shape": (10,), "names": None},
                ACTION: {"dtype": "float32", "shape": (10,), "names": None},
            },
        )


def test_pose_act_rejects_rgbd_v1_and_v2_together():
    with pytest.raises(ValueError, match="mutually exclusive"):
        PoseACTConfig(device="cpu", use_vae=False, use_rgbd_inputs=True, use_rgbd_v2_inputs=True)


def test_pose_act_rgbd_v2_does_not_require_depth_mask_feature():
    config = PoseACTConfig(
        device="cpu",
        use_vae=False,
        use_rgbd_v2_inputs=True,
        input_features={
            "observation.images.fisheye_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.depth_camera_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 64, 64)
            ),
            "observation.depth.depth_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(1, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )

    PoseACTPolicy(config)


def test_pose_act_rgbd_v2_requires_single_channel_depth():
    base_features = {
        "observation.images.fisheye_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        "observation.images.depth_camera_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        "observation.depth.depth_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(1, 64, 64)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
    }
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))}

    bad_depth = dict(base_features)
    bad_depth["observation.depth.depth_camera"] = PolicyFeature(type=FeatureType.VISUAL, shape=(2, 64, 64))
    with pytest.raises(ValueError, match="depth to have 1 channel"):
        PoseACTPolicy(
            PoseACTConfig(
                device="cpu",
                use_vae=False,
                use_rgbd_v2_inputs=True,
                input_features=bad_depth,
                output_features=output_features,
            )
        )
