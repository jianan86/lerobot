#!/usr/bin/env python
"""Create a local random-weight pose_act checkpoint for async inference testing."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pose_act import PoseACTConfig, PoseACTPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/async_piper/pose_act_random"))
    parser.add_argument("--image-key", default="observation.images.fisheye_rgb")
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--dim-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_features = {
        args.image_key: PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, args.image_height, args.image_width),
        ),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
    }
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))}

    cfg = PoseACTConfig(
        device="cpu",
        use_vae=False,
        pretrained_backbone_weights=None,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        img_obs_horizon=args.n_obs_steps,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        n_encoder_layers=1,
        n_decoder_layers=1,
        input_features=input_features,
        output_features=output_features,
    )
    policy = PoseACTPolicy(cfg)
    stats = {
        OBS_STATE: {"mean": torch.zeros(10), "std": torch.ones(10)},
        ACTION: {"mean": torch.zeros(10), "std": torch.ones(10)},
        args.image_key: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
    }
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=stats)

    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
