#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .configuration_umi_pi05 import UmiPI05Config

OPENPI_UMI_IMAGE_KEYS = (
    f"{OBS_IMAGES}.cam_high",
    f"{OBS_IMAGES}.cam_left_wrist",
    f"{OBS_IMAGES}.cam_right_wrist",
)


def is_openpi_umi_pi05_checkpoint(path: str | Path) -> bool:
    checkpoint = Path(path)
    return (
        checkpoint.is_dir()
        and not (checkpoint / "config.json").exists()
        and (checkpoint / "metadata.pt").exists()
        and (checkpoint / "model.safetensors").exists()
        and any(checkpoint.glob("assets/*/norm_stats.json"))
    )


def load_openpi_umi_pi05_config(path: str | Path, *, device: str | None = None) -> UmiPI05Config:
    # Load metadata to catch corrupt/non-checkpoint directories early. The UMI runtime
    # dimensions are fixed by the OpenPI bimanual Pika checkpoint contract.
    torch.load(Path(path) / "metadata.pt", map_location="cpu", weights_only=False)

    config = UmiPI05Config(
        device=device,
        chunk_size=10,
        n_action_steps=10,
        max_state_dim=32,
        max_action_dim=32,
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(20,)),
        **{
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
            for key in OPENPI_UMI_IMAGE_KEYS
        },
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10, 20))}
    return config


def load_openpi_umi_pi05_stats(path: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    stats_path = _find_norm_stats_path(Path(path))
    with open(stats_path) as f:
        stats = json.load(f)

    return {
        OBS_STATE: _convert_stats_entry(stats["state"]),
        ACTION: _convert_stats_entry(stats["actions"]),
    }


def _find_norm_stats_path(path: Path) -> Path:
    matches = sorted(path.glob("assets/*/norm_stats.json"))
    if not matches:
        raise FileNotFoundError(f"No OpenPI norm_stats.json found under {path / 'assets'}")
    return matches[0]


def _convert_stats_entry(entry: dict[str, Any]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    aliases = {
        "q01": ("q01", "p01", "quantile_01"),
        "q99": ("q99", "p99", "quantile_99"),
        "mean": ("mean",),
        "std": ("std",),
        "min": ("min",),
        "max": ("max",),
    }
    for target_key, source_keys in aliases.items():
        for source_key in source_keys:
            if source_key in entry:
                converted[target_key] = torch.as_tensor(entry[source_key], dtype=torch.float32)
                break

    if "q01" not in converted or "q99" not in converted:
        raise ValueError("OpenPI UMI pi05 stats must include q01/q99 (or p01/p99) quantiles.")
    return converted
