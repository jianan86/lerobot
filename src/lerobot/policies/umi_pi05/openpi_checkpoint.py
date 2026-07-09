#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_umi_pi05 import UmiPI05Config
from .processor_umi_pi05 import make_umi_pi05_pre_post_processors

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


def has_openpi_umi_pi05_files(path: str | Path) -> bool:
    checkpoint = Path(path)
    return (
        checkpoint.is_dir()
        and (checkpoint / "metadata.pt").exists()
        and (checkpoint / "model.safetensors").exists()
        and any(checkpoint.glob("assets/*/norm_stats.json"))
    )


def load_openpi_umi_pi05_config(path: str | Path, *, device: str | None = None) -> UmiPI05Config:
    # The UMI runtime dimensions are fixed by the OpenPI bimanual Pika checkpoint contract.
    # Do not unpickle metadata.pt here; OpenPI metadata can require Flax or JAX-only classes.

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
    if "norm_stats" in stats:
        stats = stats["norm_stats"]

    return {
        OBS_STATE: _convert_stats_entry(stats["state"]),
        ACTION: _convert_stats_entry(stats["actions"]),
    }


def materialize_openpi_umi_pi05_checkpoint(
    path: str | Path,
    *,
    device: str | None = "cpu",
    overwrite: bool = False,
) -> list[Path]:
    checkpoint = Path(path)
    if not has_openpi_umi_pi05_files(checkpoint):
        raise FileNotFoundError(f"{checkpoint} does not look like an OpenPI UMI pi05 checkpoint.")

    output_files = [
        checkpoint / "config.json",
        checkpoint / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        checkpoint / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    ]
    existing = [file for file in output_files if file.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "LeRobot checkpoint files already exist. Pass overwrite=True to replace: "
            + ", ".join(str(file) for file in existing)
        )

    config = load_openpi_umi_pi05_config(checkpoint, device=device)
    stats = load_openpi_umi_pi05_stats(checkpoint)


    class _TokenizerStub:
        padding_side = "right"

        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("Tokenizer is not used while materializing processor configs.")

    config.save_pretrained(checkpoint)
    with patch("lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained", return_value=_TokenizerStub()):
        preprocessor, postprocessor = make_umi_pi05_pre_post_processors(config, dataset_stats=stats)
        preprocessor.save_pretrained(
            checkpoint,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        )
        postprocessor.save_pretrained(
            checkpoint,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        )

    return sorted(
        [
            *output_files,
            *checkpoint.glob(f"{POLICY_PREPROCESSOR_DEFAULT_NAME}_step_*.safetensors"),
            *checkpoint.glob(f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}_step_*.safetensors"),
        ]
    )


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
