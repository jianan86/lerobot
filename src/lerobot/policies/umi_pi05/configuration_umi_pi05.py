#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("umi_pi05")
@dataclass
class UmiPI05Config(PI05Config):
    """PI0.5 variant for UMI pose chunks already expressed in relative pose space."""

    chunk_size: int = 10
    n_action_steps: int = 10
    max_state_dim: int = 32
    max_action_dim: int = 32
    tokenizer_name_or_path: str = "google/paligemma-3b-pt-224"
    use_relative_actions: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.use_relative_actions:
            raise ValueError(
                "umi_pi05 expects UMI pose actions that are already relative. "
                "Do not enable use_relative_actions."
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )

        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.chunk_size, self.max_action_dim),
            )

        action_shape = self.output_features[ACTION].shape
        if len(action_shape) == 2 and action_shape[0] != self.chunk_size:
            raise ValueError(
                f"umi_pi05 expects pre-chunked actions with first dimension equal to chunk_size "
                f"({self.chunk_size}), got action shape {action_shape}."
            )

    @property
    def action_delta_indices(self) -> None:
        return None
