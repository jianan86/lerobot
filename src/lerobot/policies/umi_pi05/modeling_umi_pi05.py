#!/usr/bin/env python

import builtins
from pathlib import Path
from typing import TypeVar, Unpack

import torch
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from ..pi05.modeling_pi05 import ActionSelectKwargs, PI05Policy
from .configuration_umi_pi05 import UmiPI05Config
from .openpi_checkpoint import is_openpi_umi_pi05_checkpoint, load_openpi_umi_pi05_config

T = TypeVar("T", bound="UmiPI05Policy")


class UmiPI05Policy(PI05Policy):
    """PI0.5 policy variant for UMI datasets with pre-chunked relative pose actions."""

    config_class = UmiPI05Config
    name = "umi_pi05"

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        if config is None and is_openpi_umi_pi05_checkpoint(pretrained_name_or_path):
            config = load_openpi_umi_pi05_config(
                pretrained_name_or_path,
                device=kwargs.get("device"),
            )

        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )

    def _original_action_dim(self) -> int:
        action_shape = self.config.output_features[ACTION].shape
        if len(action_shape) == 1:
            return action_shape[0]
        if len(action_shape) == 2:
            if action_shape[0] != self.config.chunk_size:
                raise ValueError(
                    f"umi_pi05 expects action chunks of length {self.config.chunk_size}, got {action_shape}."
                )
            return action_shape[-1]
        raise ValueError(f"umi_pi05 expects action shape (dim,) or (chunk, dim), got {action_shape}.")

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        self.eval()

        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)
        return actions[:, :, : self._original_action_dim()]

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)
        losses = self.model.forward(images, img_masks, tokens, masks, actions)
        losses = losses[:, :, : self._original_action_dim()]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict

        loss = losses.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict
