#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from pathlib import Path

import safetensors
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from packaging import version
from safetensors.torch import load_file

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from ..smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector, resize_with_pad
from .configuration_pose_smolvla import PoseSmolVLAConfig

logger = logging.getLogger(__name__)


class PoseSmolVLAPolicy(SmolVLAPolicy):
    """SmolVLA policy with PoseACT-style pose history semantics."""

    config_class = PoseSmolVLAConfig
    name = "pose_smolvla"

    def __init__(self, config: PoseSmolVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        if config.init_smolvla_from:
            self._init_from_smolvla(config.init_smolvla_from)

    def _init_from_smolvla(self, pretrained_name_or_path: str | Path) -> None:
        model_id = str(pretrained_name_or_path)
        model_file = (
            os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            if os.path.isdir(model_id)
            else hf_hub_download(repo_id=model_id, filename=SAFETENSORS_SINGLE_FILE)
        )
        checkpoint = load_file(model_file, device=str(self.config.device))
        own_state = self.state_dict()
        compatible = {}
        skipped = []
        unexpected = []
        for key, value in checkpoint.items():
            if key not in own_state:
                unexpected.append(key)
                continue
            if own_state[key].shape != value.shape:
                skipped.append(key)
                continue
            compatible[key] = value

        missing, unexpected_after_load = self.load_state_dict(compatible, strict=False)
        logger.info(
            "Initialized pose_smolvla from %s: loaded=%d skipped_shape=%d missing=%d unexpected=%d",
            pretrained_name_or_path,
            len(compatible),
            len(skipped),
            len(missing),
            len(unexpected) + len(unexpected_after_load),
        )
        if skipped:
            logger.info("Skipped shape-mismatched SmolVLA keys for pose_smolvla init: %s", sorted(skipped))
        if unexpected or unexpected_after_load:
            logger.info(
                "Unexpected SmolVLA keys for pose_smolvla init: %s",
                sorted(set(unexpected) | set(unexpected_after_load)),
            )

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        if strict:
            return super()._load_as_safetensor(model, model_file, map_location, strict)

        kwargs = {}
        if version.parse(safetensors.__version__) >= version.parse("0.4.3"):
            kwargs["device"] = map_location
        checkpoint = load_file(model_file, **kwargs)
        own_state = model.state_dict()
        compatible = {}
        skipped = []
        unexpected = []
        for key, value in checkpoint.items():
            if key not in own_state:
                unexpected.append(key)
                continue
            if own_state[key].shape != value.shape:
                skipped.append(key)
                continue
            compatible[key] = value
        missing, unexpected_after_load = model.load_state_dict(compatible, strict=False)
        logger.info(
            "Loaded pose_smolvla weights: loaded=%d skipped_shape=%d missing=%d unexpected=%d",
            len(compatible),
            len(skipped),
            len(missing),
            len(unexpected) + len(unexpected_after_load),
        )
        if skipped:
            logger.info("Skipped shape-mismatched pose_smolvla keys: %s", sorted(skipped))
        return model

    def _get_action_chunk(self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None, **kwargs):
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )
        return actions[:, :, :10]

    def forward(
        self, batch: dict[str, torch.Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, torch.Tensor]:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        losses = losses[:, :, : batch[ACTION].shape[-1]]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict

        loss = losses.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def prepare_images(self, batch):
        """Send every camera/history frame as a distinct SmolVLA prefix image."""
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        last_img = None
        last_mask = None
        for key in present_img_keys:
            img_history = batch[key]
            if img_history.ndim == 4:
                img_history = img_history[:, None, :, :, :]
            if img_history.ndim != 5:
                raise ValueError(f"Expected image batch {key!r} to have shape (B,T,C,H,W) or (B,C,H,W), got {img_history.shape}")

            for t in range(img_history.shape[1]):
                img = img_history[:, t, :, :, :]
                if self.config.resize_imgs_with_padding is not None:
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
                img = img * 2.0 - 1.0

                bsize = img.shape[0]
                device = img.device
                if f"{key}_padding_mask" in batch:
                    mask = batch[f"{key}_padding_mask"].bool()
                else:
                    mask = torch.ones(bsize, dtype=torch.bool, device=device)
                images.append(img)
                img_masks.append(mask)
                last_img = img
                last_mask = mask

        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            if last_img is None or last_mask is None:
                break
            img = torch.ones_like(last_img) * -1
            mask = torch.zeros_like(last_mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_state(self, batch):
        """Flatten relative pose10d history and pad to SmolVLA state width."""
        state = batch[OBS_STATE]
        if state.ndim == 2:
            state = state[:, None, :]
        if state.ndim != 3:
            raise ValueError(f"Expected pose_smolvla state shape (B,T,D) or (B,D), got {state.shape}")
        if state.shape[-1] != 10:
            raise ValueError(f"Expected pose_smolvla preprocessed state dim 10, got {state.shape[-1]}")
        state = state.flatten(start_dim=1)
        return pad_vector(state, self.config.max_state_dim)
