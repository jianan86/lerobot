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

from __future__ import annotations

from collections import deque
from copy import deepcopy

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.pose_act import ensure_pose10d, make_relative_state_history

from ..act.modeling_act import ACT, ACTTemporalEnsembler
from ..pretrained import PreTrainedPolicy
from .configuration_pose_act import PoseACTConfig
from .rgbd import fuse_pose_act_rgbd_observation

POSE_ACT_GRIPPER_INDEX = 9


def weighted_pose_act_l1_loss(
    target: Tensor,
    pred: Tensor,
    action_is_pad: Tensor,
    gripper_loss_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    l1 = F.l1_loss(target, pred, reduction="none")
    valid = ~action_is_pad.unsqueeze(-1)
    weights = torch.ones(target.shape[-1], dtype=l1.dtype, device=l1.device)
    weights[POSE_ACT_GRIPPER_INDEX] = gripper_loss_weight
    weighted_l1 = l1 * valid * weights
    l1_loss = weighted_l1.mean()

    valid_count = valid.sum().clamp_min(1)
    per_dim_l1 = (l1 * valid).sum(dim=(0, 1)) / valid_count
    gripper_l1 = per_dim_l1[POSE_ACT_GRIPPER_INDEX]
    loss_dict = {
        "l1_loss": l1_loss.item(),
        "xyz_l1_loss": per_dim_l1[:3].mean().item(),
        "rot6d_l1_loss": per_dim_l1[3:POSE_ACT_GRIPPER_INDEX].mean().item(),
        "gripper_l1_loss": gripper_l1.item(),
        "weighted_gripper_l1_loss": (gripper_l1 * gripper_loss_weight).item(),
        "gripper_loss_weight": float(gripper_loss_weight),
    }
    return l1_loss, loss_dict


class PoseACTPolicy(PreTrainedPolicy):
    """ACT-style policy over relative TCP pose chunks with observation history."""

    config_class = PoseACTConfig
    name = "pose_act"

    def __init__(self, config: PoseACTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        model_config = deepcopy(config)
        if model_config.use_rgbd_v2_inputs:
            model_config.input_features = {
                config.fisheye_rgb_key: deepcopy(config.input_features[config.fisheye_rgb_key]),
                config.depth_camera_rgb_key: deepcopy(config.input_features[config.depth_camera_rgb_key]),
                config.depth_mask_fused_key: PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(2, *config.input_features[config.depth_key].shape[1:]),
                ),
                OBS_STATE: deepcopy(config.robot_state_feature),
            }
        elif model_config.use_rgbd_inputs:
            model_config.input_features = {
                config.fisheye_rgb_key: deepcopy(config.input_features[config.fisheye_rgb_key]),
                config.rgbd_fused_key: PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(4, *config.input_features[config.depth_camera_rgb_key].shape[1:]),
                ),
                OBS_STATE: deepcopy(config.robot_state_feature),
            }
        if model_config.robot_state_feature is not None:
            step_dim = 10
            model_config.input_features = dict(model_config.input_features or {})
            model_config.input_features[OBS_STATE] = deepcopy(model_config.robot_state_feature)
            model_config.input_features[OBS_STATE].shape = (step_dim * config.n_obs_steps,)
        if model_config.action_feature is not None:
            model_config.output_features = dict(model_config.output_features or {})
            model_config.output_features[ACTION] = deepcopy(model_config.action_feature)
            model_config.output_features[ACTION].shape = (10,)
        self.model = ACT(model_config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith(("model.backbone", "model.backbones")) and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith(("model.backbone", "model.backbones")) and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._obs_queues = {OBS_STATE: deque([], maxlen=self.config.n_obs_steps)}
        for key in self.config.image_features:
            self._obs_queues[key] = deque([], maxlen=self.config.n_obs_steps)

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _move_batch_to_model_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        device = self._model_device()
        moved: dict[str, Tensor] = {}
        for key, value in batch.items():
            moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return moved

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = self._move_batch_to_model_device(dict(batch))
        if self.config.use_rgbd_inputs or self.config.use_rgbd_v2_inputs:
            batch = fuse_pose_act_rgbd_observation(batch, self.config)

        state = batch[OBS_STATE]
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.shape[1] != self.config.n_obs_steps:
            raise ValueError(
                f"pose_act expects {self.config.n_obs_steps} state steps, got shape {tuple(state.shape)}"
            )
        if state.shape[-1] == 7:
            state = make_relative_state_history(ensure_pose10d(state))
        elif state.shape[-1] != 10:
            raise ValueError(f"pose_act expects pose7d or pose10d state, got shape {tuple(state.shape)}")
        batch[OBS_STATE] = state.flatten(start_dim=1)

        batch[OBS_IMAGES] = []
        for key in self.config.model_image_feature_keys:
            images = batch[key]
            if images.ndim == 4:
                images = images.unsqueeze(1)
            if images.shape[1] != self.config.n_obs_steps:
                raise ValueError(
                    f"pose_act expects {self.config.n_obs_steps} image steps for {key}, "
                    f"got {tuple(images.shape)}"
                )
            batch[OBS_IMAGES].extend(images[:, t] for t in range(self.config.n_obs_steps))

        return batch

    def _append_observation(self, key: str, value: Tensor):
        queue = self._obs_queues[key]
        if len(queue) == 0:
            queue.extend(value.clone() for _ in range(self.config.n_obs_steps))
        else:
            queue.append(value.clone())

    def _stack_queued_observations(self) -> dict[str, Tensor]:
        batch: dict[str, Tensor] = {OBS_STATE: torch.stack(list(self._obs_queues[OBS_STATE]), dim=1)}
        for key in self.config.image_features:
            batch[key] = torch.stack(list(self._obs_queues[key]), dim=1)
        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = self._prepare_batch(batch)
        return self.model(batch)[0]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = dict(batch)
        if ACTION in batch:
            batch.pop(ACTION)

        self._append_observation(OBS_STATE, batch[OBS_STATE])
        for key in self.config.image_features:
            self._append_observation(key, batch[key])

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(self._stack_queued_observations())
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(self._stack_queued_observations())[
                :, : self.config.n_action_steps
            ]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self._prepare_batch(batch)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss, loss_dict = weighted_pose_act_l1_loss(
            batch[ACTION],
            actions_hat,
            batch["action_is_pad"],
            self.config.gripper_loss_weight,
        )
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict
