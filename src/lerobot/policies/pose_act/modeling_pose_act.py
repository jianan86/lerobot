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

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..act.modeling_act import ACT, ACTTemporalEnsembler
from ..pretrained import PreTrainedPolicy
from .configuration_pose_act import PoseACTConfig
from .utils import make_relative_state_history


class PoseACTPolicy(PreTrainedPolicy):
    """ACT-style policy over relative TCP pose chunks with observation history."""

    config_class = PoseACTConfig
    name = "pose_act"

    def __init__(self, config: PoseACTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        model_config = deepcopy(config)
        if model_config.robot_state_feature is not None:
            step_dim = model_config.robot_state_feature.shape[0]
            model_config.input_features = dict(model_config.input_features or {})
            model_config.input_features[OBS_STATE] = deepcopy(model_config.robot_state_feature)
            model_config.input_features[OBS_STATE].shape = (step_dim * config.n_obs_steps,)
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
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
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

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)

        state = batch[OBS_STATE]
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.shape[1] != self.config.n_obs_steps:
            raise ValueError(
                f"pose_act expects {self.config.n_obs_steps} state steps, got shape {tuple(state.shape)}"
            )
        batch[OBS_STATE] = make_relative_state_history(state).flatten(start_dim=1)

        batch[OBS_IMAGES] = []
        for key in self.config.image_features:
            images = batch[key]
            if images.ndim == 4:
                images = images.unsqueeze(1)
            if images.shape[1] != self.config.n_obs_steps:
                raise ValueError(
                    f"pose_act expects {self.config.n_obs_steps} image steps for {key}, got {tuple(images.shape)}"
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
            actions = self.predict_action_chunk(self._stack_queued_observations())[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self._prepare_batch(batch)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict
