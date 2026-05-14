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

import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


class PoseACTTensorRTExportWrapper(nn.Module):
    """Tensor-only wrapper around PoseACT's ACT module for ONNX/TensorRT export."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, observation_state: Tensor, *images: Tensor) -> Tensor:
        actions, _ = self.model({OBS_STATE: observation_state, OBS_IMAGES: list(images)})
        return actions


class PoseACTTensorRTPolicyAdapter:
    """Runs a loaded pose_act policy's neural network through a TensorRT engine."""

    def __init__(
        self,
        policy: Any,
        engine_path: str | Path | None,
        *,
        build_engine: bool = False,
        fp16: bool = True,
        onnx_path: str | Path | None = None,
    ) -> None:
        if policy.name != "pose_act":
            raise ValueError(f"TensorRT backend currently supports only pose_act, got {policy.name!r}.")
        if policy.config.temporal_ensemble_coeff is not None:
            raise ValueError("TensorRT backend supports pose_act action chunks, not temporal ensembling.")
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT backend requires CUDA, but torch.cuda.is_available() is False.")

        self.policy = policy
        self.config = policy.config
        self.engine_path = Path(engine_path) if engine_path is not None else None
        self.onnx_path = Path(onnx_path) if onnx_path is not None else None
        n_image_inputs = self.config.n_obs_steps * len(self.config.image_features)
        self.input_names = ["observation_state", *[f"image_{i}" for i in range(n_image_inputs)]]
        self.output_name = "actions"
        self.action_dim = self.policy.model.config.action_feature.shape[0]

        trt = _import_tensorrt()
        self.trt = trt

        if build_engine:
            engine_bytes = self._build_engine(fp16=fp16)
            if self.engine_path is not None:
                self.engine_path.parent.mkdir(parents=True, exist_ok=True)
                self.engine_path.write_bytes(engine_bytes)
        else:
            if self.engine_path is None:
                raise ValueError("TensorRT engine path is required when build_engine is False.")
            if not self.engine_path.is_file():
                raise FileNotFoundError(
                    f"TensorRT engine not found: {self.engine_path}. "
                    "Expected a model.engine file in the pretrained model directory. "
                    "Run tools/convert_pose_act_to_tensorrt.py first."
                )
            engine_bytes = self.engine_path.read_bytes()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")
        self._validate_engine_io()

    @torch.no_grad()
    def predict_action_chunk(self, observation: dict[str, Tensor]) -> Tensor:
        inputs = self._prepare_inputs(observation)
        output = torch.empty(
            (inputs[0].shape[0], self.config.chunk_size, self.action_dim),
            dtype=self._torch_dtype(self.engine.get_tensor_dtype(self.output_name)),
            device=inputs[0].device,
        )

        self.context.set_tensor_address(self.output_name, output.data_ptr())
        for name, tensor in zip(self.input_names, inputs, strict=True):
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        stream = torch.cuda.current_stream(device=output.device)
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT inference failed.")
        return output

    def _prepare_inputs(self, observation: dict[str, Tensor]) -> list[Tensor]:
        prepared = self.policy._prepare_batch(observation)  # noqa: SLF001
        state = prepared[OBS_STATE].to(dtype=torch.float32).contiguous()
        images = [image.to(dtype=torch.float32).contiguous() for image in prepared[OBS_IMAGES]]
        return [state, *images]

    def _build_engine(self, *, fp16: bool) -> bytes:
        trt = self.trt
        self.policy.eval()
        device = next(self.policy.model.parameters()).device
        wrapper = PoseACTTensorRTExportWrapper(self.policy.model).eval().to(device)
        dummy_inputs = self._make_dummy_inputs()

        if self.onnx_path is None:
            onnx_file = tempfile.NamedTemporaryFile(suffix=".onnx")
            onnx_path = Path(onnx_file.name)
        else:
            self.onnx_path.parent.mkdir(parents=True, exist_ok=True)
            onnx_file = None
            onnx_path = self.onnx_path

        try:
            self._export_onnx(wrapper, dummy_inputs, onnx_path)
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(flags)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(onnx_path.read_bytes()):
                errors = [parser.get_error(i).desc() for i in range(parser.num_errors)]
                raise RuntimeError(f"Failed to parse pose_act ONNX for TensorRT: {errors}")

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("Failed to build TensorRT engine from pose_act ONNX.")
            return bytes(serialized)
        finally:
            if onnx_file is not None:
                onnx_file.close()

    def _export_onnx(self, wrapper: nn.Module, dummy_inputs: list[Tensor], onnx_path: Path) -> None:
        torch.onnx.export(
            wrapper,
            tuple(dummy_inputs),
            str(onnx_path),
            input_names=self.input_names,
            output_names=[self.output_name],
            opset_version=17,
            # PyTorch's ONNX constant-folding pass can mix CPU constants with CUDA
            # graph tensors for ACT's positional embeddings. TensorRT performs its
            # own folding/optimization after parsing the ONNX graph.
            do_constant_folding=False,
        )

    def _make_dummy_inputs(self) -> list[Tensor]:
        device = next(self.policy.model.parameters()).device
        batch = {
            OBS_STATE: torch.zeros((1, self.config.n_obs_steps, 10), dtype=torch.float32, device=device),
        }
        for key, feature in self.config.image_features.items():
            batch[key] = torch.zeros(
                (1, self.config.n_obs_steps, *feature.shape), dtype=torch.float32, device=device
            )
        prepared = self.policy._prepare_batch(batch)  # noqa: SLF001
        return [prepared[OBS_STATE], *prepared[OBS_IMAGES]]

    def _validate_engine_io(self) -> None:
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        missing = [name for name in [*self.input_names, self.output_name] if name not in names]
        if missing:
            raise RuntimeError(f"TensorRT engine is missing expected tensor(s): {missing}")

    def _torch_dtype(self, trt_dtype: Any) -> torch.dtype:
        if trt_dtype == self.trt.float16:
            return torch.float16
        if trt_dtype == self.trt.float32:
            return torch.float32
        raise TypeError(f"Unsupported TensorRT output dtype: {trt_dtype}")


def _import_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as e:
        raise RuntimeError(
            "TensorRT backend requires the `tensorrt` Python package installed on the policy server."
        ) from e
    return trt
