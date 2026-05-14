"""Convert a LeRobot pose_act checkpoint to a TensorRT engine.

The generated engine uses the same tensor contract as the async inference
server's TensorRT backend:

- input ``observation_state``: preprocessed pose_act state tensor
- inputs ``image_0..image_N``: image tensors after pose_act history flattening
- output ``actions``: ``(1, chunk_size, action_dim)``

Example:
    uv run python tools/convert_pose_act_to_tensorrt.py \
        --pretrained-model outputs/train/pose_act_0511/checkpoints/100000/pretrained_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.async_inference.tensorrt import PoseACTTensorRTPolicyAdapter
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pose_act.configuration_pose_act import PoseACTConfig  # noqa: F401
from lerobot.policies.pose_act.modeling_pose_act import PoseACTPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained-model",
        type=Path,
        required=True,
        help="Path to a LeRobot pose_act pretrained_model directory.",
    )
    parser.add_argument(
        "--engine-path",
        type=Path,
        default=None,
        help="Output path for the serialized TensorRT engine. Defaults to <pretrained-model>/model.engine.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=None,
        help="Output path for the intermediate ONNX model. Defaults to <pretrained-model>/model.onnx.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Optional JSON metadata path. Defaults to <pretrained-model>/model.engine.json.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device used for loading and exporting the policy, e.g. cuda or cuda:0.",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable FP16 TensorRT builder flag when supported.",
    )
    return parser.parse_args()


def load_pose_act_policy(pretrained_model: Path, device: str) -> PoseACTPolicy:
    if not pretrained_model.is_dir():
        raise FileNotFoundError(f"pretrained_model directory not found: {pretrained_model}")
    config = PreTrainedConfig.from_pretrained(pretrained_model, cli_overrides=[f"--device={device}"])
    if config.type != "pose_act":
        raise ValueError(f"Expected a pose_act checkpoint, got policy type {config.type!r}.")
    policy = PoseACTPolicy.from_pretrained(pretrained_model, config=config)
    policy.to(device)
    policy.eval()
    return policy


def write_metadata(
    adapter: PoseACTTensorRTPolicyAdapter, metadata_path: Path, checkpoint_path: Path, onnx_path: Path
) -> None:
    dummy_inputs = adapter._make_dummy_inputs()  # noqa: SLF001
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "policy_type": "pose_act",
        "engine_path": str(adapter.engine_path),
        "onnx_path": str(onnx_path),
        "input_names": adapter.input_names,
        "input_shapes": {
            name: list(tensor.shape) for name, tensor in zip(adapter.input_names, dummy_inputs, strict=True)
        },
        "output_name": adapter.output_name,
        "output_shape": [1, adapter.config.chunk_size, adapter.action_dim],
        "async_server_args": {
            "inference_backend": "tensorrt",
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT conversion requires CUDA, but torch.cuda.is_available() is False.")

    policy = load_pose_act_policy(args.pretrained_model, args.device)
    engine_path = args.engine_path or args.pretrained_model / "model.engine"
    onnx_path = args.onnx_path or args.pretrained_model / "model.onnx"
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    adapter = PoseACTTensorRTPolicyAdapter(
        policy,
        engine_path,
        build_engine=True,
        fp16=args.fp16,
        onnx_path=onnx_path,
    )

    metadata_path = args.metadata_path or engine_path.with_suffix(engine_path.suffix + ".json")
    write_metadata(adapter, metadata_path, args.pretrained_model, onnx_path)

    print(f"Wrote TensorRT engine: {engine_path}")
    print(f"Wrote ONNX model: {onnx_path}")
    print(f"Wrote async input metadata: {metadata_path}")
    print(
        "Start the async policy server with: "
        "--inference_backend=tensorrt"
    )


if __name__ == "__main__":
    main()
