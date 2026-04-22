#!/usr/bin/env bash
# Phase 1: run the robot client with a MockPiperFollower against the dummy server.
# No CAN/cameras required. Validates the gRPC pipeline end-to-end.
set -euo pipefail
cd "$(dirname "$0")/../.."

SERVER="${SERVER:-127.0.0.1:8080}"
FPS="${FPS:-30}"
CHUNK="${CHUNK:-20}"

exec python -m lerobot.async_inference.robot_client \
  --server_address="${SERVER}" \
  --robot.type=mock_piper_follower \
  --robot.id=mock0 \
  --robot.fps="${FPS}" \
  --policy_type=act \
  --pretrained_name_or_path=dummy \
  --policy_device=cpu \
  --client_device=cpu \
  --actions_per_chunk="${CHUNK}" \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only \
  --fps="${FPS}" \
  --task=mock
