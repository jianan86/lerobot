#!/usr/bin/env bash
# Phase 2 dry-run: mock_piper_follower + pose_act shell server.
# Exercises the full PoseActPiperAdapter relative->absolute->ee.abs_* path with no hardware.
set -euo pipefail
cd "$(dirname "$0")/../.."

SERVER="${SERVER:-127.0.0.1:8080}"
FPS="${FPS:-30}"
CHUNK="${CHUNK:-20}"
PRETRAINED="${PRETRAINED:-outputs/async_piper/pose_act_random}"

exec python -m lerobot.async_inference.robot_client \
  --server_address="${SERVER}" \
  --robot.type=mock_piper_follower \
  --robot.id=mock0 \
  --robot.fps="${FPS}" \
  --robot.cameras="{ fisheye_rgb: {height: 64, width: 64, fps: 30} }" \
  --policy_type=pose_act \
  --pretrained_name_or_path="${PRETRAINED}" \
  --policy_device=cpu \
  --client_device=cpu \
  --actions_per_chunk="${CHUNK}" \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only \
  --fps="${FPS}" \
  --task=pose_act_random
