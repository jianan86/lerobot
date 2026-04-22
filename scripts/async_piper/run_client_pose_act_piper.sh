#!/usr/bin/env bash
# Phase 2: real piper_follower + pose_act shell server on LAN IP (B-point).
# SAFETY: run verify_piper_motion.py first; start with a conservative max_relative_target.
set -euo pipefail
cd "$(dirname "$0")/../.."

SERVER="${SERVER:-192.168.10.98:8080}"
CAN_NAME="${CAN_NAME:-can0}"
FPS="${FPS:-30}"
CHUNK="${CHUNK:-20}"
MAX_REL="${MAX_REL:-0.01}"  # meters / radians per step; tune for safety!
PRETRAINED="${PRETRAINED:-outputs/async_piper/pose_act_random_front}"
RS_SERIAL="${RS_SERIAL:-412622273326}"  # RealSense D405 serial number; override via env if needed.

exec python -m lerobot.async_inference.robot_client \
  --server_address="${SERVER}" \
  --robot.type=piper_follower \
  --robot.can_name="${CAN_NAME}" \
  --robot.id=piper0 \
  --robot.max_relative_target="${MAX_REL}" \
  --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: \"${RS_SERIAL}\", width: 640, height: 480, fps: 30} }" \
  --policy_type=pose_act \
  --pretrained_name_or_path="${PRETRAINED}" \
  --policy_device=cpu \
  --client_device=cpu \
  --actions_per_chunk="${CHUNK}" \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only \
  --fps="${FPS}" \
  --task=pose_act_random
