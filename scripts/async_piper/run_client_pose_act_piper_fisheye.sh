#!/usr/bin/env bash
# Phase 2 (USB fisheye variant): real piper_follower + pose_act shell server on LAN IP.
# Uses a UVC fisheye camera (OpenCV backend) instead of the Intel RealSense.
# SAFETY: run verify_piper_motion.py first; start with a conservative max_relative_target.
set -euo pipefail
cd "$(dirname "$0")/../.."

SERVER="${SERVER:-192.168.10.98:8080}"
CAN_NAME="${CAN_NAME:-can0}"
FPS="${FPS:-30}"
CHUNK="${CHUNK:-20}"
MAX_REL="${MAX_REL:-0.01}"  # meters / radians per step; tune for safety!
# UVC fisheye device path. Prefer udev symlinks (e.g. /dev/video60 created by pika_ros
# scripts/start_single_gripper.bash) over raw indices so it survives re-plug / reboot.
FISHEYE_DEV="${FISHEYE_DEV:-/dev/video60}"

exec python -m lerobot.async_inference.robot_client \
  --server_address="${SERVER}" \
  --robot.type=piper_follower \
  --robot.can_name="${CAN_NAME}" \
  --robot.id=piper0 \
  --robot.max_relative_target="${MAX_REL}" \
  --robot.cameras="{ fisheye_rgb: {type: opencv, index_or_path: \"${FISHEYE_DEV}\", width: 640, height: 480, fps: 30} }" \
  --policy_type=pose_act \
  --pretrained_name_or_path=shell \
  --policy_device=cpu \
  --client_device=cpu \
  --actions_per_chunk="${CHUNK}" \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only \
  --fps="${FPS}" \
  --task=pose_act_shell
