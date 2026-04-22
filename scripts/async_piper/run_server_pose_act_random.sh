#!/usr/bin/env bash
# Run the real async policy server path. The client chooses the pose_act checkpoint.
set -euo pipefail
cd "$(dirname "$0")/../.."

SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8080}"
FPS="${FPS:-30}"
LATENCY="${LATENCY:-0.0}"
SAFE_RETURN="${SAFE_RETURN:-true}"

exec python -m lerobot.async_inference.policy_server \
  --host="${SERVER_HOST}" \
  --port="${SERVER_PORT}" \
  --fps="${FPS}" \
  --inference_latency="${LATENCY}" \
  --obs_queue_timeout=2 \
  --pose_act_safe_return_current_pose="${SAFE_RETURN}"
