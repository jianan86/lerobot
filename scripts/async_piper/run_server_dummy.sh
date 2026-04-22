#!/usr/bin/env bash
# Phase 1: run the policy server in dummy mode on localhost.
# No checkpoint is loaded; the server returns zero action chunks of `dummy_action_dim` floats.
set -euo pipefail
cd "$(dirname "$0")/../.."

ACTION_DIM="${ACTION_DIM:-7}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"

exec python -m lerobot.async_inference.policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --inference_latency=0.033 \
  --obs_queue_timeout=2 \
  --dummy_policy=true \
  --dummy_action_dim="${ACTION_DIM}"
