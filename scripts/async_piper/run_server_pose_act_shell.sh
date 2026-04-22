#!/usr/bin/env bash
# Phase 2: run the policy server in pose_act shell mode.
# Accepts real pose_act observation shape, validates + prints, sleeps inference_latency,
# returns a (actions_per_chunk, 10) relative pose10d chunk.
#
# Motion semantics (SHELL_MOTION):
#   - zero            : all-zero chunk; client holds current TCP / gripper (safe default for regression).
#   - drift_stop_osc  : +1mm/step relative x for ~1s, then stops; gripper sinusoid ~0.5-3.5cm @ 0.5Hz.
#                       Intended to exercise the full client decoding + EndPoseCtrl + GripperCtrl path
#                       while staying well inside --robot.max_relative_target safety caps.
set -euo pipefail
cd "$(dirname "$0")/../.."

# NOTE: 避免使用 $HOST，conda 会把它设成 x86_64-conda-linux-gnu（构建三元组）。
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8080}"
FPS="${FPS:-30}"
LATENCY="${LATENCY:-0.5}"
SHELL_MOTION="${SHELL_MOTION:-drift_stop_osc}"

exec python -m lerobot.async_inference.policy_server \
  --host="${SERVER_HOST}" \
  --port="${SERVER_PORT}" \
  --fps="${FPS}" \
  --inference_latency="${LATENCY}" \
  --obs_queue_timeout=2 \
  --pose_act_shell=true \
  --pose_act_shell_motion="${SHELL_MOTION}"
