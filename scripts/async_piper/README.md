# Piper async inference (scripts)

End-to-end helpers for the async gRPC pipeline (`lerobot.async_inference`).

## Phase 1: framework / gRPC validation (localhost, no hardware)

Two terminals on the same host:

```bash
# Terminal 1 - dummy policy server (returns zero action chunks)
bash scripts/async_piper/run_server_dummy.sh

# Terminal 2 - mock Piper client, drives observation/action loop
bash scripts/async_piper/run_client_mock.sh
```

Expected: server logs `Action chunk #... generated`, client logs `[MockPiperFollower] action#...`.

## Phase 2: pose_act real policy server + Piper

Generate a local random-weight checkpoint first:

```bash
python scripts/async_piper/create_pose_act_random_checkpoint.py \
  --output-dir outputs/async_piper/pose_act_random \
  --image-key observation.images.fisheye_rgb \
  --overwrite
```

Then run the real server path and a client. The client sends real images plus
virtual base-frame pose7d state `[x,y,z,roll,pitch,yaw,gripper_width]`; the
server converts it to pose10d/history, runs `PoseACTPolicy`, logs inference
timing, and returns base-frame pose7d actions.

```bash
# A
bash scripts/async_piper/run_server_pose_act_random.sh

# B, no hardware
bash scripts/async_piper/run_client_pose_act_mock.sh
```

For the USB fisheye Piper client:

```bash
SERVER=<B_IP>:8080 CAN_NAME=can0 MAX_REL=0.01 \
  bash scripts/async_piper/run_client_pose_act_piper_fisheye.sh
```

For the RealSense `front` client, generate a matching checkpoint:

```bash
python scripts/async_piper/create_pose_act_random_checkpoint.py \
  --output-dir outputs/async_piper/pose_act_random_front \
  --image-key observation.images.front \
  --overwrite
```

## Phase 3: pose_act shell server + Piper

The shell server (`--pose_act_shell=true`) does NOT load a checkpoint. On each
observation it:
1. validates and prints the observation (state shape, camera keys, image shapes);
2. sleeps `inference_latency`;
3. returns a `(actions_per_chunk, 10)` zero relative pose10d chunk.

Client-side, `PoseActPiperAdapter` takes each relative pose10d step, reads the
live Piper TCP, maps to an absolute target and emits `ee.abs_x/y/z/rx/ry/rz` +
`gripper.pos`. `PiperFollower._send_ee_abs_action` then issues
`MotionCtrl_2 + EndPoseCtrl`.

### 3a. Dry-run with MockPiperFollower (no hardware)

```bash
# A
bash scripts/async_piper/run_server_pose_act_shell.sh

# B
PRETRAINED=shell bash scripts/async_piper/run_client_pose_act_mock.sh
```

### 3b. Real Piper on LAN (A = client with CAN/cams, B = server)

On B (server):
```bash
HOST=0.0.0.0 bash scripts/async_piper/run_server_pose_act_shell.sh
```

On A (client):
```bash
SERVER=<B_IP>:8080 CAN_NAME=can0 MAX_REL=0.01 \
  PRETRAINED=shell bash scripts/async_piper/run_client_pose_act_piper.sh
```

SAFETY: run `python -m lerobot.scripts.lerobot_teleoperate` or
`tests/hardware/verify_piper_motion.py` first and keep `MAX_REL` conservative
during shell testing; zero relative pose10d does not perfectly round-trip
through rot6d -> matrix -> euler, so expect small commanded jitter that gets
clipped by `max_relative_target`.
