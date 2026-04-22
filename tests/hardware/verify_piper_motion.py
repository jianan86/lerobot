#!/usr/bin/env python
"""Minimal, conservative motion test for PiperFollower.

Moves joint_1 gently within +/- 0.05 rad (~3 degrees) using small 0.01 rad
increments at 10% speed rate, then returns to the original pose. All other
joints are held at their starting positions. A safety cap `max_relative_target`
ensures no individual command exceeds 0.02 rad vs. the current feedback.

Press Ctrl+C to abort; the script will try to hold position on exit.
"""

from __future__ import annotations

import time

from lerobot.robots.piper_follower import PiperFollower, PiperFollowerConfig

SPEED_RATE = 10  # percent
STEP = 0.01  # rad per command
AMPLITUDE = 0.05  # rad, peak offset from start
DWELL = 0.3  # s between commands
MAX_REL = 0.02  # rad, hard cap per command vs current feedback


def go_to(robot: PiperFollower, targets: dict[str, float]) -> None:
    action = {f"{name}.pos": float(val) for name, val in targets.items()}
    robot.send_action(action)


def main() -> None:
    cfg = PiperFollowerConfig(
        can_name="can0",
        enable_on_connect=True,
        disable_on_disconnect=False,  # leave enabled so the arm holds position
        move_spd_rate_ctrl=SPEED_RATE,
        max_relative_target=MAX_REL,
    )
    robot = PiperFollower(cfg)
    print(f"Connecting, enabling motors, speed={SPEED_RATE}%, max_rel={MAX_REL} rad...")
    robot.connect()

    start = {k.removesuffix(".pos"): v for k, v in robot._get_motor_positions().items()}
    print("Start pose:", {k: round(v, 4) for k, v in start.items()})

    j1_start = start["joint_1"]

    try:
        # Build waypoint sequence: 0 -> +A -> -A -> 0 in STEP increments.
        def ramp(a: float, b: float):
            n = max(1, int(abs(b - a) / STEP))
            return [a + (b - a) * (i + 1) / n for i in range(n)]

        waypoints = (
            ramp(0.0, +AMPLITUDE) + ramp(+AMPLITUDE, -AMPLITUDE) + ramp(-AMPLITUDE, 0.0)
        )

        for i, offset in enumerate(waypoints):
            target = {**start, "joint_1": j1_start + offset}
            go_to(robot, target)
            time.sleep(DWELL)
            cur = robot._get_motor_positions()["joint_1.pos"]
            print(
                f"  step {i + 1:02d}/{len(waypoints)}: cmd_j1={j1_start + offset:+.4f}  "
                f"fb_j1={cur:+.4f}  err={cur - (j1_start + offset):+.4f}"
            )

        # Final hold at start.
        go_to(robot, start)
        time.sleep(0.5)
        final = robot._get_motor_positions()
        print("Final fb:", {k: round(v, 4) for k, v in final.items()})
        print("OK - motion verification complete.")
    except KeyboardInterrupt:
        print("\nInterrupted, holding current position.")
    finally:
        robot.disconnect()
        print("Disconnected (motors left enabled to hold pose).")


if __name__ == "__main__":
    main()
