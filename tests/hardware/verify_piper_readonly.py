#!/usr/bin/env python
"""Read-only Piper follower sanity check.

Connects the PiperFollower robot (via piper_sdk) to can0, reads joint
positions, end-effector pose and gripper state a few times, then disconnects
without disabling the arm. No motion commands are issued.
"""

from __future__ import annotations

import time

from lerobot.robots.piper_follower import PiperFollower, PiperFollowerConfig


def main() -> None:
    cfg = PiperFollowerConfig(
        can_name="can0",
        enable_on_connect=False,  # do not enable motors for read-only test
        disable_on_disconnect=False,
    )
    robot = PiperFollower(cfg)
    print("[1/3] Connecting to Piper on can0 (read-only, motors NOT enabled)...")
    robot.connect()
    print("      connected =", robot.is_connected)

    print("[2/3] Reading joint/gripper state and end-effector pose...")
    for i in range(5):
        obs = robot.get_observation()
        pose = robot._get_end_pose()
        joints = {k: round(v, 4) for k, v in obs.items() if k.endswith(".pos")}
        pose_r = {k: round(v, 4) for k, v in pose.items()}
        print(f"  iter {i}: joints={joints}  ee_pose={pose_r}")
        time.sleep(0.2)

    print("[3/3] Disconnecting (leaving motors in their current state)...")
    robot.disconnect()
    print("OK - read-only verification complete.")


if __name__ == "__main__":
    main()
