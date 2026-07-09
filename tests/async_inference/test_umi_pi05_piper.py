from __future__ import annotations

import pytest
import torch

from lerobot.async_inference.adapters.umi_pi05_piper import (
    accelerate_gripper_closure,
    limit_bimanual_pose7_steps,
    relative_actions_to_absolute_tcp,
    replace_zero_rot6d_with_identity,
)
from lerobot.utils.pose_act import pose7d_to_pose10d, pose10d_to_pose7d


def test_zero_rot6d_identity_fallback():
    pose = torch.zeros(10)

    fixed = replace_zero_rot6d_with_identity(pose)

    torch.testing.assert_close(fixed[3:9], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


def test_relative_actions_to_absolute_tcp_zero_action_holds_pose():
    right = pose7d_to_pose10d(torch.tensor([0.2, 0.0, 0.3, 0.1, -0.2, 0.3, 0.02]))
    left = pose7d_to_pose10d(torch.tensor([-0.2, 0.1, 0.25, -0.1, 0.2, -0.3, 0.03]))
    state = torch.cat((right, left), dim=0)
    relative = torch.zeros(2, 20)

    absolute = relative_actions_to_absolute_tcp(relative, state)

    expected = torch.cat((pose10d_to_pose7d(right), pose10d_to_pose7d(left)), dim=0)
    expected[6] = 0.0
    expected[13] = 0.0
    torch.testing.assert_close(absolute[0], expected, rtol=0, atol=1e-5)
    torch.testing.assert_close(absolute[1], expected, rtol=0, atol=1e-5)


def test_relative_actions_to_absolute_tcp_translates_each_arm():
    state = torch.cat((pose7d_to_pose10d(torch.zeros(7)), pose7d_to_pose10d(torch.zeros(7))), dim=0)
    relative = torch.zeros(1, 20)
    relative[:, 0] = 0.01
    relative[:, 10 + 1] = -0.02

    absolute = relative_actions_to_absolute_tcp(relative, state)

    assert absolute[0, 0].item() == pytest.approx(0.01)
    assert absolute[0, 8].item() == pytest.approx(-0.02)


def test_accelerate_gripper_closure_for_both_arms():
    actions = torch.ones(2, 14) * 0.05
    actions[0, 6] = 0.03
    actions[1, 13] = 0.01

    adjusted = accelerate_gripper_closure(actions, close_threshold=0.04, target_width=0.0)

    assert adjusted[0, 6].item() == pytest.approx(0.0)
    assert adjusted[1, 13].item() == pytest.approx(0.0)
    assert adjusted[1, 6].item() == pytest.approx(0.05)


def test_limit_bimanual_pose7_steps_clamps_xyz_and_rpy():
    actions = torch.zeros(2, 14)
    actions[1, 0] = 1.0
    actions[1, 3] = 1.0
    actions[1, 7 + 2] = -1.0
    actions[1, 7 + 5] = -1.0

    limited = limit_bimanual_pose7_steps(actions, max_xyz_step=0.03, max_rpy_step=0.2)

    assert limited[1, 0].item() == pytest.approx(0.03)
    assert limited[1, 3].item() == pytest.approx(0.2)
    assert limited[1, 9].item() == pytest.approx(-0.03)
    assert limited[1, 12].item() == pytest.approx(-0.2)
