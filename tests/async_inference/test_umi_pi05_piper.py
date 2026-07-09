from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lerobot.async_inference.adapters.umi_pi05_piper import (
    accelerate_gripper_closure,
    build_relative_state,
    limit_bimanual_pose7_steps,
    relative_actions_to_absolute_tcp,
    replace_zero_rot6d_with_identity,
)
from lerobot.scripts import lerobot_umi_pi05_piper_client as umi_client
from lerobot.scripts.lerobot_umi_pi05_piper_client import preprocess_fisheye
from lerobot.utils.pose_act import pose7d_to_pose10d, pose10d_to_pose7d, relative_pose10d


def _write_device_summary(path):
    path.write_text(
        """
        {
          "groups": [
            {
              "device_name": "right_gripper",
              "status": "ok",
              "physical_device_id": "right-id",
              "usb_port": {"devnode": "/dev/serial/by-id/right"},
              "fisheye": {"devnode": "/dev/video-right"}
            },
            {
              "device_name": "left_gripper",
              "status": "ok",
              "physical_device_id": "left-id",
              "usb_port": {"devnode": "/dev/serial/by-id/left"},
              "fisheye": {"devnode": "/dev/video-left"}
            }
          ]
        }
        """
    )


def test_load_pika_device_bindings(tmp_path):
    summary = tmp_path / "summary.json"
    _write_device_summary(summary)

    bindings = umi_client.load_pika_device_bindings(summary)

    assert bindings["right"].gripper_port == "/dev/serial/by-id/right"
    assert bindings["right"].fisheye_device == "/dev/video-right"
    assert bindings["left"].physical_device_id == "left-id"


def test_resolve_pika_devices_uses_bindings_for_missing_values(tmp_path):
    summary = tmp_path / "summary.json"
    _write_device_summary(summary)
    cfg = SimpleNamespace(
        arm_mode="dual",
        right_pika_gripper_port=None,
        right_pika_fisheye_device=None,
        left_pika_gripper_port=None,
        left_pika_fisheye_device=None,
        pika_device_summary_path=str(summary),
    )

    resolved = umi_client._resolve_pika_devices(cfg)

    assert resolved == (
        "/dev/serial/by-id/right",
        "/dev/video-right",
        "/dev/serial/by-id/left",
        "/dev/video-left",
    )


def test_resolve_pika_devices_manual_values_override_bindings(tmp_path):
    summary = tmp_path / "summary.json"
    _write_device_summary(summary)
    cfg = SimpleNamespace(
        arm_mode="single",
        right_pika_gripper_port="/dev/manual-gripper",
        right_pika_fisheye_device=3,
        left_pika_gripper_port=None,
        left_pika_fisheye_device=None,
        pika_device_summary_path=str(summary),
    )

    resolved = umi_client._resolve_pika_devices(cfg)

    assert resolved == ("/dev/manual-gripper", 3, None, None)


def test_resolve_pika_devices_errors_without_manual_or_summary():
    cfg = SimpleNamespace(
        arm_mode="single",
        right_pika_gripper_port=None,
        right_pika_fisheye_device=None,
        left_pika_gripper_port=None,
        left_pika_fisheye_device=None,
        pika_device_summary_path=None,
    )

    with pytest.raises(ValueError, match="pika_device_summary_path"):
        umi_client._resolve_pika_devices(cfg)


def test_build_relative_state_uses_previous_relative_to_current():
    previous = torch.tensor(
        [0.21, 0.02, 0.33, 0.1, -0.2, 0.3, 0.02, -0.18, 0.08, 0.24, -0.1, 0.2, -0.3, 0.03]
    )
    current = torch.tensor(
        [0.20, 0.00, 0.30, 0.05, -0.1, 0.2, 0.04, -0.20, 0.10, 0.25, -0.2, 0.1, -0.1, 0.05]
    )

    state = build_relative_state(previous, current)

    expected_right = relative_pose10d(pose7d_to_pose10d(previous[:7]), pose7d_to_pose10d(current[:7]))
    expected_left = relative_pose10d(pose7d_to_pose10d(previous[7:]), pose7d_to_pose10d(current[7:]))
    torch.testing.assert_close(state, torch.cat((expected_right, expected_left), dim=0))


def test_build_relative_state_zero_motion_has_identity_rotation():
    pose = torch.tensor([0.2, 0.0, 0.3, 0.1, -0.2, 0.3, 0.02, -0.2, 0.1, 0.25, -0.1, 0.2, -0.3, 0.03])

    state = build_relative_state(pose, pose)

    torch.testing.assert_close(state[:3], torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(state[3:9], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), atol=1e-6, rtol=0)
    assert state[9].item() == pytest.approx(0.02)
    torch.testing.assert_close(state[10:13], torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(state[13:19], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), atol=1e-6, rtol=0)
    assert state[19].item() == pytest.approx(0.03)


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


def test_relative_actions_to_absolute_tcp_accepts_pose7_base():
    base = torch.tensor([0.2, 0.0, 0.3, 0.1, -0.2, 0.3, 0.02, -0.2, 0.1, 0.25, -0.1, 0.2, -0.3, 0.03])
    relative = torch.zeros(1, 20)
    relative[:, 0] = 0.01
    relative[:, 10 + 1] = -0.02

    absolute = relative_actions_to_absolute_tcp(relative, base)

    assert absolute.shape == (1, 14)
    assert absolute[0, 6].item() == pytest.approx(0.0)
    assert absolute[0, 13].item() == pytest.approx(0.0)


def test_umi_pi05_client_does_not_import_relative_action_conversion():
    assert not hasattr(umi_client, "relative_actions_to_absolute_tcp")


def test_preprocess_fisheye_resize_pad_uint8_contiguous():
    image = torch.arange(10 * 20 * 3, dtype=torch.float32).reshape(10, 20, 3).numpy() / 255.0

    processed = preprocess_fisheye(image, size=16)

    assert processed.shape == (16, 16, 3)
    assert processed.dtype == "uint8"
    assert processed.flags.c_contiguous
    assert processed[:4].max() == 0
    assert processed[-4:].max() == 0
