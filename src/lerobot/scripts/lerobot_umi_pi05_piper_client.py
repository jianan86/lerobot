#!/usr/bin/env python

from __future__ import annotations

import pickle  # nosec
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import grpc
import numpy as np
import torch

from lerobot.async_inference.adapters import (
    UmiPI05PiperAdapter,
    accelerate_gripper_closure,
    limit_bimanual_pose7_steps,
    relative_actions_to_absolute_tcp,
)
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.async_inference.robot_client import PikaGripper
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class UmiPI05PiperClientConfig:
    right_robot: RobotConfig
    left_robot: RobotConfig
    pretrained_name_or_path: str
    instruction: str
    device: str = "cuda"
    actions_per_chunk: int = 10
    fps: float = 10.0
    server_address: str = "127.0.0.1:8080"
    no_tunnel: bool = False
    ssh_host: str = "jianan@183.230.224.121"
    ssh_port: int = 50210
    remote_grpc_port: int = 8080
    right_pika_gripper_port: str | None = None
    left_pika_gripper_port: str | None = None
    pika_gripper_min_width_m: float = 0.0
    pika_gripper_max_width_m: float = 0.085
    max_xyz_step: float = 0.03
    max_rpy_step: float = 0.35
    gripper_close_threshold: float = 0.04


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_tunnel(cfg: UmiPI05PiperClientConfig) -> tuple[str, subprocess.Popen[bytes] | None]:
    if cfg.no_tunnel:
        return cfg.server_address, None
    local_port = _free_port()
    proc = subprocess.Popen(  # noqa: S603
        [
            "ssh",
            "-N",
            "-L",
            f"127.0.0.1:{local_port}:127.0.0.1:{cfg.remote_grpc_port}",
            "-p",
            str(cfg.ssh_port),
            cfg.ssh_host,
        ]
    )
    time.sleep(0.5)
    return f"127.0.0.1:{local_port}", proc


def _setup_gripper(port: str | None, cfg: UmiPI05PiperClientConfig) -> PikaGripper | None:
    if port is None:
        return None
    gripper = PikaGripper(
        port,
        min_width_m=cfg.pika_gripper_min_width_m,
        max_width_m=cfg.pika_gripper_max_width_m,
    )
    gripper.connect()
    return gripper


def _black_image() -> np.ndarray:
    return np.zeros((224, 224, 3), dtype=np.uint8)


def _camera_or_black(robot, names: tuple[str, ...]) -> np.ndarray:
    get_observation = getattr(robot, "get_observation", None)
    if get_observation is None:
        return _black_image()
    observation = get_observation()
    for name in names:
        value = observation.get(name)
        if value is not None and getattr(value, "ndim", 0) == 3:
            return np.asarray(value, dtype=np.uint8)
    return _black_image()


def _send_observation(stub, observation: TimedObservation) -> None:
    payload = pickle.dumps(observation)  # nosec
    chunks = send_bytes_in_chunks(payload, services_pb2.Observation, log_prefix="[umi_pi05]")
    stub.SendObservations(chunks)


def _get_actions(stub) -> list:
    response = stub.GetActions(services_pb2.Empty())
    if not response.data:
        return []
    return pickle.loads(response.data)  # nosec


@draccus.wrap()
def run(cfg: UmiPI05PiperClientConfig) -> None:
    init_logging()
    print(pformat(asdict(cfg)))

    address, tunnel = _start_tunnel(cfg)
    right_robot = make_robot_from_config(cfg.right_robot)
    left_robot = make_robot_from_config(cfg.left_robot)
    right_gripper = _setup_gripper(cfg.right_pika_gripper_port, cfg)
    left_gripper = _setup_gripper(cfg.left_pika_gripper_port, cfg)

    channel = grpc.insecure_channel(address)
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    adapter = UmiPI05PiperAdapter(right_robot, left_robot, right_gripper=right_gripper, left_gripper=left_gripper)

    try:
        right_robot.connect()
        left_robot.connect()
        stub.Ready(services_pb2.Empty())
        policy_config = RemotePolicyConfig(
            policy_type="umi_pi05",
            pretrained_name_or_path=cfg.pretrained_name_or_path,
            lerobot_features={},
            actions_per_chunk=cfg.actions_per_chunk,
            device=cfg.device,
        )
        stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy_config)))  # nosec

        timestep = 0
        dt = 1.0 / cfg.fps
        while True:
            start = time.perf_counter()
            state = adapter.current_tcp_pose10_state()
            obs = {
                OBS_STATE: state.numpy(),
                f"{OBS_IMAGES}.cam_high": _black_image(),
                f"{OBS_IMAGES}.cam_left_wrist": _camera_or_black(left_robot, ("cam_left_wrist", "fisheye_rgb")),
                f"{OBS_IMAGES}.cam_right_wrist": _camera_or_black(right_robot, ("cam_right_wrist", "fisheye_rgb")),
                "task": cfg.instruction,
                "async_loop_request_id": timestep,
            }
            _send_observation(stub, TimedObservation(time.time(), timestep, obs, must_go=True))
            timed_actions = _get_actions(stub)
            if timed_actions:
                relative = torch.stack([a.get_action().to(torch.float32).cpu() for a in timed_actions], dim=0)
                tcp_actions = relative_actions_to_absolute_tcp(relative, state)
                tcp_actions = accelerate_gripper_closure(
                    tcp_actions,
                    close_threshold=cfg.gripper_close_threshold,
                )
                tcp_actions = limit_bimanual_pose7_steps(
                    tcp_actions,
                    max_xyz_step=cfg.max_xyz_step,
                    max_rpy_step=cfg.max_rpy_step,
                )
                right_action, left_action = adapter.tcp_pose7_to_ee_actions(tcp_actions[0])
                right_width = right_action.pop("gripper.pos", None)
                left_width = left_action.pop("gripper.pos", None)
                right_robot.send_action(right_action)
                left_robot.send_action(left_action)
                if right_gripper is not None and right_width is not None:
                    right_gripper.execute_width(right_width)
                if left_gripper is not None and left_width is not None:
                    left_gripper.execute_width(left_width)

            timestep += 1
            time.sleep(max(0.0, dt - (time.perf_counter() - start)))
    finally:
        right_robot.disconnect()
        left_robot.disconnect()
        if right_gripper is not None:
            right_gripper.disconnect()
        if left_gripper is not None:
            left_gripper.disconnect()
        channel.close()
        if tunnel is not None:
            tunnel.terminate()


def main() -> None:
    register_third_party_plugins()
    run()


if __name__ == "__main__":
    main()
