#!/usr/bin/env python

import logging
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.configs import parser
from lerobot.replay import PiperReplayExecutor, PiperReplayTransportClient
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    piper_follower,
    reachy2,
    so_follower,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, log_say


@dataclass
class PiperReplayClientConfig:
    robot: RobotConfig
    server_address: str
    episode: int
    chunk_size: int = 64
    prefetch_threshold: int = 16
    fps: int | None = None
    play_sounds: bool = True


def _parse_server_address(value: str) -> tuple[str, int]:
    host, port_str = value.rsplit(":", maxsplit=1)
    return host, int(port_str)


@parser.wrap()
def replay(cfg: PiperReplayClientConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type != "piper_follower":
        raise ValueError(f"Dual-machine replay only supports robot.type='piper_follower', got {cfg.robot.type!r}.")

    host, port = _parse_server_address(cfg.server_address)
    transport = PiperReplayTransportClient(host=host, port=port)
    meta = transport.open_episode(cfg.episode)

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    try:
        log_say("Replaying episode", cfg.play_sounds, blocking=True)
        executor = PiperReplayExecutor(
            robot=robot,
            transport=transport,
            episode_token=meta.episode_token,
            chunk_size=cfg.chunk_size,
            prefetch_threshold=cfg.prefetch_threshold,
            fps=cfg.fps,
        )
        executor.run_episode()
    finally:
        robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
