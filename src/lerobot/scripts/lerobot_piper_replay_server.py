#!/usr/bin/env python

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.replay import PiperReplayDatasetProvider, build_replay_request_handler
from lerobot.replay.piper_remote import ThreadedTCPServer
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class ReplayDatasetConfig:
    repo_id: str
    root: str | Path | None = None


@dataclass
class PiperReplayServerConfig:
    dataset: ReplayDatasetConfig
    host: str = "127.0.0.1"
    port: int = 8090


@parser.wrap()
def serve(cfg: PiperReplayServerConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    provider = PiperReplayDatasetProvider(cfg.dataset.repo_id, root=cfg.dataset.root)
    handler = build_replay_request_handler(provider)
    with ThreadedTCPServer((cfg.host, cfg.port), handler) as server:
        logging.info("Piper replay server listening on %s:%s", cfg.host, cfg.port)
        server.serve_forever()


def main() -> None:
    register_third_party_plugins()
    serve()


if __name__ == "__main__":
    main()
