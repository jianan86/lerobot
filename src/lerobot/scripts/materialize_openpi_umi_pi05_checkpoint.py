#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.policies.umi_pi05 import materialize_openpi_umi_pi05_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write LeRobot config and processor files into an OpenPI UMI pi05 checkpoint directory."
    )
    parser.add_argument("checkpoint", type=Path, help="OpenPI checkpoint directory to update in-place.")
    parser.add_argument("--device", default="cpu", help="Device recorded in the generated LeRobot config.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing LeRobot config files.")
    args = parser.parse_args()

    written = materialize_openpi_umi_pi05_checkpoint(
        args.checkpoint,
        device=args.device,
        overwrite=args.overwrite,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
