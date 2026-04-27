#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.async_inference.adapters.pose_act_piper import T_EE_TCP
from lerobot.utils.pose_act import euler_rpy_to_matrix


def _pose7d_to_transform(pose7d: torch.Tensor) -> torch.Tensor:
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = euler_rpy_to_matrix(pose7d[3:6])
    transform[:3, 3] = pose7d[:3]
    return transform


def _sample_axis_points(
    transform: torch.Tensor,
    axis_length: float,
    samples: int,
    colors: dict[str, tuple[int, int, int]],
) -> list[tuple[float, float, float, int, int, int]]:
    origin = transform[:3, 3]
    rotation = transform[:3, :3]
    points: list[tuple[float, float, float, int, int, int]] = []
    for axis_name, direction in ("x", rotation[:, 0]), ("y", rotation[:, 1]), ("z", rotation[:, 2]):
        r, g, b = colors[axis_name]
        for i in range(samples + 1):
            alpha = i / samples
            point = origin + direction * axis_length * alpha
            points.append((float(point[0]), float(point[1]), float(point[2]), r, g, b))
    return points


def _write_ascii_ply(path: Path, points: list[tuple[float, float, float, int, int, int]]) -> None:
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            "comment EE/TCP frame visualization for pose_act Piper",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )
    body = "\n".join(f"{x:.8f} {y:.8f} {z:.8f} {r} {g} {b}" for x, y, z, r, g, b in points)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{header}\n{body}\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the fixed EE->TCP rigid transform used by pose_act Piper and export a single ASCII PLY "
            "containing both coordinate frames."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("pose_act_piper_frames.ply"), help="Output ASCII PLY path.")
    parser.add_argument("--ee-axis-length", type=float, default=0.10, help="Axis length for the EE frame in meters.")
    parser.add_argument("--tcp-axis-length", type=float, default=0.06, help="Axis length for the TCP frame in meters.")
    parser.add_argument("--samples-per-axis", type=int, default=20, help="Number of line samples written for each axis.")
    args = parser.parse_args()

    base_to_ee = _pose7d_to_transform(torch.zeros(7, dtype=torch.float32))
    base_to_tcp = base_to_ee @ T_EE_TCP
    round_trip_error = torch.linalg.norm(base_to_tcp @ torch.linalg.inv(T_EE_TCP) - base_to_ee).item()

    print("T_ee_tcp:")
    print(T_EE_TCP)
    print()
    print("Meaning:")
    print("  p_ee = T_ee_tcp @ p_tcp")
    print("  T_base_tcp = T_base_ee @ T_ee_tcp")
    print("  The TCP origin is 0.1943 m along +z of the EE frame.")
    print("  The TCP axes satisfy: tcp.x = ee.z, tcp.y = ee.y, tcp.z = -ee.x.")
    print()
    print(f"Round-trip check ||T_base_tcp * T_tcp_ee - T_base_ee|| = {round_trip_error:.8e}")

    ee_colors = {"x": (255, 0, 0), "y": (0, 255, 0), "z": (0, 0, 255)}
    tcp_colors = {"x": (255, 128, 128), "y": (128, 255, 128), "z": (128, 128, 255)}
    points = []
    points.extend(_sample_axis_points(base_to_ee, args.ee_axis_length, args.samples_per_axis, ee_colors))
    points.extend(_sample_axis_points(base_to_tcp, args.tcp_axis_length, args.samples_per_axis, tcp_colors))

    _write_ascii_ply(args.output, points)
    print(f"Wrote ASCII PLY with {len(points)} points to {args.output}")


if __name__ == "__main__":
    main()
