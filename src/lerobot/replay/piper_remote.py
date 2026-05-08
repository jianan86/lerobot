#!/usr/bin/env python

from __future__ import annotations

import pickle  # nosec
import socket
import socketserver
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.async_inference.adapters.pose_act_piper import PoseActPiperAdapter
from lerobot.datasets import LeRobotDataset
from lerobot.robots import Robot
from lerobot.utils.constants import ACTION
from lerobot.utils.pose_act import absolute_pose10d, pose10d_to_pose7d, relative_pose10d
from lerobot.utils.robot_utils import precise_sleep

POSE7D_ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper_width")


@dataclass(frozen=True)
class EpisodeMeta:
    episode_token: str
    episode_index: int
    num_frames: int
    fps: int
    action_names: tuple[str, ...]


@dataclass(frozen=True)
class RelativePoseChunk:
    episode_token: str
    frame_start: int
    poses: list[list[float]]


@dataclass(frozen=True)
class _LoadedEpisode:
    meta: EpisodeMeta
    relative_actions: torch.Tensor


def _validate_pose7d_action_names(names: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if names is None:
        raise ValueError("Replay dataset action feature is missing names; expected pose7d action schema.")

    action_names = tuple(str(name) for name in names)
    if action_names != POSE7D_ACTION_NAMES:
        raise ValueError(
            f"Replay dataset action schema must be {POSE7D_ACTION_NAMES}, got {action_names}."
        )
    return action_names


def compute_relative_pose_targets(actions: torch.Tensor) -> torch.Tensor:
    actions = torch.as_tensor(actions, dtype=torch.float32)
    if actions.ndim != 2 or actions.shape[-1] != len(POSE7D_ACTION_NAMES):
        raise ValueError(
            f"Expected pose7d action tensor with shape (T, {len(POSE7D_ACTION_NAMES)}), got {tuple(actions.shape)}."
        )
    if actions.shape[0] == 0:
        raise ValueError("Cannot build replay targets from an empty action sequence.")

    relative_actions = relative_pose10d(actions, actions[0])
    return pose10d_to_pose7d(relative_actions)


def apply_relative_pose_targets(relative_actions: torch.Tensor, base_pose: torch.Tensor) -> torch.Tensor:
    relative_actions = torch.as_tensor(relative_actions, dtype=torch.float32)
    base_pose = torch.as_tensor(base_pose, dtype=torch.float32)
    absolute_actions = absolute_pose10d(relative_actions, base_pose)
    return pose10d_to_pose7d(absolute_actions)


class PiperReplayDatasetProvider:
    def __init__(self, repo_id: str, root: str | Path | None = None):
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else None
        self._episodes: dict[str, _LoadedEpisode] = {}

    def open_episode(self, episode_index: int) -> EpisodeMeta:
        token = self._episode_token(episode_index)
        if token not in self._episodes:
            self._episodes[token] = self._load_episode(episode_index)
        return self._episodes[token].meta

    def get_episode_meta(self, episode_token: str) -> EpisodeMeta:
        return self._get_loaded_episode(episode_token).meta

    def get_relative_chunk(self, episode_token: str, frame_start: int, max_frames: int) -> RelativePoseChunk:
        if max_frames <= 0:
            raise ValueError(f"max_frames must be > 0, got {max_frames}.")

        episode = self._get_loaded_episode(episode_token)
        if frame_start < 0 or frame_start >= episode.meta.num_frames:
            raise IndexError(
                f"frame_start {frame_start} is out of bounds for episode with {episode.meta.num_frames} frames."
            )

        frame_end = min(frame_start + max_frames, episode.meta.num_frames)
        poses = episode.relative_actions[frame_start:frame_end].tolist()
        return RelativePoseChunk(episode_token=episode_token, frame_start=frame_start, poses=poses)

    def _get_loaded_episode(self, episode_token: str) -> _LoadedEpisode:
        episode = self._episodes.get(episode_token)
        if episode is None:
            raise KeyError(
                f"Episode token {episode_token!r} is not loaded. Call open_episode before requesting data."
            )
        return episode

    def _load_episode(self, episode_index: int) -> _LoadedEpisode:
        dataset = LeRobotDataset(self.repo_id, root=self.root, episodes=[episode_index])
        actions = dataset.select_columns(ACTION)
        action_names = _validate_pose7d_action_names(dataset.features[ACTION]["names"])
        action_tensor = torch.stack(
            [torch.as_tensor(actions[idx][ACTION], dtype=torch.float32) for idx in range(dataset.num_frames)],
            dim=0,
        )
        relative_actions = compute_relative_pose_targets(action_tensor)
        meta = EpisodeMeta(
            episode_token=self._episode_token(episode_index),
            episode_index=episode_index,
            num_frames=dataset.num_frames,
            fps=dataset.fps,
            action_names=action_names,
        )
        return _LoadedEpisode(meta=meta, relative_actions=relative_actions)

    @staticmethod
    def _episode_token(episode_index: int) -> str:
        return f"episode-{episode_index}"


def _read_exact(rfile, size: int) -> bytes:
    data = rfile.read(size)
    if data is None or len(data) != size:
        raise ConnectionError(f"Expected {size} bytes but received {0 if data is None else len(data)}.")
    return data


def _read_message(rfile) -> dict[str, Any]:
    size = int.from_bytes(_read_exact(rfile, 8), byteorder="big")
    payload = _read_exact(rfile, size)
    request = pickle.loads(payload)  # nosec
    if not isinstance(request, dict):
        raise TypeError(f"Expected dict request payload, got {type(request)}.")
    return request


def _write_message(wfile, payload: dict[str, Any]) -> None:
    data = pickle.dumps(payload)
    wfile.write(len(data).to_bytes(8, byteorder="big"))
    wfile.write(data)
    wfile.flush()


def build_replay_request_handler(provider: PiperReplayDatasetProvider) -> type[socketserver.StreamRequestHandler]:
    class PiperReplayRequestHandler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            try:
                request = _read_message(self.rfile)
                response = _dispatch_request(provider, request)
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            _write_message(self.wfile, response)

    return PiperReplayRequestHandler


def _dispatch_request(provider: PiperReplayDatasetProvider, request: dict[str, Any]) -> dict[str, Any]:
    method = request.get("method")
    if method == "open_episode":
        meta = provider.open_episode(int(request["episode_index"]))
        return {"ok": True, "meta": asdict(meta)}
    if method == "get_episode_meta":
        meta = provider.get_episode_meta(str(request["episode_token"]))
        return {"ok": True, "meta": asdict(meta)}
    if method == "get_relative_chunk":
        chunk = provider.get_relative_chunk(
            episode_token=str(request["episode_token"]),
            frame_start=int(request["frame_start"]),
            max_frames=int(request["max_frames"]),
        )
        return {"ok": True, "chunk": asdict(chunk)}
    raise ValueError(f"Unknown replay method: {method!r}.")


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class PiperReplayTransportClient:
    def __init__(self, host: str, port: int, timeout_s: float = 5.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

    def open_episode(self, episode_index: int) -> EpisodeMeta:
        response = self._request({"method": "open_episode", "episode_index": episode_index})
        return EpisodeMeta(**response["meta"])

    def get_episode_meta(self, episode_token: str) -> EpisodeMeta:
        response = self._request({"method": "get_episode_meta", "episode_token": episode_token})
        return EpisodeMeta(**response["meta"])

    def get_relative_chunk(self, episode_token: str, frame_start: int, max_frames: int) -> RelativePoseChunk:
        response = self._request(
            {
                "method": "get_relative_chunk",
                "episode_token": episode_token,
                "frame_start": frame_start,
                "max_frames": max_frames,
            }
        )
        return RelativePoseChunk(**response["chunk"])

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = pickle.dumps(payload)
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as sock:
            sock.sendall(len(data).to_bytes(8, byteorder="big"))
            sock.sendall(data)
            size = int.from_bytes(_recv_exact(sock, 8), byteorder="big")
            response = pickle.loads(_recv_exact(sock, size))  # nosec

        if not isinstance(response, dict):
            raise TypeError(f"Expected dict response payload, got {type(response)}.")
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "Unknown replay server error.")))
        return response


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"Socket closed before receiving {size} bytes.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class PiperReplayExecutor:
    def __init__(
        self,
        robot: Robot,
        transport: PiperReplayTransportClient,
        episode_token: str,
        chunk_size: int = 64,
        prefetch_threshold: int = 16,
        fps: int | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
        if prefetch_threshold < 0:
            raise ValueError(f"prefetch_threshold must be >= 0, got {prefetch_threshold}.")

        self.robot = robot
        self.transport = transport
        self.episode_token = episode_token
        self.chunk_size = chunk_size
        self.prefetch_threshold = prefetch_threshold
        self.fps = fps
        self._pose_adapter = PoseActPiperAdapter(robot)

    def run_episode(self) -> None:
        meta = self.transport.get_episode_meta(self.episode_token)
        fps = self.fps if self.fps is not None else meta.fps
        base_pose = self._pose_adapter.current_pose7d()

        next_frame_start = 0
        buffered_poses: deque[torch.Tensor] = deque()
        while next_frame_start < meta.num_frames and not buffered_poses:
            next_frame_start = self._append_next_chunk(meta, next_frame_start, buffered_poses)

        for frame_idx in range(meta.num_frames):
            start_t = time.perf_counter()
            if not buffered_poses:
                raise RuntimeError(f"Replay buffer exhausted before frame {frame_idx}.")

            relative_pose = buffered_poses.popleft()
            absolute_pose = apply_relative_pose_targets(relative_pose, base_pose)
            action = self._pose_adapter.convert(absolute_pose)
            _ = self.robot.send_action(action)

            if len(buffered_poses) <= self.prefetch_threshold and next_frame_start < meta.num_frames:
                next_frame_start = self._append_next_chunk(meta, next_frame_start, buffered_poses)

            precise_sleep(max(1.0 / fps - (time.perf_counter() - start_t), 0.0))

    def _append_next_chunk(
        self, meta: EpisodeMeta, next_frame_start: int, buffered_poses: deque[torch.Tensor]
    ) -> int:
        chunk = self.transport.get_relative_chunk(
            episode_token=meta.episode_token,
            frame_start=next_frame_start,
            max_frames=min(self.chunk_size, meta.num_frames - next_frame_start),
        )
        if not chunk.poses:
            raise RuntimeError(f"Replay server returned an empty chunk at frame {next_frame_start}.")

        for pose in chunk.poses:
            buffered_poses.append(torch.tensor(pose, dtype=torch.float32))

        return next_frame_start + len(chunk.poses)
