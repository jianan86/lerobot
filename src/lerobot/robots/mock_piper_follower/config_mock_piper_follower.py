from dataclasses import dataclass, field

from ..config import RobotConfig


@dataclass
class MockCameraSpec:
    height: int = 480
    width: int = 640
    fps: int = 30


@RobotConfig.register_subclass("mock_piper_follower")
@dataclass
class MockPiperFollowerConfig(RobotConfig):
    """Mock of PiperFollower for async-inference framework testing without CAN/cameras."""

    fps: int = 30
    verbose: bool = True
    print_every_n: int = 30
    cameras: dict[str, MockCameraSpec] = field(
        default_factory=lambda: {"front": MockCameraSpec(height=480, width=640, fps=30)}
    )
