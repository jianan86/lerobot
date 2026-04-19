"""This script demonstrates how to train PoseACT on a LeRobot dataset."""

from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pose_act import PoseACTConfig, PoseACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.constants import OBS_STATE


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def main():
    output_directory = Path("outputs/robot_learning_tutorial/pose_act")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    dataset_id = "lerobot/svla_so101_pickplace"

    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = PoseACTConfig(input_features=input_features, output_features=output_features, use_vae=False)
    policy = PoseACTPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train().to(device)

    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
        OBS_STATE: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    batch = next(iter(dataloader))
    batch = preprocessor(batch)
    loss, _ = policy.forward(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
