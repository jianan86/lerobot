#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_dataset_viz import _drop_depth_features_for_viz, visualize_dataset


class _FakeMetadata:
    def __init__(self):
        self.info = {
            "features": {
                "observation.images.front": {"dtype": "video"},
                "observation.depth.depth_camera": {"dtype": "depth_video"},
                "observation.depth.side": {"dtype": "depth_image"},
                "observation.state": {"dtype": "float32"},
            }
        }

    @property
    def features(self):
        return self.info["features"]

    @property
    def camera_keys(self):
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def depth_video_keys(self):
        return [key for key, ft in self.features.items() if ft["dtype"] == "depth_video"]

    @property
    def depth_image_keys(self):
        return [key for key, ft in self.features.items() if ft["dtype"] == "depth_image"]


class _FakeDataset:
    def __init__(self):
        self.meta = _FakeMetadata()


def test_drop_depth_features_for_viz_keeps_rgb_cameras():
    dataset = _FakeDataset()

    _drop_depth_features_for_viz(dataset)

    assert "observation.depth.depth_camera" not in dataset.meta.features
    assert "observation.depth.side" not in dataset.meta.features
    assert dataset.meta.camera_keys == ["observation.images.front"]


@pytest.mark.skip("TODO: add dummy videos")
def test_visualize_local_dataset(tmp_path, lerobot_dataset_factory):
    root = tmp_path / "dataset"
    output_dir = tmp_path / "outputs"
    dataset = lerobot_dataset_factory(root=root)
    rrd_path = visualize_dataset(
        dataset,
        episode_index=0,
        batch_size=32,
        save=True,
        output_dir=output_dir,
    )
    assert rrd_path.exists()
