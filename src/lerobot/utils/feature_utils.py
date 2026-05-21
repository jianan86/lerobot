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
"""Lightweight feature-manipulation utilities.

These functions are intentionally kept free of heavy dependencies (e.g. the
HuggingFace ``datasets`` library) so that they can be imported from anywhere
in the codebase – including modules that are part of the *minimal* install –
without triggering the ``lerobot.datasets`` package guard.
"""

from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig

from .constants import ACTION, DEFAULT_FEATURES, OBS_ENV_STATE, OBS_STATE, OBS_STR


def _validate_feature_names(features: dict[str, dict]) -> None:
    """Validate that feature names do not contain invalid characters.

    Args:
        features (dict): The LeRobot features dictionary.

    Raises:
        ValueError: If any feature name contains '/'.
    """
    invalid_features = {name: ft for name, ft in features.items() if "/" in name}
    if invalid_features:
        raise ValueError(f"Feature names should not contain '/'. Found '/' in '{invalid_features}'.")


def hw_to_dataset_features(
    hw_features: dict[str, type | tuple], prefix: str, use_video: bool = True
) -> dict[str, dict]:
    """Convert hardware-specific features to a LeRobot dataset feature dictionary.

    This function takes a dictionary describing hardware outputs (like joint states
    or camera image shapes) and formats it into the standard LeRobot feature
    specification.

    Args:
        hw_features (dict): Dictionary mapping feature names to their type (float for
            joints) or shape (tuple for images).
        prefix (str): The prefix to add to the feature keys (e.g., "observation"
            or "action").
        use_video (bool): If True, image features are marked as "video", otherwise "image".

    Returns:
        dict: A LeRobot features dictionary.
    """
    features = {}
    joint_fts = {
        key: ftype
        for key, ftype in hw_features.items()
        if ftype is float or (isinstance(ftype, PolicyFeature) and ftype.type != FeatureType.VISUAL)
    }
    cam_fts = {key: shape for key, shape in hw_features.items() if isinstance(shape, tuple)}

    if joint_fts and prefix == ACTION:
        features[prefix] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    if joint_fts and prefix == OBS_STR:
        features[f"{prefix}.state"] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    for key, shape in cam_fts.items():
        features[f"{prefix}.images.{key}"] = {
            "dtype": "video" if use_video else "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    _validate_feature_names(features)
    return features


def build_dataset_frame(
    ds_features: dict[str, dict], values: dict[str, Any], prefix: str
) -> dict[str, np.ndarray]:
    """Construct a single data frame from raw values based on dataset features.

    A "frame" is a dictionary containing all the data for a single timestep,
    formatted as numpy arrays according to the feature specification.

    Args:
        ds_features (dict): The LeRobot dataset features dictionary.
        values (dict): A dictionary of raw values from the hardware/environment.
        prefix (str): The prefix to filter features by (e.g., "observation"
            or "action").

    Returns:
        dict: A dictionary representing a single frame of data.
    """
    frame = {}
    for key, ft in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        elif ft["dtype"] in ["image", "video"]:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]

    return frame


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    """Convert dataset features to policy features.

    This function transforms the dataset's feature specification into a format
    that a policy can use, classifying features by type (e.g., visual, state,
    action) and ensuring correct shapes (e.g., channel-first for images).

    Args:
        features (dict): The LeRobot dataset features dictionary.

    Returns:
        dict: A dictionary mapping feature keys to `PolicyFeature` objects.

    Raises:
        ValueError: If an image feature does not have a 3D shape.
    """
    # TODO(aliberts): Implement "type" in dataset features and simplify this
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video", "depth_image", "depth_video"]:
            type = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")

            names = ft["names"]
            # Backward compatibility for "channel", introduced by mistake in some v2.0 datasets.
            if names is not None and (
                (len(names) > 2 and names[2] in ["channel", "channels"])
                or (ft["dtype"] in ["depth_image", "depth_video"] and names[0] in ["height", "width"])
            ):  # (h, w, c) -> (c, h, w)
                shape = (shape[2], shape[0], shape[1])
        elif key == OBS_ENV_STATE:
            type = FeatureType.ENV
        elif key.startswith(OBS_STR):
            type = FeatureType.STATE
        elif key.startswith(ACTION):
            type = FeatureType.ACTION
        else:
            continue

        policy_features[key] = PolicyFeature(
            type=type,
            shape=shape,
        )

    return policy_features


def infer_policy_feature_sets(
    cfg: PreTrainedConfig, features: dict[str, dict]
) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    """Infer policy input/output features from dataset features and config overrides."""
    policy_features = dataset_to_policy_features(features)
    output_features = {key: ft for key, ft in policy_features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in policy_features.items() if key not in output_features}

    if cfg.type in {"pose_act", "pose_smolvla"}:
        input_features = {
            key: ft
            for key, ft in input_features.items()
            if key == OBS_STATE or ft.type is FeatureType.VISUAL
        }
        if cfg.type == "pose_smolvla":
            input_features = {
                key: ft
                for key, ft in input_features.items()
                if key == OBS_STATE
                or (
                    ft.type is FeatureType.VISUAL
                    and ft.shape[0] == 3
                    and "depth" not in key.lower()
                    and "rgbd" not in key.lower()
                )
            }
        elif getattr(cfg, "use_rgbd_inputs", False) or getattr(cfg, "use_rgbd_v2_inputs", False):
            required = [
                cfg.fisheye_rgb_key,
                cfg.depth_camera_rgb_key,
                cfg.depth_key,
            ]
            missing = [key for key in required if key not in input_features]
            if missing:
                mode_name = "RGBD v2" if getattr(cfg, "use_rgbd_v2_inputs", False) else "RGBD"
                raise ValueError(
                    f"pose_act {mode_name} mode requires dataset feature(s): {missing}. "
                    f"Available features: {sorted(policy_features)}."
                )
            input_features = {
                key: input_features[key] for key in [OBS_STATE, *required] if key in input_features
            }

    image_feature_key = getattr(cfg, "image_feature_key", None)
    if image_feature_key is None:
        return input_features, output_features

    if image_feature_key not in policy_features:
        raise ValueError(
            f"Requested image feature {image_feature_key!r} is not present in the dataset features. "
            f"Available features: {sorted(policy_features)}."
        )
    if policy_features[image_feature_key].type is not FeatureType.VISUAL:
        raise ValueError(f"Requested image feature {image_feature_key!r} is not a visual feature.")
    if image_feature_key not in input_features:
        raise ValueError(
            f"Requested image feature {image_feature_key!r} is not supported by policy type {cfg.type!r}. "
            f"Available policy input features: {sorted(input_features)}."
        )

    input_features = {
        key: ft
        for key, ft in input_features.items()
        if ft.type is not FeatureType.VISUAL or key == image_feature_key
    }
    return input_features, output_features


def combine_feature_dicts(*dicts: dict) -> dict:
    """Merge LeRobot grouped feature dicts.

    - For 1D numeric specs (dtype not image/video/string) with "names": we merge the names and recompute the shape.
    - For others (e.g. `observation.images.*`), the last one wins (if they are identical).

    Args:
        *dicts: A variable number of LeRobot feature dictionaries to merge.

    Returns:
        dict: A single merged feature dictionary.

    Raises:
        ValueError: If there's a dtype mismatch for a feature being merged.
    """
    out: dict = {}
    for d in dicts:
        for key, value in d.items():
            if not isinstance(value, dict):
                out[key] = value
                continue

            dtype = value.get("dtype")
            shape = value.get("shape")
            is_vector = (
                dtype not in ("image", "video", "string")
                and isinstance(shape, tuple)
                and len(shape) == 1
                and "names" in value
            )

            if is_vector:
                # Initialize or retrieve the accumulating dict for this feature key
                target = out.setdefault(key, {"dtype": dtype, "names": [], "shape": (0,)})
                # Ensure consistent data types across merged entries
                if "dtype" in target and dtype != target["dtype"]:
                    raise ValueError(f"dtype mismatch for '{key}': {target['dtype']} vs {dtype}")

                # Merge feature names: append only new ones to preserve order without duplicates
                seen = set(target["names"])
                for n in value["names"]:
                    if n not in seen:
                        target["names"].append(n)
                        seen.add(n)
                # Recompute the shape to reflect the updated number of features
                target["shape"] = (len(target["names"]),)
            else:
                # For images/videos and non-1D entries: override with the latest definition
                out[key] = value
    return out
