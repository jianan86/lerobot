import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.umi_pi05 import (
    UmiPI05Config,
    UmiPI05Policy,
    is_openpi_umi_pi05_checkpoint,
    load_openpi_umi_pi05_config,
    load_openpi_umi_pi05_stats,
    make_umi_pi05_pre_post_processors,
    materialize_openpi_umi_pi05_checkpoint,
)
from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


class _DummyTokenizer:
    padding_side = "right"


@pytest.fixture(autouse=True)
def mock_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: _DummyTokenizer(),
    )


def test_umi_pi05_config_defaults_for_prechunked_actions():
    config = UmiPI05Config(device="cpu")

    assert config.type == "umi_pi05"
    assert config.chunk_size == 10
    assert config.n_action_steps == 10
    assert config.action_delta_indices is None
    assert not config.use_relative_actions


def test_umi_pi05_rejects_relative_actions():
    with pytest.raises(ValueError, match="already relative"):
        UmiPI05Config(device="cpu", use_relative_actions=True)


def test_umi_pi05_factory_registration():
    assert isinstance(make_policy_config("umi_pi05", device="cpu"), UmiPI05Config)
    assert get_policy_class("umi_pi05") is UmiPI05Policy


def test_umi_pi05_processors_do_not_apply_relative_actions():
    config = UmiPI05Config(device="cpu")
    config.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(20,))}
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10, 20))}

    preprocessor, postprocessor = make_umi_pi05_pre_post_processors(config, dataset_stats=None)
    step_types = [type(step) for step in preprocessor.steps + postprocessor.steps]

    assert RelativeActionsProcessorStep not in step_types
    assert AbsoluteActionsProcessorStep not in step_types
    assert step_types.index(NormalizerProcessorStep) < len(preprocessor.steps)
    assert UnnormalizerProcessorStep in step_types


def test_umi_pi05_factory_uses_umi_processor_for_subclass():
    config = UmiPI05Config(device="cpu")
    config.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(20,))}
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10, 20))}

    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=None)
    step_types = [type(step) for step in preprocessor.steps + postprocessor.steps]

    assert RelativeActionsProcessorStep not in step_types
    assert AbsoluteActionsProcessorStep not in step_types


def test_umi_pi05_uses_last_action_shape_dim():
    config = UmiPI05Config(device="cpu")
    config.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(20,))}
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10, 20))}

    policy = object.__new__(UmiPI05Policy)
    policy.config = config

    assert policy._original_action_dim() == 20


def _write_openpi_checkpoint(tmp_path):
    checkpoint = tmp_path / "openpi"
    stats_dir = checkpoint / "assets" / "pika"
    stats_dir.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"stub")
    torch.save({"config": "stub"}, checkpoint / "metadata.pt")
    (stats_dir / "norm_stats.json").write_text(
        """
        {
          "state": {"p01": [0.0, 0.0], "p99": [1.0, 1.0]},
          "actions": {"q01": [-1.0, -1.0], "q99": [1.0, 1.0]}
        }
        """
    )
    return checkpoint


def test_openpi_umi_pi05_checkpoint_detection_and_config(tmp_path):
    checkpoint = _write_openpi_checkpoint(tmp_path)

    assert is_openpi_umi_pi05_checkpoint(checkpoint)
    config = load_openpi_umi_pi05_config(checkpoint, device="cpu")

    assert isinstance(config, UmiPI05Config)
    assert config.chunk_size == 10
    assert config.input_features[OBS_STATE].shape == (20,)
    assert config.output_features[ACTION].shape == (10, 20)
    assert set(config.image_features) == {
        f"{OBS_IMAGES}.cam_high",
        f"{OBS_IMAGES}.cam_left_wrist",
        f"{OBS_IMAGES}.cam_right_wrist",
    }


def test_openpi_umi_pi05_stats_mapping(tmp_path):
    checkpoint = _write_openpi_checkpoint(tmp_path)

    stats = load_openpi_umi_pi05_stats(checkpoint)

    assert set(stats) == {OBS_STATE, ACTION}
    torch.testing.assert_close(stats[OBS_STATE]["q01"], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(stats[OBS_STATE]["q99"], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(stats[ACTION]["q01"], torch.tensor([-1.0, -1.0]))


def test_materialize_openpi_umi_pi05_checkpoint_writes_lerobot_files(tmp_path):
    checkpoint = _write_openpi_checkpoint(tmp_path)

    written = materialize_openpi_umi_pi05_checkpoint(checkpoint, device="cpu")

    written_names = {path.name for path in written}
    assert "config.json" in written_names
    assert "policy_preprocessor.json" in written_names
    assert "policy_postprocessor.json" in written_names
    assert any(name.startswith("policy_preprocessor_step_") for name in written_names)
    assert any(name.startswith("policy_postprocessor_step_") for name in written_names)

    config = PreTrainedConfig.from_pretrained(checkpoint)
    assert config.input_features[OBS_STATE].shape == (20,)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"

