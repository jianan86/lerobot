import pytest

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.umi_pi05 import UmiPI05Config, UmiPI05Policy, make_umi_pi05_pre_post_processors
from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.utils.constants import ACTION


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
