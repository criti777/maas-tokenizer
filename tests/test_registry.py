from pathlib import Path

import pytest

from maas_tokenizer.errors import UnknownModelError
from maas_tokenizer.registry import ModelRegistry


PROFILES = Path("models/profiles.json")


def test_registry_contains_exact_supported_profiles() -> None:
    registry = ModelRegistry.from_file(PROFILES)
    assert {profile.profile_id for profile in registry.profiles} == {
        "deepseek-v3",
        "deepseek-v3.2",
        "deepseek-v4",
        "kimi-k2.6",
        "glm-5.1",
        "glm-5.2",
        "minimax-m2.7",
    }


def test_registry_rejects_unknown_model() -> None:
    registry = ModelRegistry.from_file(PROFILES)
    with pytest.raises(UnknownModelError):
        registry.resolve("not-a-model")

