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
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-v4.1-flash",
        "kimi-k2.6",
        "kimi-k3",
        "glm-5.1",
        "glm-5.2",
        "glm-5.3",
        "glm-5.3-flash",
        "minimax-m2.7",
        "minimax-m3",
    }


def test_registry_rejects_unknown_model() -> None:
    registry = ModelRegistry.from_file(PROFILES)
    with pytest.raises(UnknownModelError):
        registry.resolve("not-a-model")


@pytest.mark.parametrize("alias", [
    "deepseek-v4", "deepseekv4",
    "deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro",
    "DeepSeek-V4-Flash-0731", "DeepSeek-V4-Pro-0813",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "deepseek-ai/DeepSeek-V4-Pro-0813",
    " deepseek-v4-pro", "deepseek-v4-flash ",
])
def test_v4_accepts_only_public_profile_names(alias: str) -> None:
    registry = ModelRegistry.from_file(PROFILES)
    with pytest.raises(UnknownModelError):
        registry.resolve(alias)


@pytest.mark.parametrize(("name", "repository", "revision"), [
    ("deepseek-v4-flash", "deepseek-ai/DeepSeek-V4-Flash-0731",
     "7872f01b1d1fe23eabc4c98b48bffcef5a386062"),
    ("deepseek-v4-pro", "deepseek-ai/DeepSeek-V4-Pro-0813",
     "72e1d3230f6c080a530b0a1d46f8eb4602340597"),
])
def test_v4_routes_to_its_own_official_assets(
    name: str, repository: str, revision: str,
) -> None:
    from maas_tokenizer.assets import verify_asset_directory

    profile = ModelRegistry.from_file(PROFILES).resolve(name)
    assert profile.repository == repository
    assert profile.revision == revision
    assert verify_asset_directory(profile, Path("model_assets")) == (
        Path("model_assets") / repository.replace("/", "--") / revision
    )


def test_glm52_thinking_behavior_is_profile_driven() -> None:
    registry = ModelRegistry.from_file(PROFILES)

    glm52 = registry.resolve("glm-5.2")
    assert glm52.minimal_disables_thinking is True
    assert glm52.chat_template_defaults == {"clear_thinking": False}

    glm51 = registry.resolve("glm-5.1")
    assert glm51.minimal_disables_thinking is False
    assert glm51.chat_template_defaults == {}


@pytest.mark.parametrize(
    ("alias", "expected_profile"),
    [
        ("GLM-5.2", "glm-5.2"),
        ("DeepSeek-V4-Flash", "deepseek-v4-flash"),
        ("DEEPSEEK-V4-PRO", "deepseek-v4-pro"),
        ("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-v4.1-flash"),
        ("GLM-5.3-Flash", "glm-5.3-flash"),
        ("GLM-5.3", "glm-5.3"),
        ("zai-org/GLM-5.3", "glm-5.3"),
        ("zai-org/GLM-5.3-Flash", "glm-5.3-flash"),
        ("minimaxai/minimax-m2.7", "minimax-m2.7"),
        ("MiniMaxAI/MiniMax-M3", "minimax-m3"),
        ("moonshotai/Kimi-K3", "kimi-k3"),
    ],
)
def test_registry_resolves_model_aliases_case_insensitively(
    alias: str, expected_profile: str
) -> None:
    registry = ModelRegistry.from_file(PROFILES)
    assert registry.resolve(alias).profile_id == expected_profile
