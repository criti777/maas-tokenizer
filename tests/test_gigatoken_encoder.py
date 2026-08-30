from pathlib import Path

import pytest

from maas_tokenizer.assets import verify_asset_directory
from maas_tokenizer.protocol import ChatCompletionRequest
from maas_tokenizer.registry import ModelRegistry
from maas_tokenizer.renderers import build_renderer


@pytest.mark.parametrize(
    "profile_id",
    [
        pytest.param("deepseek-v3", marks=pytest.mark.model("deepseek-v3")),
        pytest.param("deepseek-v3.2", marks=pytest.mark.model("deepseek-v3.2")),
        pytest.param("deepseek-v4", marks=pytest.mark.model("deepseek-v4")),
        pytest.param("kimi-k2.6", marks=pytest.mark.model("kimi-k2.6")),
        pytest.param("glm-5.1", marks=pytest.mark.model("glm-5.1")),
        pytest.param("glm-5.2", marks=pytest.mark.model("glm-5.2")),
        pytest.param("glm-5.3-flash", marks=pytest.mark.model("glm-5.3-flash")),
        pytest.param("minimax-m2.7", marks=pytest.mark.model("minimax-m2.7")),
        pytest.param("minimax-m3", marks=pytest.mark.model("minimax-m3")),
    ],
)
@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_native_gigatoken_matches_hf_for_rendered_prompt(
    profile_id: str, add_special_tokens: bool
) -> None:
    registry = ModelRegistry.from_file(Path("models/profiles.json"))
    profile = registry.resolve(profile_id)
    asset_path = verify_asset_directory(profile, Path("model_assets"))
    renderer = build_renderer(profile, asset_path)
    parsed = ChatCompletionRequest.model_validate(
        {
            "model": profile_id,
            "messages": [
                {"role": "system", "content": "回答要准确。"},
                {"role": "user", "content": "你好, world 🌍"},
            ],
            "add_special_tokens": add_special_tokens,
        }
    )

    rendered = renderer.render(parsed)
    expected = renderer.template_tokenizer.encode(
        rendered.text,
        add_special_tokens=rendered.add_special_tokens,
    )
    actual = renderer.encoder.encode(
        rendered.text,
        add_special_tokens=rendered.add_special_tokens,
    )

    assert list(actual) == list(expected)


def test_renderer_returns_text_before_encoding() -> None:
    registry = ModelRegistry.from_file(Path("models/profiles.json"))
    profile = registry.resolve("glm-5.2")
    asset_path = verify_asset_directory(profile, Path("model_assets"))
    renderer = build_renderer(profile, asset_path)
    parsed = ChatCompletionRequest.model_validate(
        {"model": "glm-5.2", "messages": [{"role": "user", "content": "你好"}]}
    )

    rendered = renderer.render(parsed)

    assert isinstance(rendered.text, str)
    assert rendered.text
    assert rendered.add_special_tokens is False
