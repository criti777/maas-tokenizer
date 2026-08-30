from pathlib import Path

import pytest

from maas_tokenizer.service import TokenCountService


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        pytest.param("deepseek-v3.2", 9, marks=pytest.mark.model("deepseek-v3.2")),
        pytest.param("deepseek-v4", 9, marks=pytest.mark.model("deepseek-v4")),
    ],
)
def test_specialized_model_matches_pinned_basic_count(profile: str, expected: int) -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    assert service.count(
        {"model": profile, "messages": [{"role": "user", "content": "你好, world 🌍"}]}
    ) == expected


@pytest.mark.model("deepseek-v4")
@pytest.mark.parametrize(
    "alias",
    [
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "deepseek-ai/DeepSeek-V4-Pro-0813",
    ],
)
def test_deepseek_v4_new_variant_aliases_match_pinned_renderer(alias: str) -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    request = {"messages": [{"role": "user", "content": "你好, world 🌍"}]}

    assert service.count({"model": alias, **request}) == service.count(
        {"model": "deepseek-v4", **request}
    )


@pytest.mark.model("deepseek-v4")
def test_v4_xhigh_reasoning_effort_matches_pinned_count() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    assert service.count(
        {
            "model": "deepseek-v4",
            "reasoning_effort": "xhigh",
            "messages": [{"role": "user", "content": "prove it"}],
        }
    ) == 86


@pytest.mark.model("deepseek-v4")
def test_v4_assistant_prefix_without_eos_matches_pinned_count() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    assert service.count(
        {
            "model": "deepseek-v4",
            "add_generation_prompt": False,
            "messages": [
                {"role": "user", "content": "draft"},
                {"role": "assistant", "content": "partial", "wo_eos": True},
            ],
        }
    ) == 7
