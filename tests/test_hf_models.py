from pathlib import Path

import pytest

from maas_tokenizer.service import TokenCountService


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        pytest.param("deepseek-v3", 8, marks=pytest.mark.model("deepseek-v3")),
        pytest.param("kimi-k2.6", 13, marks=pytest.mark.model("kimi-k2.6")),
        pytest.param("glm-5.1", 11, marks=pytest.mark.model("glm-5.1")),
        pytest.param("glm-5.2", 18, marks=pytest.mark.model("glm-5.2")),
        pytest.param("minimax-m2.7", 43, marks=pytest.mark.model("minimax-m2.7")),
    ],
)
def test_hf_model_matches_pinned_basic_count(profile: str, expected: int) -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    assert service.count(
        {"model": profile, "messages": [{"role": "user", "content": "你好, world 🌍"}]}
    ) == expected

