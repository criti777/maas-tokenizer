from pathlib import Path

import pytest

from maas_tokenizer.errors import ProcessorRequiredError
from maas_tokenizer.service import TokenCountService


@pytest.mark.model("kimi-k2.6")
def test_kimi_counts_official_media_placeholder_without_network() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    assert service.count(
        {
            "model": "kimi-k2.6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image_url", "image_url": {"url": "https://invalid.example/x.png"}},
                        {"type": "text", "text": "after"},
                    ],
                }
            ],
        }
    ) == 16


@pytest.mark.parametrize(
    "profile",
    [
        pytest.param("glm-5.2", marks=pytest.mark.model("glm-5.2")),
        pytest.param("deepseek-v3.2", marks=pytest.mark.model("deepseek-v3.2")),
        pytest.param("deepseek-v4", marks=pytest.mark.model("deepseek-v4")),
    ],
)
def test_processor_dependent_media_is_rejected(profile: str) -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    with pytest.raises(ProcessorRequiredError):
        service.count(
            {
                "model": profile,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                        ],
                    }
                ],
            }
        )
