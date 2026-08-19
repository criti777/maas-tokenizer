from pathlib import Path

import pytest

from maas_tokenizer.errors import ProcessorRequiredError, RequestProcessingError
from maas_tokenizer.service import TokenCountService


def test_glm52_basic_request_has_expected_count() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    count = service.count(
        {
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "你好"}],
        }
    )
    assert count == 13


def test_invalid_message_is_rejected() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    with pytest.raises(RequestProcessingError):
        service.count({"model": "glm-5.2", "messages": [{"content": "missing role"}]})


def test_media_requiring_processor_is_rejected() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    with pytest.raises(ProcessorRequiredError):
        service.count(
            {
                "model": "glm-5.2",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": "https://invalid.example/x.png"}},
                        ],
                    }
                ],
            }
        )
