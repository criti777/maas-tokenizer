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


def test_top_level_thinking_matches_vllm_template_kwargs() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    messages = [{"role": "user", "content": "请分析这个问题"}]

    from_xds = service.count(
        {"model": "glm-5.2", "messages": messages, "thinking": False}
    )
    from_vllm = service.count(
        {
            "model": "glm-5.2",
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )

    assert from_xds == from_vllm


def test_prefix_matches_vllm_continuation_fields() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    messages = [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "答案是"},
    ]

    from_xds = service.count(
        {
            "model": "deepseek-v3.2",
            "messages": [messages[0], {**messages[1], "prefix": True}],
        }
    )
    from_vllm = service.count(
        {
            "model": "deepseek-v3.2",
            "messages": messages,
            "continue_final_message": True,
            "add_generation_prompt": False,
        }
    )

    assert from_xds == from_vllm
