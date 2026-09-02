from pathlib import Path

import pytest

from maas_tokenizer.errors import ProcessorRequiredError, RequestProcessingError
from maas_tokenizer.protocol import ChatCompletionRequest
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


def test_template_kwargs_include_thinking_auxiliary_fields() -> None:
    parsed = ChatCompletionRequest.model_validate(
        {
            "messages": [],
            "clear_thinking": True,
            "preserve_thinking": True,
        }
    )

    kwargs = parsed.template_kwargs(None)

    assert kwargs["clear_thinking"] is True
    assert kwargs["preserve_thinking"] is True


def test_template_kwargs_omit_unspecified_thinking_auxiliary_fields() -> None:
    parsed = ChatCompletionRequest.model_validate({"messages": []})

    kwargs = parsed.template_kwargs(None)

    assert "clear_thinking" not in kwargs
    assert "preserve_thinking" not in kwargs


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


def test_service_delegates_complete_encoding_to_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EncodeOnlyRenderer:
        def encode(self, parsed: object) -> list[int]:
            assert getattr(parsed, "model") == "glm-5.2"
            return [1, 2, 3, 4]

    service = TokenCountService(assets_root=Path("model_assets"))
    monkeypatch.setattr(
        service,
        "_renderer_for",
        lambda _profile: EncodeOnlyRenderer(),
    )

    assert service.count(
        {"model": "glm-5.2", "messages": [{"role": "user", "content": "你好"}]}
    ) == 4
