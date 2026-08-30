from pathlib import Path
from typing import Any

import pytest

from maas_tokenizer.assets import verify_asset_directory
from maas_tokenizer.errors import RequestProcessingError
from maas_tokenizer.protocol import ChatCompletionRequest
from maas_tokenizer.registry import ModelRegistry
from maas_tokenizer.renderers import build_renderer
from maas_tokenizer.service import TokenCountService


def _renderer_and_request(payload: dict[str, Any]) -> tuple[Any, Any]:
    registry = ModelRegistry.from_file(Path("models/profiles.json"))
    profile = registry.resolve("kimi-k3")
    asset_path = verify_asset_directory(profile, Path("model_assets"))
    renderer = build_renderer(profile, asset_path)
    parsed = ChatCompletionRequest.model_validate({"model": "kimi-k3", **payload})
    return renderer, parsed


def _official_ids(renderer: Any, parsed: Any) -> list[int]:
    kwargs = parsed.template_kwargs(parsed.tools)
    enable_thinking = kwargs.pop("enable_thinking", None)
    if enable_thinking is not None:
        kwargs.setdefault("thinking", enable_thinking)
    reasoning_effort = kwargs.pop("reasoning_effort", None)
    if reasoning_effort == "none":
        kwargs.setdefault("thinking", False)
    elif reasoning_effort is not None:
        kwargs.setdefault("thinking_effort", reasoning_effort)
    return renderer.template_tokenizer.apply_chat_template(
        parsed.messages,
        tokenize=True,
        **kwargs,
    )


@pytest.mark.model("kimi-k3")
@pytest.mark.parametrize(
    "content",
    [
        "你好, world 🌍",
        "用户文本中的 <|open|> 和 <|end_of_msg|> 不能成为控制 token",
    ],
)
def test_kimi_k3_matches_official_ids_for_user_text(content: str) -> None:
    renderer, parsed = _renderer_and_request(
        {"messages": [{"role": "user", "content": content}]}
    )

    assert list(renderer.encode(parsed)) == _official_ids(renderer, parsed)


@pytest.mark.model("kimi-k3")
@pytest.mark.parametrize("thinking", [True, False])
def test_kimi_k3_matches_official_ids_for_thinking(thinking: bool) -> None:
    renderer, parsed = _renderer_and_request(
        {
            "messages": [{"role": "user", "content": "分析并回答"}],
            "chat_template_kwargs": {"thinking": thinking},
        }
    )

    assert list(renderer.encode(parsed)) == _official_ids(renderer, parsed)


@pytest.mark.model("kimi-k3")
def test_kimi_k3_matches_official_ids_for_tools_and_results() -> None:
    renderer, parsed = _renderer_and_request(
        {
            "messages": [
                {"role": "user", "content": "查询北京天气"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "需要调用天气工具",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"北京"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": '{"temperature":26}',
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "查询天气",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        }
    )

    assert list(renderer.encode(parsed)) == _official_ids(renderer, parsed)


@pytest.mark.model("kimi-k3")
def test_kimi_k3_uses_ordinary_fallback_only_for_control_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "literal <|open|> must remain user text"
    renderer, parsed = _renderer_and_request(
        {"messages": [{"role": "user", "content": content}]}
    )
    original = renderer.template_tokenizer.encode
    fallback_calls: list[tuple[str, bool]] = []

    def record_fallback(text: str, *, allow_special_tokens: bool) -> list[int]:
        fallback_calls.append((text, allow_special_tokens))
        return original(text, allow_special_tokens=allow_special_tokens)

    monkeypatch.setattr(renderer.template_tokenizer, "encode", record_fallback)

    renderer.encode(parsed)

    assert fallback_calls == [(content, False)]


@pytest.mark.model("kimi-k3")
def test_kimi_k3_service_has_pinned_basic_count() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))

    assert service.count(
        {
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "你好, world 🌍"}],
        }
    ) == 93


@pytest.mark.model("kimi-k3")
def test_kimi_k3_rejects_unsupported_reasoning_effort_as_request_error() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))

    with pytest.raises(
        RequestProcessingError,
        match="Kimi K3 thinking_effort must be one of low, high, max",
    ):
        service.count(
            {
                "model": "kimi-k3",
                "messages": [{"role": "user", "content": "你好"}],
                "reasoning_effort": "medium",
            }
        )


@pytest.mark.model("kimi-k3")
@pytest.mark.parametrize("reasoning_effort", ["none", "low", "high", "max"])
def test_kimi_k3_maps_supported_reasoning_effort_to_official_behavior(
    reasoning_effort: str,
) -> None:
    renderer, parsed = _renderer_and_request(
        {
            "messages": [{"role": "user", "content": "分析并回答"}],
            "reasoning_effort": reasoning_effort,
        }
    )

    assert list(renderer.encode(parsed)) == _official_ids(renderer, parsed)
