import pytest

from vendor.vllm.extracted.chat_utils import (
    UnsupportedMultimodalError,
    parse_chat_messages,
)
from vendor.vllm.extracted.deepseek_v32_encoding import (
    encode_messages as encode_v32,
    render_message as render_v32,
)
from vendor.vllm.extracted.deepseek_v4_encoding import (
    encode_messages as encode_v4,
    merge_tool_messages,
    render_message as render_v4,
    sort_tool_results_by_call_order,
)


def test_chat_normalization_preserves_supported_extensions() -> None:
    messages = parse_chat_messages(
        [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "r"}, "answer"],
                "reasoning_content": "reason",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"x":1}'},
                    }
                ],
                "prefix": True,
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [{"type": "text", "text": "result"}],
            },
            {"role": "developer", "content": "rules", "tools": []},
        ],
        "openai",
    )
    assert messages[0]["reasoning"] == "reason"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert messages[1]["content"] == "result"
    assert messages[2]["tools"] == []


@pytest.mark.parametrize(
    "content",
    [123, [{"type": "unknown"}], [{"type": "text", "text": 1}], [object()]],
)
def test_chat_normalization_rejects_invalid_content(content: object) -> None:
    with pytest.raises(ValueError):
        parse_chat_messages([{"role": "user", "content": content}], "string")


def test_string_content_rejects_media() -> None:
    with pytest.raises(UnsupportedMultimodalError):
        parse_chat_messages(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
            "string",
        )


def test_v32_roles_tools_and_thinking_paths() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    messages = [
        {"role": "system", "content": "system", "tools": tools},
        {"role": "developer", "content": "rules"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": None,
            "reasoning": "think",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"q": "x", "n": 1}},
                }
            ],
        },
        {"role": "tool", "content": "result"},
    ]
    prompt = encode_v32(messages, thinking_mode="thinking", drop_thinking=False)
    assert "## Tools" in prompt
    assert "function_calls" in prompt
    assert "<function_results>" in prompt
    with pytest.raises(ValueError):
        render_v32(-1, messages, "chat")
    with pytest.raises(ValueError):
        render_v32(0, messages, "invalid")
    with pytest.raises(NotImplementedError):
        render_v32(0, [{"role": "alien", "content": "x"}], "chat")


def test_v4_merges_and_orders_tool_results() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "a", "arguments": {}}},
                {"id": "b", "type": "function", "function": {"name": "b", "arguments": {}}},
            ],
        },
        {"role": "tool", "tool_call_id": "b", "content": "second"},
        {"role": "tool", "tool_call_id": "a", "content": "first"},
        {"role": "user", "content": "continue", "task": "query"},
    ]
    merged = sort_tool_results_by_call_order(merge_tool_messages(messages))
    blocks = merged[1]["content_blocks"]
    assert [block.get("content") for block in blocks[:2]] == ["first", "second"]
    prompt = encode_v4(messages, thinking_mode="chat")
    assert prompt.index("first") < prompt.index("second")
    assert "<｜query｜>" in render_v4(
        0, [{"role": "user", "content": "continue", "task": "query"}], "chat"
    )


def test_v4_developer_reminder_content_blocks_and_error_paths() -> None:
    messages = [
        {"role": "developer", "content": "rules"},
        {
            "role": "user",
            "content_blocks": [
                {"type": "text", "text": "text"},
                {"type": "tool_result", "content": [{"type": "text", "text": "ok"}]},
                {"type": "other"},
            ],
        },
        {"role": "latest_reminder", "content": "remember"},
        {"role": "assistant", "content": "done", "reasoning": "why", "wo_eos": True},
    ]
    prompt = encode_v4(
        messages,
        thinking_mode="thinking",
        drop_thinking=False,
        reasoning_effort="max",
    )
    assert "Reasoning Effort" in prompt
    assert "<｜latest_reminder｜>" in prompt
    block_prompt = render_v4(
        0,
        [
            {
                "role": "user",
                "content_blocks": [
                    {"type": "text", "text": "text"},
                    {"type": "tool_result", "content": [{"type": "text", "text": "ok"}]},
                    {"type": "other"},
                ],
            }
        ],
        "chat",
    )
    assert "[Unsupported other]" in block_prompt
    with pytest.raises(NotImplementedError):
        render_v4(0, [{"role": "tool", "content": "x"}], "chat")
    with pytest.raises(NotImplementedError):
        render_v4(0, [{"role": "alien", "content": "x"}], "chat")
    with pytest.raises(AssertionError):
        render_v4(0, [{"role": "user", "content": "x"}], "bad")
