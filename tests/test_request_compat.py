from copy import deepcopy

import pytest

from maas_tokenizer.errors import RequestProcessingError
from maas_tokenizer.request_compat import normalize_compatibility_fields


@pytest.mark.parametrize(
    "source",
    [
        {"thinking": True},
        {"chat_template_kwargs": {"thinking": True}},
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"reasoning_effort": "high"},
    ],
)
def test_thinking_source_fills_compatibility_union(source: dict) -> None:
    request = {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": {"custom": "kept"},
    }
    request.update(source)
    if "chat_template_kwargs" in source:
        request["chat_template_kwargs"] = {
            "custom": "kept",
            **source["chat_template_kwargs"],
        }
    original = deepcopy(request)

    normalized = normalize_compatibility_fields(request)

    assert normalized["thinking"] is True
    assert normalized["chat_template_kwargs"] == {
        "custom": "kept",
        "thinking": True,
        "enable_thinking": True,
    }
    assert request == original
    assert normalized is not request
    assert normalized["messages"] is not request["messages"]


def test_reasoning_effort_none_fills_disabled_thinking_union() -> None:
    normalized = normalize_compatibility_fields(
        {
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "none",
        }
    )

    assert normalized["thinking"] is False
    assert normalized["chat_template_kwargs"]["thinking"] is False
    assert normalized["chat_template_kwargs"]["enable_thinking"] is False
    assert normalized["reasoning_effort"] == "none"


def test_request_without_compatibility_fields_is_only_copied() -> None:
    request = {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "hello"}],
    }

    normalized = normalize_compatibility_fields(request)

    assert normalized == request
    assert normalized is not request
    assert normalized["messages"] is not request["messages"]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "thinking": True,
            "chat_template_kwargs": {"thinking": False},
        },
        {
            "chat_template_kwargs": {
                "thinking": True,
                "enable_thinking": False,
            },
        },
        {
            "thinking": True,
            "reasoning_effort": "none",
        },
        {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "high",
        },
    ],
)
def test_conflicting_thinking_options_are_rejected(payload: dict) -> None:
    payload.update({"model": "glm-5.2", "messages": []})

    with pytest.raises(RequestProcessingError, match="conflicting thinking options"):
        normalize_compatibility_fields(payload)


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"thinking": 1}, "thinking"),
        ({"chat_template_kwargs": {"thinking": "true"}}, "thinking"),
        (
            {"chat_template_kwargs": {"enable_thinking": None}},
            "enable_thinking",
        ),
    ],
)
def test_non_boolean_thinking_options_are_rejected(
    payload: dict, field: str
) -> None:
    payload.update({"model": "glm-5.2", "messages": []})

    with pytest.raises(RequestProcessingError, match=field):
        normalize_compatibility_fields(payload)


def test_non_object_chat_template_kwargs_are_rejected_when_normalizing() -> None:
    with pytest.raises(
        RequestProcessingError, match="chat_template_kwargs must be an object"
    ):
        normalize_compatibility_fields(
            {
                "model": "glm-5.2",
                "messages": [],
                "thinking": True,
                "chat_template_kwargs": "invalid",
            }
        )


def test_prefix_fills_top_level_continuation_fields() -> None:
    request = {
        "model": "deepseek-v3.2",
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "answer prefix",
                "prefix": True,
                "custom": "kept",
            },
        ],
    }
    original = deepcopy(request)

    normalized = normalize_compatibility_fields(request)

    assert normalized["continue_final_message"] is True
    assert normalized["add_generation_prompt"] is False
    assert normalized["messages"][-1]["prefix"] is True
    assert normalized["messages"][-1]["custom"] == "kept"
    assert request == original


def test_top_level_continuation_fields_fill_prefix() -> None:
    normalized = normalize_compatibility_fields(
        {
            "model": "glm-5.2",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer prefix"},
            ],
            "continue_final_message": True,
            "add_generation_prompt": False,
        }
    )

    assert normalized["continue_final_message"] is True
    assert normalized["add_generation_prompt"] is False
    assert normalized["messages"][-1]["prefix"] is True


def test_existing_continuation_union_remains_consistent() -> None:
    request = {
        "model": "glm-5.2",
        "messages": [
            {"role": "assistant", "content": "answer prefix", "prefix": True}
        ],
        "continue_final_message": True,
        "add_generation_prompt": False,
    }

    assert normalize_compatibility_fields(request) == request


def test_prefix_false_does_not_synthesize_top_level_fields() -> None:
    request = {
        "model": "glm-5.2",
        "messages": [
            {"role": "assistant", "content": "complete", "prefix": False}
        ],
    }

    assert normalize_compatibility_fields(request) == request


@pytest.mark.parametrize(
    "payload",
    [
        {
            "messages": [
                {"role": "assistant", "content": "prefix", "prefix": True}
            ],
            "continue_final_message": False,
        },
        {
            "messages": [
                {"role": "assistant", "content": "prefix", "prefix": True}
            ],
            "add_generation_prompt": True,
        },
        {
            "messages": [
                {"role": "assistant", "content": "prefix", "prefix": False}
            ],
            "continue_final_message": True,
            "add_generation_prompt": False,
        },
    ],
)
def test_conflicting_continuation_options_are_rejected(payload: dict) -> None:
    payload["model"] = "glm-5.2"

    with pytest.raises(
        RequestProcessingError, match="conflicting message continuation options"
    ):
        normalize_compatibility_fields(payload)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "question"}],
    ],
)
def test_top_level_continuation_requires_final_assistant(messages: list) -> None:
    with pytest.raises(
        RequestProcessingError,
        match="continuation requires a final assistant message",
    ):
        normalize_compatibility_fields(
            {
                "model": "glm-5.2",
                "messages": messages,
                "continue_final_message": True,
                "add_generation_prompt": False,
            }
        )


def test_prefix_true_is_only_allowed_on_final_message() -> None:
    with pytest.raises(
        RequestProcessingError, match="prefix is only valid on the final message"
    ):
        normalize_compatibility_fields(
            {
                "model": "glm-5.2",
                "messages": [
                    {"role": "assistant", "content": "old", "prefix": True},
                    {"role": "user", "content": "new"},
                ],
            }
        )


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        (
            {
                "messages": [
                    {"role": "assistant", "content": "prefix", "prefix": 1}
                ]
            },
            "prefix",
        ),
        (
            {
                "messages": [{"role": "assistant", "content": "prefix"}],
                "continue_final_message": "true",
            },
            "continue_final_message",
        ),
        (
            {
                "messages": [{"role": "assistant", "content": "prefix"}],
                "add_generation_prompt": 0,
            },
            "add_generation_prompt",
        ),
    ],
)
def test_non_boolean_continuation_options_are_rejected(
    payload: dict, field: str
) -> None:
    payload["model"] = "glm-5.2"

    with pytest.raises(RequestProcessingError, match=field):
        normalize_compatibility_fields(payload)
