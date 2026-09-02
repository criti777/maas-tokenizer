"""Restore equivalent request fields removed by upstream adapters."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from .errors import RequestProcessingError


_REASONING_EFFORTS = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}


def normalize_compatibility_fields(
    request: Mapping[str, Any],
    *,
    minimal_disables_thinking: bool = False,
) -> dict[str, Any]:
    """Return a copied request with strict Thinking and continuation unions."""
    normalized = dict(request)
    _normalize_thinking(
        normalized,
        minimal_disables_thinking=minimal_disables_thinking,
    )
    _normalize_continuation(normalized)
    return normalized


def _normalize_thinking(
    request: dict[str, Any],
    *,
    minimal_disables_thinking: bool,
) -> None:
    values: list[bool] = []
    clear_thinking: bool | None = None
    if "thinking" in request:
        thinking, clear_thinking = _parse_thinking(request["thinking"])
        if thinking is not None:
            values.append(thinking)

    legacy_thinking = request.get("enable_thinking")
    if legacy_thinking is not None:
        values.append(_require_bool("enable_thinking", legacy_thinking))

    if "preserve_thinking" in request:
        _require_bool("preserve_thinking", request["preserve_thinking"])

    raw_kwargs = request.get("chat_template_kwargs")
    if raw_kwargs is None:
        kwargs: dict[str, Any] = {}
    elif isinstance(raw_kwargs, Mapping):
        kwargs = dict(raw_kwargs)
    else:
        raise RequestProcessingError("chat_template_kwargs must be an object")

    for key in ("thinking", "enable_thinking"):
        if key in kwargs:
            values.append(_require_bool(f"chat_template_kwargs.{key}", kwargs[key]))

    reasoning_effort = request.get("reasoning_effort")
    forced_disabled = minimal_disables_thinking and reasoning_effort in {
        "none",
        "minimal",
    }
    if reasoning_effort in _REASONING_EFFORTS and not forced_disabled:
        values.append(reasoning_effort != "none")

    if len(set(values)) > 1:
        raise RequestProcessingError("conflicting thinking options")
    if clear_thinking is not None:
        request["clear_thinking"] = clear_thinking
    if not values and not forced_disabled:
        return

    thinking = False if forced_disabled else values[0]
    request["thinking"] = thinking
    kwargs["thinking"] = thinking
    kwargs["enable_thinking"] = thinking
    request["chat_template_kwargs"] = kwargs


def _parse_thinking(value: Any) -> tuple[bool | None, bool | None]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return value, None
    if not isinstance(value, Mapping):
        raise RequestProcessingError("thinking must be a boolean, object, or null")

    type_value = value.get("type", "enabled")
    if not isinstance(type_value, str):
        raise RequestProcessingError(
            "thinking.type must be enabled, disabled, or auto"
        )
    if type_value == "auto":
        thinking = None
    elif type_value in {"enabled", "disabled"}:
        thinking = type_value == "enabled"
    else:
        raise RequestProcessingError(
            "thinking.type must be enabled, disabled, or auto"
        )

    clear_thinking = value.get("clear_thinking")
    if clear_thinking is not None:
        clear_thinking = _require_bool(
            "thinking.clear_thinking", clear_thinking
        )
    return thinking, clear_thinking


def _normalize_continuation(request: dict[str, Any]) -> None:
    for key in ("continue_final_message", "add_generation_prompt"):
        if key in request:
            _require_bool(key, request[key])

    messages = request.get("messages")
    if not isinstance(messages, list):
        return

    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or "prefix" not in message:
            continue
        prefix = _require_bool(f"messages[{index}].prefix", message["prefix"])
        if prefix and index != len(messages) - 1:
            raise RequestProcessingError("prefix is only valid on the final message")

    final = messages[-1] if messages else None
    final_prefix_present = isinstance(final, Mapping) and "prefix" in final
    final_prefix = bool(final["prefix"]) if final_prefix_present else False
    top_level_continuation = (
        request.get("continue_final_message") is True
        and request.get("add_generation_prompt") is False
    )

    if not final_prefix and not top_level_continuation:
        return
    if not isinstance(final, MutableMapping) or final.get("role") != "assistant":
        raise RequestProcessingError("continuation requires a final assistant message")

    if final_prefix:
        if request.get("continue_final_message") is False:
            raise RequestProcessingError("conflicting message continuation options")
        if request.get("add_generation_prompt") is True:
            raise RequestProcessingError("conflicting message continuation options")
    elif final_prefix_present:
        raise RequestProcessingError("conflicting message continuation options")

    request["continue_final_message"] = True
    request["add_generation_prompt"] = False
    if not final_prefix:
        copied_messages = list(messages)
        copied_final = dict(final)
        copied_final["prefix"] = True
        copied_messages[-1] = copied_final
        request["messages"] = copied_messages


def _require_bool(field: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise RequestProcessingError(f"{field} must be a boolean")
    return value
