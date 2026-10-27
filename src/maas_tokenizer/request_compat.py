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
    """Adapt top-level request fields for the model renderers."""
    normalized = dict(request)
    # Client template kwargs are not an input to this service. Renderer kwargs
    # below are generated solely from supported top-level request fields.
    normalized.pop("chat_template_kwargs", None)
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
    raw_thinking = request.get("thinking")
    switch: bool | None = None
    if isinstance(raw_thinking, bool):
        switch = raw_thinking
    elif isinstance(raw_thinking, Mapping):
        if "clear_thinking" in raw_thinking:
            request["clear_thinking"] = raw_thinking["clear_thinking"]
        kind = raw_thinking.get("type", "enabled")
        if kind == "enabled":
            switch = True
        elif kind == "disabled":
            switch = False

    legacy = request.get("enable_thinking")
    if switch is None and isinstance(legacy, bool):
        switch = legacy

    effort = request.get("reasoning_effort")
    forced_disabled = (
        minimal_disables_thinking
        and isinstance(effort, str)
        and effort in {"none", "minimal"}
    )
    if forced_disabled:
        switch = False
    elif switch is None and isinstance(effort, str) and effort in _REASONING_EFFORTS:
        switch = effort != "none"
    elif switch is not None and isinstance(effort, str) and effort in _REASONING_EFFORTS:
        # An explicit switch wins over contradictory effort, including "none".
        # Do not let a specialized renderer reverse the decision later.
        if switch != (effort != "none"):
            request["reasoning_effort"] = None

    if switch is not None:
        request["thinking"] = switch
        request["chat_template_kwargs"] = {
            "thinking": switch,
            "enable_thinking": switch,
        }


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
