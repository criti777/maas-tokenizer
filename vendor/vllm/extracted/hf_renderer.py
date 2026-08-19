# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text-only extraction of vLLM's Hugging Face renderer v0.26.0."""

from __future__ import annotations

from typing import Any

from .chat_utils import parse_chat_messages
from .template_format import detect_content_format


def _supports_developer(chat_template: str) -> bool:
    return '"developer"' in chat_template or "'developer'" in chat_template


def _convert_developer(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in conversation:
        item = dict(message)
        if item["role"] == "developer":
            item["role"] = "system"
            item.pop("tools", None)
        converted.append(item)
    system: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for message in converted:
        (system if message["role"] == "system" else rest).append(message)
    if len(system) <= 1 and (not system or converted[0]["role"] == "system"):
        return converted
    contents = [message.get("content", "") for message in system]
    return [{"role": "system", "content": "\n\n".join(filter(None, contents))}, *rest]


def render_and_encode(
    *,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_template: str | None,
    content_format: str,
    template_kwargs: dict[str, Any],
    add_special_tokens: bool,
) -> tuple[list[dict[str, Any]], str, list[int], str]:
    resolved_template = tokenizer.get_chat_template(chat_template, tools=tools)
    resolved_format = (
        detect_content_format(resolved_template)
        if content_format == "auto"
        else content_format
    )
    conversation = parse_chat_messages(messages, resolved_format)
    if any(message["role"] == "developer" for message in conversation) and not _supports_developer(
        resolved_template
    ):
        conversation = _convert_developer(conversation)
    rendered = tokenizer.apply_chat_template(
        conversation=conversation,
        tools=tools,
        chat_template=resolved_template,
        tokenize=False,
        **{key: value for key, value in template_kwargs.items() if key != "tools"},
    )
    token_ids = tokenizer.encode(rendered, add_special_tokens=add_special_tokens)
    return conversation, rendered, list(token_ids), resolved_format

