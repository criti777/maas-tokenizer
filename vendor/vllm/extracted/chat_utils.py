# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text-only extraction of vLLM chat message parsing v0.26.0."""

from __future__ import annotations

import copy
import json
from typing import Any, Literal


ContentFormat = Literal["string", "openai"]
_TEXT_TYPES = {"text", "input_text", "output_text", "refusal", "thinking"}
_MEDIA_TYPES = {
    "image",
    "image_url",
    "input_image",
    "video",
    "video_url",
    "audio",
    "audio_url",
    "input_audio",
}


class UnsupportedMultimodalError(ValueError):
    pass


def _normalize_part(part: Any, content_format: ContentFormat) -> Any:
    if isinstance(part, str):
        return {"type": "text", "text": part} if content_format == "openai" else part
    if not isinstance(part, dict):
        raise ValueError("chat content parts must be strings or objects")
    part_type = part.get("type")
    if part_type in _TEXT_TYPES:
        text = part.get("text")
        if text is None and part_type in {"refusal", "thinking"}:
            text = part.get(part_type)
        if text is None:
            return None
        if not isinstance(text, str):
            raise ValueError(f"content part {part_type!r} must contain string text")
        if content_format == "string":
            return text
        result = {"type": "text", "text": text}
        result.update(
            {key: value for key, value in part.items() if key not in {"type", "text"}}
        )
        return result
    if part_type == "tool_reference":
        return copy.deepcopy(part) if content_format == "openai" else part.get("name", "")
    if part_type in _MEDIA_TYPES:
        if content_format == "openai":
            return copy.deepcopy(part)
        raise UnsupportedMultimodalError(
            f"content part {part_type!r} requires a multimodal processor"
        )
    raise ValueError(f"unsupported chat content part type: {part_type!r}")


def _normalize_content(content: Any, content_format: ContentFormat) -> Any:
    if content is None:
        parts: list[Any] = []
    elif isinstance(content, str):
        parts = [content]
    elif isinstance(content, list):
        parts = content
    else:
        raise ValueError("message content must be a string, array, or null")
    normalized = [
        normalized
        for part in parts
        if (normalized := _normalize_part(part, content_format)) is not None
    ]
    if content_format == "openai":
        return normalized
    return "\n".join(normalized)


def parse_chat_messages(
    messages: list[dict[str, Any]], content_format: ContentFormat
) -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = []
    for source in messages:
        message: dict[str, Any] = {
            "role": source["role"],
            "content": _normalize_content(source.get("content"), content_format),
        }
        for key in ("name", "task"):
            if isinstance(source.get(key), str):
                message[key] = source[key]
        for key in ("wo_eos", "prefix", "mask"):
            if key in source:
                message[key] = source[key]
        if isinstance(source.get("content_blocks"), list):
            message["content_blocks"] = copy.deepcopy(source["content_blocks"])
        if source["role"] == "assistant":
            if source.get("tool_calls") is not None:
                message["tool_calls"] = copy.deepcopy(source["tool_calls"])
            reasoning = source.get("reasoning", source.get("reasoning_content"))
            if reasoning is not None:
                message["reasoning"] = reasoning
                message["reasoning_content"] = reasoning
        elif source["role"] == "tool":
            if "tool_call_id" in source:
                message["tool_call_id"] = source["tool_call_id"]
            if isinstance(message["content"], list) and all(
                isinstance(item, dict) and item.get("type") == "text"
                for item in message["content"]
            ):
                message["content"] = "\n".join(
                    item.get("text", "") for item in message["content"]
                )
        elif source["role"] == "developer":
            message["tools"] = source.get("tools")
        conversation.append(message)
    _postprocess_messages(conversation)
    return conversation


def _postprocess_messages(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        if message["role"] != "assistant" or "tool_calls" not in message:
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        if not tool_calls:
            message.pop("tool_calls", None)
            continue
        for item in tool_calls:
            if not isinstance(item, dict):
                raise ValueError("assistant tool_calls entries must be objects")
            function = item.get("function")
            if item.get("type", "function") != "function" or not isinstance(
                function, dict
            ):
                raise ValueError("only assistant function tool_calls are supported")
            arguments = function.get("arguments")
            if arguments:
                if not isinstance(arguments, (dict, list)):
                    parsed = json.loads(arguments)
                    function["arguments"] = parsed if parsed is not None else {}
            else:
                function["arguments"] = {}
