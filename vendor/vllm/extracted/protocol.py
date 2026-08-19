# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text-preprocessing subset of vLLM ChatCompletionRequest v0.26.0."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    messages: list[dict[str, Any]]
    model: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = "none"
    reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh", "max"
    ] | None = None
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    add_special_tokens: bool = False
    documents: list[dict[str, str]] | None = None
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    chat_template_content_format: Literal["auto", "string", "openai"] = "auto"

    @model_validator(mode="after")
    def validate_messages(self) -> "ChatCompletionRequest":
        for index, message in enumerate(self.messages):
            role = message.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError(f"messages[{index}].role is required")
        if self.add_generation_prompt and self.continue_final_message:
            raise ValueError(
                "add_generation_prompt and continue_final_message cannot both be true"
            )
        return self

    def template_kwargs(self, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        kwargs = dict(self.chat_template_kwargs or {})
        defaults = {
            "add_generation_prompt": self.add_generation_prompt,
            "continue_final_message": self.continue_final_message,
            "documents": self.documents,
            "reasoning_effort": self.reasoning_effort,
            "tools": tools,
        }
        for key, value in defaults.items():
            if key not in kwargs and value is not None:
                kwargs[key] = value
        if self.reasoning_effort is not None and "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = self.reasoning_effort != "none"
        return kwargs

