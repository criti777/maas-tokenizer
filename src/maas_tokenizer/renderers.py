"""Model-specific prompt rendering without model weights."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any, Protocol

from transformers import AutoTokenizer

from vendor.vllm.extracted.chat_utils import parse_chat_messages
from vendor.vllm.extracted.deepseek_v32_encoding import encode_messages as encode_v32
from vendor.vllm.extracted.deepseek_v4_encoding import encode_messages as encode_v4
from vendor.vllm.extracted.hf_renderer import render_and_encode

from .registry import ModelProfile


_REMOTE_CODE_LOCK = Lock()


class Renderer(Protocol):
    def render(self, parsed: Any) -> list[int]: ...


class HFRenderer:
    def __init__(self, tokenizer: Any, content_format: str | None = None) -> None:
        self.tokenizer = tokenizer
        self.content_format = content_format

    @classmethod
    def from_assets(cls, path: Path, profile: ModelProfile) -> "HFRenderer":
        if profile.trust_remote_code:
            import transformers.dynamic_module_utils as dynamic_modules

            with _REMOTE_CODE_LOCK:
                original = dynamic_modules.HF_MODULES_CACHE
                with TemporaryDirectory(prefix="maas-tokenizer-") as cache:
                    dynamic_modules.HF_MODULES_CACHE = cache
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            path, local_files_only=True, trust_remote_code=True
                        )
                    finally:
                        dynamic_modules.HF_MODULES_CACHE = original
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )
        content_format = "openai" if profile.capabilities.get("content_parts") else None
        return cls(tokenizer, content_format)

    def render(self, parsed: Any) -> list[int]:
        _, _, token_ids, _ = render_and_encode(
            tokenizer=self.tokenizer,
            messages=parsed.messages,
            tools=parsed.tools,
            chat_template=parsed.chat_template,
            content_format=self.content_format or parsed.chat_template_content_format,
            template_kwargs=parsed.template_kwargs(parsed.tools),
            add_special_tokens=parsed.add_special_tokens,
        )
        return token_ids


class DeepSeekV32Renderer:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def render(self, parsed: Any) -> list[int]:
        conversation = parse_chat_messages(parsed.messages, "string")
        if parsed.tools:
            conversation.insert(0, {"role": "system", "content": "", "tools": parsed.tools})
        kwargs = parsed.template_kwargs(parsed.tools)
        thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
        text = encode_v32(
            conversation,
            thinking_mode="thinking" if thinking else "chat",
            drop_thinking=bool(conversation and conversation[-1]["role"] == "user"),
        )
        return list(self.tokenizer.encode(text, add_special_tokens=False))


class DeepSeekV4Renderer:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def render(self, parsed: Any) -> list[int]:
        conversation = parse_chat_messages(parsed.messages, "string")
        if parsed.tools:
            conversation.insert(0, {"role": "system", "content": "", "tools": parsed.tools})
        kwargs = parsed.template_kwargs(parsed.tools)
        thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
        effort = kwargs.get("reasoning_effort")
        if effort == "none":
            thinking, effort = False, None
        elif isinstance(effort, str):
            effort = "max" if effort in {"max", "xhigh"} else "high"
        else:
            effort = None
        text = encode_v4(
            conversation,
            thinking_mode="thinking" if thinking else "chat",
            drop_thinking=bool(kwargs.get("drop_thinking", True)),
            reasoning_effort=effort,
        )
        return list(self.tokenizer.encode(text, add_special_tokens=False))


def build_renderer(profile: ModelProfile, asset_path: Path) -> Renderer:
    if profile.renderer == "hf":
        return HFRenderer.from_assets(asset_path, profile)
    tokenizer = AutoTokenizer.from_pretrained(asset_path, local_files_only=True)
    if profile.renderer == "deepseek_v32":
        return DeepSeekV32Renderer(tokenizer)
    if profile.renderer == "deepseek_v4":
        return DeepSeekV4Renderer(tokenizer)
    raise ValueError(f"unknown renderer: {profile.renderer}")

