"""Model-specific prompt rendering without model weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any, Protocol

from transformers import AutoTokenizer

from vendor.vllm.extracted.chat_utils import parse_chat_messages
from vendor.vllm.extracted.deepseek_v32_encoding import encode_messages as encode_v32
from vendor.vllm.extracted.deepseek_v4_encoding import encode_messages as encode_v4
from vendor.vllm.extracted.hf_renderer import render_chat

from .encoders import GigaTokenEncoder, TokenEncoder
from .registry import ModelProfile


_REMOTE_CODE_LOCK = Lock()


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    add_special_tokens: bool


class Renderer(Protocol):
    template_tokenizer: Any
    encoder: TokenEncoder

    def render(self, parsed: Any) -> RenderedPrompt: ...


class HFRenderer:
    def __init__(
        self,
        template_tokenizer: Any,
        encoder: TokenEncoder,
        content_format: str | None = None,
    ) -> None:
        self.template_tokenizer = template_tokenizer
        self.encoder = encoder
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
        return cls(
            tokenizer,
            GigaTokenEncoder.from_assets(path),
            content_format,
        )

    def render(self, parsed: Any) -> RenderedPrompt:
        _, rendered, _ = render_chat(
            tokenizer=self.template_tokenizer,
            messages=parsed.messages,
            tools=parsed.tools,
            chat_template=parsed.chat_template,
            content_format=self.content_format or parsed.chat_template_content_format,
            template_kwargs=parsed.template_kwargs(parsed.tools),
        )
        return RenderedPrompt(rendered, parsed.add_special_tokens)


class DeepSeekV32Renderer:
    def __init__(self, template_tokenizer: Any, encoder: TokenEncoder) -> None:
        self.template_tokenizer = template_tokenizer
        self.encoder = encoder

    def render(self, parsed: Any) -> RenderedPrompt:
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
        return RenderedPrompt(text, False)


class DeepSeekV4Renderer:
    def __init__(self, template_tokenizer: Any, encoder: TokenEncoder) -> None:
        self.template_tokenizer = template_tokenizer
        self.encoder = encoder

    def render(self, parsed: Any) -> RenderedPrompt:
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
        return RenderedPrompt(text, False)


def build_renderer(profile: ModelProfile, asset_path: Path) -> Renderer:
    if profile.renderer == "hf":
        return HFRenderer.from_assets(asset_path, profile)
    tokenizer = AutoTokenizer.from_pretrained(asset_path, local_files_only=True)
    encoder = GigaTokenEncoder.from_assets(asset_path)
    if profile.renderer == "deepseek_v32":
        return DeepSeekV32Renderer(tokenizer, encoder)
    if profile.renderer == "deepseek_v4":
        return DeepSeekV4Renderer(tokenizer, encoder)
    raise ValueError(f"unknown renderer: {profile.renderer}")
