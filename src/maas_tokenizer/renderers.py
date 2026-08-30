"""Model-specific prompt rendering without model weights."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any, Protocol

from transformers import AutoTokenizer

from vendor.vllm.extracted.chat_utils import parse_chat_messages
from vendor.vllm.extracted.deepseek_v32_encoding import encode_messages as encode_v32
from vendor.vllm.extracted.deepseek_v4_encoding import encode_messages as encode_v4
from vendor.vllm.extracted.hf_renderer import render_chat

from .encoders import GigaTokenEncoder, TokenEncoder, TokenIds
from .errors import RequestProcessingError
from .registry import ModelProfile


_REMOTE_CODE_LOCK = Lock()
_REMOTE_MODULE_CACHE = TemporaryDirectory(prefix="maas-tokenizer-")
_K3_THINKING_EFFORTS = {"low", "high", "max"}


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    add_special_tokens: bool


class Renderer(Protocol):
    template_tokenizer: Any
    encoder: TokenEncoder

    def encode(self, parsed: Any) -> TokenIds: ...


def _encode_rendered(renderer: Any, parsed: Any) -> TokenIds:
    rendered = renderer.render(parsed)
    return renderer.encoder.encode(
        rendered.text,
        add_special_tokens=rendered.add_special_tokens,
    )


def _load_tokenizer(path: Path, *, trust_remote_code: bool) -> Any:
    if not trust_remote_code:
        return AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )

    import transformers.dynamic_module_utils as dynamic_modules

    with _REMOTE_CODE_LOCK:
        original = dynamic_modules.HF_MODULES_CACHE
        dynamic_modules.HF_MODULES_CACHE = _REMOTE_MODULE_CACHE.name
        try:
            return AutoTokenizer.from_pretrained(
                path, local_files_only=True, trust_remote_code=True
            )
        finally:
            dynamic_modules.HF_MODULES_CACHE = original


class HFRenderer:
    def __init__(
        self,
        template_tokenizer: Any,
        encoder: TokenEncoder,
        content_format: str | None = None,
        template_thinking_mode: bool = False,
    ) -> None:
        self.template_tokenizer = template_tokenizer
        self.encoder = encoder
        self.content_format = content_format
        self.template_thinking_mode = template_thinking_mode

    @classmethod
    def from_assets(cls, path: Path, profile: ModelProfile) -> "HFRenderer":
        tokenizer = _load_tokenizer(
            path, trust_remote_code=profile.trust_remote_code
        )
        content_format = "openai" if profile.capabilities.get("content_parts") else None
        return cls(
            tokenizer,
            GigaTokenEncoder.from_assets(path),
            content_format,
            profile.template_thinking_mode,
        )

    def render(self, parsed: Any) -> RenderedPrompt:
        template_kwargs = parsed.template_kwargs(parsed.tools)
        if self.template_thinking_mode:
            thinking = template_kwargs.get(
                "thinking", template_kwargs.get("enable_thinking")
            )
            if isinstance(thinking, bool):
                template_kwargs["thinking_mode"] = (
                    "enabled" if thinking else "disabled"
                )
        _, rendered, _ = render_chat(
            tokenizer=self.template_tokenizer,
            messages=parsed.messages,
            tools=parsed.tools,
            chat_template=parsed.chat_template,
            content_format=self.content_format or parsed.chat_template_content_format,
            template_kwargs=template_kwargs,
        )
        return RenderedPrompt(rendered, parsed.add_special_tokens)

    def encode(self, parsed: Any) -> TokenIds:
        return _encode_rendered(self, parsed)


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

    def encode(self, parsed: Any) -> TokenIds:
        return _encode_rendered(self, parsed)


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

    def encode(self, parsed: Any) -> TokenIds:
        return _encode_rendered(self, parsed)


class KimiK3Renderer:
    """Encode K3's trusted XTML controls separately from untrusted text."""

    def __init__(self, template_tokenizer: Any, encoder: TokenEncoder) -> None:
        self.template_tokenizer = template_tokenizer
        self.encoder = encoder
        package = template_tokenizer.__class__.__module__.rsplit(".", 1)[0]
        self._build_chat_segments = import_module(
            f"{package}.encoding_k3"
        ).build_chat_segments
        self._special_tokens = tuple(template_tokenizer.special_tokens)

    @classmethod
    def from_assets(cls, path: Path, profile: ModelProfile) -> "KimiK3Renderer":
        tokenizer = _load_tokenizer(
            path, trust_remote_code=profile.trust_remote_code
        )
        return cls(tokenizer, GigaTokenEncoder.from_assets(path))

    def encode(self, parsed: Any) -> TokenIds:
        kwargs = parsed.template_kwargs(parsed.tools)
        enable_thinking = kwargs.pop("enable_thinking", None)
        if enable_thinking is not None:
            kwargs.setdefault("thinking", enable_thinking)

        reasoning_effort = kwargs.pop("reasoning_effort", None)
        if reasoning_effort == "none":
            kwargs.setdefault("thinking", False)
        elif reasoning_effort is not None:
            kwargs.setdefault("thinking_effort", reasoning_effort)

        thinking_effort = kwargs.setdefault("thinking_effort", "max")
        if thinking_effort not in _K3_THINKING_EFFORTS:
            raise RequestProcessingError(
                "Kimi K3 thinking_effort must be one of low, high, max"
            )
        thinking = kwargs.pop("thinking", True)
        segments = self._build_chat_segments(
            parsed.messages,
            thinking=thinking,
            **kwargs,
        )

        token_ids: list[int] = []
        for segment in segments:
            if segment.allow_special or not any(
                token in segment.text for token in self._special_tokens
            ):
                token_ids.extend(
                    self.encoder.encode(segment.text, add_special_tokens=False)
                )
            else:
                token_ids.extend(
                    self.template_tokenizer.encode(
                        segment.text,
                        allow_special_tokens=False,
                    )
                )
        return token_ids


def build_renderer(profile: ModelProfile, asset_path: Path) -> Renderer:
    if profile.renderer == "hf":
        return HFRenderer.from_assets(asset_path, profile)
    if profile.renderer == "kimi_k3":
        return KimiK3Renderer.from_assets(asset_path, profile)
    tokenizer = AutoTokenizer.from_pretrained(asset_path, local_files_only=True)
    encoder = GigaTokenEncoder.from_assets(asset_path)
    if profile.renderer == "deepseek_v32":
        return DeepSeekV32Renderer(tokenizer, encoder)
    if profile.renderer == "deepseek_v4":
        return DeepSeekV4Renderer(tokenizer, encoder)
    raise ValueError(f"unknown renderer: {profile.renderer}")
