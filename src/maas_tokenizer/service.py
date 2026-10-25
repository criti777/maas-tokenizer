"""Application service for prompt token counting."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import logging
from threading import Lock
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from vendor.vllm.extracted.chat_utils import InvalidToolArgumentsError

from .assets import verify_asset_directory
from .errors import RequestProcessingError
from .protocol import ChatCompletionRequest
from .registry import ModelProfile, ModelRegistry
from .request_compat import normalize_compatibility_fields
from .renderers import Renderer, build_renderer


PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_ASSETS = PROJECT_ROOT / "model_assets"
DEFAULT_PROFILES = PROJECT_ROOT / "models" / "profiles.json"
_TEXT_CONTENT_TYPES = {
    "text",
    "input_text",
    "output_text",
    "refusal",
    "thinking",
    "tool_reference",
    "tool_result",
}
_DROP_PART = object()
_RUN_LOGGER = logging.getLogger("maas_tokenizer.run")


def _strip_multimodal_payload(value: Any) -> Any:
    """Remove media blocks recursively while preserving text and message metadata."""
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := _strip_multimodal_payload(item)) is not _DROP_PART
        ]
    if isinstance(value, Mapping):
        part_type = value.get("type")
        if isinstance(part_type, str):
            if part_type not in _TEXT_CONTENT_TYPES:
                return _DROP_PART
            if part_type != "tool_result":
                return dict(value)
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _strip_multimodal_payload(item)) is not _DROP_PART
        }
    return value


def _strip_multimodal_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    result = []
    for message in messages:
        if not isinstance(message, Mapping):
            result.append(message)
            continue
        cleaned = dict(message)
        for field in ("content", "content_blocks"):
            if field not in cleaned:
                continue
            value = _strip_multimodal_payload(cleaned[field])
            cleaned[field] = None if value is _DROP_PART else value
        result.append(cleaned)
    return result


def _apply_chat_template_defaults(
    request: dict[str, Any], defaults: Mapping[str, Any]
) -> None:
    if not defaults:
        return

    raw_kwargs = request.get("chat_template_kwargs")
    if raw_kwargs is None:
        template_kwargs: dict[str, Any] = {}
    elif isinstance(raw_kwargs, Mapping):
        template_kwargs = dict(raw_kwargs)
    else:
        raise RequestProcessingError("chat_template_kwargs must be an object")

    changed = False
    for key, value in defaults.items():
        if key in request or key in template_kwargs:
            continue
        template_kwargs[key] = value
        changed = True

    if changed:
        request["chat_template_kwargs"] = template_kwargs


class TokenCountService:
    """Count prompt token IDs after model-specific preprocessing."""

    def __init__(
        self,
        *,
        assets_root: Path = DEFAULT_ASSETS,
        registry_path: Path = DEFAULT_PROFILES,
    ) -> None:
        self.assets_root = assets_root
        self.registry = ModelRegistry.from_file(registry_path)
        self._renderers: dict[str, Renderer] = {}
        self._locks: dict[str, Lock] = {}
        self._locks_guard = Lock()

    @property
    def cached_profiles(self) -> frozenset[str]:
        return frozenset(self._renderers)

    def _lock_for(self, profile_id: str) -> Lock:
        with self._locks_guard:
            return self._locks.setdefault(profile_id, Lock())

    def _renderer_for(self, profile: ModelProfile) -> Renderer:
        renderer = self._renderers.get(profile.profile_id)
        if renderer is not None:
            return renderer
        with self._lock_for(profile.profile_id):
            renderer = self._renderers.get(profile.profile_id)
            if renderer is None:
                started_at = perf_counter()
                asset_path = verify_asset_directory(profile, self.assets_root)
                renderer = build_renderer(profile, asset_path)
                self._renderers[profile.profile_id] = renderer
                _RUN_LOGGER.info(
                    "event=model_loaded|profile_id=%s|duration_ms=%.2f",
                    profile.profile_id,
                    (perf_counter() - started_at) * 1000,
                )
            return renderer

    def count(self, request: Mapping[str, Any]) -> int:
        model = request.get("model")
        if not isinstance(model, str) or not model:
            raise RequestProcessingError("model is required")
        profile = self.registry.resolve(model)
        request_dict = normalize_compatibility_fields(
            request,
            minimal_disables_thinking=profile.minimal_disables_thinking,
        )
        request_dict["messages"] = _strip_multimodal_messages(
            request_dict.get("messages")
        )
        _apply_chat_template_defaults(
            request_dict,
            profile.chat_template_defaults,
        )
        try:
            parsed = ChatCompletionRequest.model_validate(request_dict)
        except ValidationError as error:
            raise RequestProcessingError(str(error)) from error
        try:
            renderer = self._renderer_for(profile)
            token_ids = renderer.encode(parsed)
        except InvalidToolArgumentsError as error:
            raise RequestProcessingError(str(error)) from error
        return len(token_ids)
