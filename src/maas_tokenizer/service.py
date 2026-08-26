"""Application service for prompt token counting."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import logging
from threading import Lock
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from vendor.vllm.extracted.chat_utils import UnsupportedMultimodalError

from .assets import verify_asset_directory
from .errors import ProcessorRequiredError, RequestProcessingError
from .protocol import ChatCompletionRequest
from .registry import ModelProfile, ModelRegistry
from .request_compat import normalize_compatibility_fields
from .renderers import Renderer, build_renderer


PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_ASSETS = PROJECT_ROOT / "model_assets"
DEFAULT_PROFILES = PROJECT_ROOT / "models" / "profiles.json"
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
_RUN_LOGGER = logging.getLogger("maas_tokenizer.run")


def _contains_media(messages: object) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, Mapping) and part.get("type") in _MEDIA_TYPES
            for part in content
        ):
            return True
    return False


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
        request_dict = normalize_compatibility_fields(request)
        model = request_dict.get("model")
        if not isinstance(model, str) or not model:
            raise RequestProcessingError("model is required")
        profile = self.registry.resolve(model)
        try:
            parsed = ChatCompletionRequest.model_validate(request_dict)
        except ValidationError as error:
            raise RequestProcessingError(str(error)) from error
        if _contains_media(parsed.messages) and not profile.capabilities.get(
            "content_parts", False
        ):
            raise ProcessorRequiredError(
                f"{profile.profile_id} requires a multimodal processor"
            )
        try:
            renderer = self._renderer_for(profile)
            rendered = renderer.render(parsed)
            token_ids = renderer.encoder.encode(
                rendered.text,
                add_special_tokens=rendered.add_special_tokens,
            )
        except UnsupportedMultimodalError as error:
            raise ProcessorRequiredError(str(error)) from error
        return len(token_ids)
