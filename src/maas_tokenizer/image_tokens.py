"""Geometry-only Kimi image accounting; never opens media or allocates pixels.

Algorithm: official media_utils.navit_resize_image. Parameters are pinned to
K2.6 7eb5002f6aadc958aed6a9177b7ed26bb94011bb and
K3 a590ce090cb049c93a33dfe8c208ec652aa20503.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from .errors import RequestProcessingError


_BUDGETS = {"kimi-k2.6": 16384, "kimi-k3": 65536}
_IMAGES = {"image", "image_url"}
_OTHER_MEDIA = {"input_image", "video", "video_url", "audio", "audio_url", "input_audio"}


@dataclass(frozen=True)
class ImageGeometry:
    width: int
    height: int
    pad_width: int
    pad_height: int
    tokens: int


def image_geometry(width: int, height: int, profile_id: str) -> ImageGeometry:
    """Return official resized dimensions, padding and visual token count."""
    if any(type(x) is not int or x <= 0 for x in (width, height)):
        raise RequestProcessingError("multimodal_metadata shape must contain positive integers")
    try:
        budget = _BUDGETS[profile_id]
    except KeyError as error:
        raise RequestProcessingError("multimodal_metadata supports only Kimi K2.6/K3") from error
    try:
        patches = max(1.0, width // 14) * max(1.0, height // 14)
        scale = min(1.0, math.sqrt(budget / patches), 7168 / width, 7168 / height)
        w = min(7168, max(1, int(width * scale)))
        h = min(7168, max(1, int(height * scale)))
    except (OverflowError, ValueError) as error:
        raise RequestProcessingError("multimodal_metadata dimensions are too large") from error
    pw, ph = (-w) % 28, (-h) % 28
    return ImageGeometry(w, h, pw, ph, ((w + pw) // 28) * ((h + ph) // 28))


@dataclass(frozen=True)
class ImageAccounting:
    shapes: tuple[tuple[int, int], ...]
    adjustment: int

    def prepare(self, parsed: Any, profile_id: str) -> Any:
        if profile_id != "kimi-k3" or not self.shapes:
            return parsed
        kwargs = dict(parsed.chat_template_kwargs or {})
        kwargs["image_prompts"] = [
            f"<|media_begin|>image {w}x{h}<|media_content|><|media_pad|><|media_end|>"
            for w, h in self.shapes
        ]
        return parsed.model_copy(update={"chat_template_kwargs": kwargs})


def validate_image_metadata(metadata: Any, parsed: Any, profile_id: str) -> ImageAccounting:
    """Match metadata to structured image parts before any assets are loaded."""
    if metadata is None:
        return ImageAccounting((), 0)
    if not isinstance(metadata, list):
        raise RequestProcessingError("multimodal_metadata must be an array")
    image_count = 0
    for message in parsed.messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type")
            if kind in _OTHER_MEDIA:
                raise RequestProcessingError("multimodal_metadata supports image/image_url only")
            if kind in _IMAGES:
                if message.get("role") != "user":
                    raise RequestProcessingError(
                        "multimodal_metadata images must be in user messages"
                    )
                image_count += 1
    if len(metadata) != image_count:
        raise RequestProcessingError("multimodal_metadata count must match image parts")
    if not metadata:
        return ImageAccounting((), 0)
    if profile_id not in _BUDGETS:
        raise RequestProcessingError("multimodal_metadata supports only Kimi K2.6/K3")
    if parsed.chat_template is not None:
        raise RequestProcessingError("multimodal_metadata cannot be used with a custom chat_template")
    if "image_prompts" in (parsed.chat_template_kwargs or {}):
        raise RequestProcessingError("multimodal_metadata conflicts with image_prompts")
    if profile_id == "kimi-k3":
        # Official K3 also consumes inline placeholders in ordinary text. Mixed
        # inline/structured input would shift the ordered metadata association.
        pending = [parsed.messages, parsed.tools, parsed.chat_template_kwargs]
        while pending:
            value = pending.pop()
            if isinstance(value, str) and "<|kimi_image_placeholder|>" in value:
                raise RequestProcessingError(
                    "multimodal_metadata cannot accompany literal K3 image placeholders"
                )
            if isinstance(value, Mapping):
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
    shapes = []
    adjustment = 0
    for item in metadata:
        if not isinstance(item, Mapping) or item.get("media_type") != "image":
            raise RequestProcessingError("multimodal_metadata media_type must be image")
        shape = item.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise RequestProcessingError("multimodal_metadata shape must be [width, height]")
        geometry = image_geometry(shape[0], shape[1], profile_id)
        shapes.append((shape[0], shape[1]))
        # Each structural image emits exactly one media_pad in the official path.
        # Do not scan token IDs: literal user text can contain the same token.
        adjustment += geometry.tokens - 1
    return ImageAccounting(tuple(shapes), adjustment)
