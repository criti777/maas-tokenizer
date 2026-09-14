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


def count_image_metadata(metadata: Any, profile_id: str) -> int:
    """Sum image geometry tokens independently of messages and rendering."""
    if metadata is None:
        return 0
    if not isinstance(metadata, list):
        raise RequestProcessingError("multimodal_metadata must be an array")
    total = 0
    for item in metadata:
        if not isinstance(item, Mapping):
            raise RequestProcessingError("multimodal_metadata entries must be objects")
        if item.get("media_type") != "image":
            continue
        shape = item.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise RequestProcessingError("multimodal_metadata shape must be [width, height]")
        total += image_geometry(shape[0], shape[1], profile_id).tokens
    return total
