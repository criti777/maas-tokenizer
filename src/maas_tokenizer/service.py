"""Application service for prompt token counting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class TokenCountService:
    """Count prompt token IDs after model-specific preprocessing."""

    def count(self, request: Mapping[str, Any]) -> int:
        raise NotImplementedError

