"""Token counting for OpenAI-style chat requests."""

from typing import Any


__all__ = ["TokenCountService"]


def __getattr__(name: str) -> Any:
    if name == "TokenCountService":
        from .service import TokenCountService

        return TokenCountService
    raise AttributeError(name)
