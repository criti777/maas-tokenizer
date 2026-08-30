"""Token-ID encoders used after model-specific prompt rendering."""

from __future__ import annotations

from collections.abc import Iterable, Sized
from pathlib import Path
from typing import Protocol

import gigatoken as gt


class TokenIds(Sized, Iterable[int], Protocol):
    """Minimal token-ID result needed by the counting service and tests."""


class TokenEncoder(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> TokenIds: ...


class GigaTokenEncoder:
    """Native Gigatoken encoder loaded entirely from pinned local assets."""

    def __init__(self, tokenizer: gt.Tokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_assets(cls, asset_path: Path) -> "GigaTokenEncoder":
        return cls(gt.Tokenizer(str(asset_path)))

    def encode(self, text: str, *, add_special_tokens: bool) -> TokenIds:
        # The string-rendered tokenizers have no encode-time post-processor, so
        # add_special_tokens=True and False produce the same IDs. Model tests
        # lock that invariant against each Hugging Face tokenizer.
        _ = add_special_tokens
        return self.tokenizer.encode(text)
