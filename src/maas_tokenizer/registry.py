"""Immutable model profiles and strict alias resolution."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .errors import UnknownModelError


RendererName = Literal["hf", "deepseek_v32", "deepseek_v4"]


class ModelProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    repository: str = Field(pattern=r"^[^/]+/[^/]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    renderer: RendererName
    trust_remote_code: bool = False
    template_thinking_mode: bool = False
    asset_manifest: str = Field(min_length=1)
    capabilities: dict[str, bool]


class ModelRegistry:
    def __init__(self, profiles: Iterable[ModelProfile]) -> None:
        self._profiles = tuple(profiles)
        by_alias: dict[str, ModelProfile] = {}
        for profile in self._profiles:
            for alias in (profile.profile_id, profile.repository, *profile.aliases):
                normalized_alias = alias.casefold()
                if normalized_alias in by_alias:
                    raise ValueError(f"duplicate model alias: {alias}")
                by_alias[normalized_alias] = profile
        self._by_alias = by_alias

    @classmethod
    def from_file(cls, path: Path) -> "ModelRegistry":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(ModelProfile.model_validate(item) for item in payload)

    @property
    def profiles(self) -> tuple[ModelProfile, ...]:
        return self._profiles

    def resolve(self, alias: str) -> ModelProfile:
        try:
            return self._by_alias[alias.casefold()]
        except KeyError as error:
            raise UnknownModelError(f"unknown model alias: {alias}") from error
