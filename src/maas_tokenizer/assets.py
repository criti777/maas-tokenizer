"""Offline model text-asset integrity verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .registry import ModelProfile


PROJECT_ROOT = Path(__file__).parents[2]


class AssetIntegrityError(RuntimeError):
    pass


class AssetFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=1)


class AssetManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int = Field(ge=1)
    profile_id: str
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: dict[str, AssetFile]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset_directory(profile: ModelProfile, assets_root: Path) -> Path:
    manifest_path = PROJECT_ROOT / profile.asset_manifest
    asset_path = assets_root / profile.repository.replace("/", "--") / profile.revision
    if not manifest_path.is_file() or not asset_path.is_dir():
        raise AssetIntegrityError(f"assets unavailable for {profile.profile_id}")
    manifest = AssetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if (manifest.profile_id, manifest.repository, manifest.revision) != (
        profile.profile_id,
        profile.repository,
        profile.revision,
    ):
        raise AssetIntegrityError(f"asset manifest mismatch for {profile.profile_id}")
    tracked = set(manifest.files)
    for relative, expected in manifest.files.items():
        path = asset_path / relative
        if not path.is_file() or path.stat().st_size != expected.size:
            raise AssetIntegrityError(f"missing or invalid asset: {relative}")
        if _sha256(path) != expected.sha256:
            raise AssetIntegrityError(f"asset hash mismatch: {relative}")
    for path in asset_path.rglob("*.py"):
        relative = path.relative_to(asset_path).as_posix()
        if not relative.startswith(".cache/") and relative not in tracked:
            raise AssetIntegrityError(f"untracked Python asset: {relative}")
    return asset_path

