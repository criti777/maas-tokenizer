from pathlib import Path

import pytest

from maas_tokenizer.assets import AssetIntegrityError, verify_asset_directory
from maas_tokenizer.registry import ModelRegistry


def test_all_registered_assets_verify_offline() -> None:
    registry = ModelRegistry.from_file(Path("models/profiles.json"))
    for profile in registry.profiles:
        path = verify_asset_directory(profile, Path("model_assets"))
        assert path.is_dir()


def test_missing_asset_directory_is_rejected(tmp_path: Path) -> None:
    profile = ModelRegistry.from_file(Path("models/profiles.json")).resolve("glm-5.2")
    with pytest.raises(AssetIntegrityError):
        verify_asset_directory(profile, tmp_path)
