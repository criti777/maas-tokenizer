from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
import pytest

import maas_tokenizer.service as service_module
from maas_tokenizer.service import TokenCountService
class FixedRenderer:
    def encode(self, parsed: object) -> list[int]:
        return [1, 2, 3]


def test_concurrent_first_load_builds_renderer_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    calls_lock = Lock()

    def build(profile: object, path: Path, *, encoder: object) -> FixedRenderer:
        nonlocal calls
        with calls_lock:
            calls += 1
        return FixedRenderer()

    monkeypatch.setattr(service_module, "build_renderer", build)
    monkeypatch.setattr(service_module, "verify_asset_directory", lambda profile, root: root)
    service = TokenCountService(assets_root=Path("unused"))
    monkeypatch.setattr(service, "_encoder_for", lambda profile, path: object())
    request = {"model": "glm-5.2", "messages": [{"role": "user", "content": "hi"}]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(lambda _: service.count(request), range(16)))

    assert counts == [3] * 16
    assert calls == 1
    assert service.cached_profiles == frozenset({"glm-5.2"})


def test_failed_renderer_build_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def build(profile: object, path: Path, *, encoder: object) -> FixedRenderer:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first build fails")
        return FixedRenderer()

    monkeypatch.setattr(service_module, "build_renderer", build)
    monkeypatch.setattr(service_module, "verify_asset_directory", lambda profile, root: root)
    service = TokenCountService(assets_root=Path("unused"))
    monkeypatch.setattr(service, "_encoder_for", lambda profile, path: object())
    request = {"model": "glm-5.2", "messages": [{"role": "user", "content": "hi"}]}

    with pytest.raises(RuntimeError, match="first build fails"):
        service.count(request)
    assert service.count(request) == 3
    assert attempts == 2


def test_encoder_groups_share_only_the_requested_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    sources: list[tuple[Path, int]] = []

    def from_assets(path: Path, *, max_cache_bytes: int) -> object:
        sources.append((path, max_cache_bytes))
        return object()

    class RendererWithEncoder(FixedRenderer):
        def __init__(self, encoder: object) -> None:
            self.encoder = encoder

    monkeypatch.setattr(service_module.GigaTokenEncoder, "from_assets", from_assets)
    monkeypatch.setattr(
        service_module, "verify_asset_directory",
        lambda profile, root: root / profile.profile_id,
    )
    monkeypatch.setattr(
        service_module,
        "build_renderer",
        lambda profile, path, *, encoder: RendererWithEncoder(encoder),
    )
    service = TokenCountService(assets_root=Path("assets"))
    groups = (
        ("glm-5.3-flash", "glm-5.1", "glm-5.2", "glm-5.3"),
        ("deepseek-v3", "deepseek-v4-pro", "deepseek-v3.2",
         "deepseek-v4-flash", "deepseek-v4.1-flash"),
        ("kimi-k2.6",),
        ("kimi-k3",),
    )
    encoders = []
    for group in groups:
        for model in group:
            assert service.count({
                "model": model, "messages": [{"role": "user", "content": "hi"}]
            }) == 3
        group_encoders = {id(service._renderers[model].encoder) for model in group}
        assert len(group_encoders) == 1
        encoders.append(service._renderers[group[0]].encoder)

    assert len({id(encoder) for encoder in encoders}) == 4
    assert sources == [
        (Path("assets/glm-5.2"), 512 << 20),
        (Path("assets/deepseek-v4.1-flash"), 512 << 20),
        (Path("assets/kimi-k2.6"), 256 << 20),
        (Path("assets/kimi-k3"), 256 << 20),
    ]
