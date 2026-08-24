from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

import maas_tokenizer.service as service_module
from maas_tokenizer.service import TokenCountService
from maas_tokenizer.renderers import RenderedPrompt


class FixedEncoder:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == "rendered"
        assert add_special_tokens is False
        return [1, 2, 3]


class FixedRenderer:
    encoder = FixedEncoder()

    def render(self, parsed: object) -> RenderedPrompt:
        return RenderedPrompt("rendered", False)


def test_concurrent_first_load_builds_renderer_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    calls_lock = Lock()

    def build(profile: object, path: Path) -> FixedRenderer:
        nonlocal calls
        with calls_lock:
            calls += 1
        return FixedRenderer()

    monkeypatch.setattr(service_module, "build_renderer", build)
    monkeypatch.setattr(service_module, "verify_asset_directory", lambda profile, root: root)
    service = TokenCountService(assets_root=Path("unused"))
    request = {"model": "glm-5.2", "messages": [{"role": "user", "content": "hi"}]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(lambda _: service.count(request), range(16)))

    assert counts == [3] * 16
    assert calls == 1
    assert service.cached_profiles == frozenset({"glm-5.2"})


def test_failed_renderer_build_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def build(profile: object, path: Path) -> FixedRenderer:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first build fails")
        return FixedRenderer()

    monkeypatch.setattr(service_module, "build_renderer", build)
    monkeypatch.setattr(service_module, "verify_asset_directory", lambda profile, root: root)
    service = TokenCountService(assets_root=Path("unused"))
    request = {"model": "glm-5.2", "messages": [{"role": "user", "content": "hi"}]}

    with pytest.raises(RuntimeError, match="first build fails"):
        service.count(request)
    assert service.count(request) == 3
    assert attempts == 2
