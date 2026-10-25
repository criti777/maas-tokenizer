import hashlib
import json
from pathlib import Path

import pytest

from maas_tokenizer.assets import verify_asset_directory
from maas_tokenizer.protocol import ChatCompletionRequest
from maas_tokenizer.registry import ModelRegistry
from maas_tokenizer.request_compat import normalize_compatibility_fields
from maas_tokenizer.renderers import build_renderer
from maas_tokenizer.service import TokenCountService


V4_PROFILES = [
    pytest.param("deepseek-v4-flash", marks=pytest.mark.model("deepseek-v4-flash")),
    pytest.param("deepseek-v4-pro", marks=pytest.mark.model("deepseek-v4-pro")),
]
OFFICIAL_CASES = json.loads(
    Path("tests/fixtures/deepseek_v4_official.json").read_text(encoding="utf-8")
)["cases"]


@pytest.fixture(scope="module")
def service() -> TokenCountService:
    return TokenCountService(assets_root=Path("model_assets"))


@pytest.fixture(scope="module", params=V4_PROFILES)
def v4_runtime(request):
    profile_id = request.param
    profile = ModelRegistry.from_file(Path("models/profiles.json")).resolve(profile_id)
    assets = verify_asset_directory(profile, Path("model_assets"))
    return profile_id, build_renderer(profile, assets)


@pytest.mark.parametrize("case", OFFICIAL_CASES, ids=lambda case: case["name"])
def test_v4_matches_independent_official_prompt_and_token_goldens(
    v4_runtime, service: TokenCountService, case: dict,
) -> None:
    profile_id, renderer = v4_runtime
    request = {"model": profile_id, **case["request"]}
    parsed = ChatCompletionRequest.model_validate(normalize_compatibility_fields(request))
    rendered = renderer.render(parsed)

    # Goldens were generated with unmodified official code, not our renderer.
    assert hashlib.sha256(rendered.text.encode()).hexdigest() == case["prompt_sha256"]
    assert service.count(request) == case["prompt_tokens"]
    assert list(renderer.encode(parsed)) == renderer.template_tokenizer.encode(
        rendered.text, add_special_tokens=False
    )


@pytest.mark.model("deepseek-v4.1-flash")
def test_deepseek_v41_flash_matches_official_text_encoder_count() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    assert service.count(
        {
            "model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": "你好, world 🌍"}],
        }
    ) == 9


@pytest.mark.model("deepseek-v4.1-flash")
def test_deepseek_v41_flash_counts_reasoning_effort_and_tools() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    assert service.count(
        {
            "model": "deepseek-v4.1-flash",
            "thinking": True,
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "prove it"}],
        }
    ) == 33
    assert service.count(
        {
            "model": "deepseek-v4.1-flash",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "messages": [{"role": "user", "content": "weather in Paris"}],
        }
    ) == 256


@pytest.mark.parametrize(
    "profile",
    [
        pytest.param("deepseek-v3.2", marks=pytest.mark.model("deepseek-v3.2")),
        *V4_PROFILES,
    ],
)
def test_specialized_model_matches_pinned_basic_count(
    service: TokenCountService, profile: str,
) -> None:
    assert service.count(
        {"model": profile, "messages": [{"role": "user", "content": "你好, world 🌍"}]}
    ) == 9


@pytest.mark.parametrize("profile", V4_PROFILES)
def test_v4_assistant_prefix_without_eos_matches_pinned_count(
    service: TokenCountService, profile: str,
) -> None:
    assert service.count(
        {
            "model": profile,
            "add_generation_prompt": False,
            "messages": [
                {"role": "user", "content": "draft"},
                {"role": "assistant", "content": "partial", "wo_eos": True},
            ],
        }
    ) == 7


@pytest.mark.parametrize(
    "profile",
    [
        pytest.param("deepseek-v3.2", marks=pytest.mark.model("deepseek-v3.2")),
        *V4_PROFILES,
    ],
)
@pytest.mark.parametrize("enabled", [True, False])
def test_deepseek_object_thinking_matches_boolean_form(
    service: TokenCountService, profile: str, enabled: bool,
) -> None:
    request = {
        "model": profile,
        "messages": [{"role": "user", "content": "分析后回答"}],
    }
    assert service.count(
        {**request, "thinking": {"type": "enabled" if enabled else "disabled"}}
    ) == service.count({**request, "thinking": enabled})


@pytest.mark.parametrize("profile", V4_PROFILES)
def test_v4_http_routes_and_counts_real_requests(
    profile: str, tmp_path, monkeypatch,
) -> None:
    from fastapi.testclient import TestClient
    from maas_tokenizer.api import app, get_token_count_service

    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(tmp_path / "access.log"))
    monkeypatch.setenv("TOKENIZER_RUN_LOG_PATH", str(tmp_path / "run.log"))
    service = TokenCountService(assets_root=Path("model_assets"))
    app.dependency_overrides[get_token_count_service] = lambda: service
    try:
        with TestClient(app) as client:
            for effort, expected in [("low", 7), ("high", 86), ("max", 99)]:
                response = client.post("/tokenizer", json={
                    "model": profile, "reasoning_effort": effort,
                    "messages": [{"role": "user", "content": "prove it"}],
                })
                assert response.status_code == 200
                assert response.json() == {"prompt_tokens": expected}
            for old_name in [
                "deepseek-v4", "DeepSeek-V4-Flash-0731", "DeepSeek-V4-Pro-0813",
                "deepseek-ai/DeepSeek-V4-Flash-0731",
                "deepseek-ai/DeepSeek-V4-Pro-0813",
            ]:
                response = client.post("/tokenizer", json={
                    "model": old_name,
                    "messages": [{"role": "user", "content": "hello"}],
                })
                assert response.status_code == 404
        assert service.cached_profiles == frozenset({profile})
    finally:
        app.dependency_overrides.pop(get_token_count_service, None)
