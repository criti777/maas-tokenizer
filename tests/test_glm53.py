import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from maas_tokenizer.assets import verify_asset_directory
from maas_tokenizer.protocol import ChatCompletionRequest
from maas_tokenizer.registry import ModelRegistry
from maas_tokenizer.request_compat import normalize_compatibility_fields
from maas_tokenizer.renderers import build_renderer
from maas_tokenizer.service import TokenCountService


pytestmark = pytest.mark.model("glm-5.3")
FIXTURE = json.loads(Path("tests/fixtures/glm53_official.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def glm53_runtime():
    profile = ModelRegistry.from_file(Path("models/profiles.json")).resolve("glm-5.3")
    assets = verify_asset_directory(profile, Path("model_assets"))
    return TokenCountService(), build_renderer(profile, assets)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_glm53_matches_latest_official_template_and_token_goldens(glm53_runtime, case):
    service, renderer = glm53_runtime
    request = {"model": "glm-5.3", **case["request"]}
    parsed = ChatCompletionRequest.model_validate(normalize_compatibility_fields(request))
    rendered = renderer.render(parsed)

    assert hashlib.sha256(rendered.text.encode()).hexdigest() == case["prompt_sha256"]
    assert service.count(request) == case["prompt_tokens"]
    assert list(renderer.encode(parsed)) == renderer.template_tokenizer.encode(
        rendered.text, add_special_tokens=False
    )


def test_glm53_uses_separate_latest_assets_and_preserves_flash_profile():
    registry = ModelRegistry.from_file(Path("models/profiles.json"))
    profile = registry.resolve("glm-5.3")
    assert profile.repository == "zai-org/GLM-5.3"
    assert profile.revision == FIXTURE["provenance"]["revision"]
    assert profile.preserve_tool_result_content is True
    assets = verify_asset_directory(profile, Path("model_assets"))
    for filename, key in [
        ("chat_template.jinja", "template_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
    ]:
        assert hashlib.sha256((assets / filename).read_bytes()).hexdigest() == (
            FIXTURE["provenance"][key]
        )
    flash = registry.resolve("glm-5.3-flash")
    assert flash.repository == "zai-org/GLM-5.3-Flash"
    assert flash.revision == "04c4e9e95c5da8862dced7e5056455116f83a7e0"
    assert flash.preserve_tool_result_content is False


def test_glm53_http_routes_and_counts_real_requests(glm53_runtime, tmp_path, monkeypatch):
    from maas_tokenizer.api import app, get_token_count_service

    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(tmp_path / "access.log"))
    monkeypatch.setenv("TOKENIZER_RUN_LOG_PATH", str(tmp_path / "run.log"))
    service, _ = glm53_runtime
    app.dependency_overrides[get_token_count_service] = lambda: service
    try:
        with TestClient(app) as client:
            for alias in ["glm-5.3", "GLM-5.3", "zai-org/GLM-5.3"]:
                for case in [FIXTURE["cases"][0], next(
                    case for case in FIXTURE["cases"] if case["name"] == "grouped_outputs"
                )]:
                    response = client.post("/tokenizer", json={
                        "model": alias, **case["request"],
                    })
                    assert response.status_code == 200
                    assert response.json() == {"prompt_tokens": case["prompt_tokens"]}
    finally:
        app.dependency_overrides.pop(get_token_count_service, None)
