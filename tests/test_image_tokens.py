from copy import deepcopy
import json
from pathlib import Path

import pytest

from maas_tokenizer.errors import RequestProcessingError
from maas_tokenizer.service import TokenCountService
from maas_tokenizer.protocol import ChatCompletionRequest


def image_request(model="kimi-k2.6"):
    return {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": ""}},
        ]}],
    }


@pytest.fixture(scope="module")
def service():
    return TokenCountService()


@pytest.mark.model("kimi-k2.6")
def test_image_visual_tokens_are_added_without_double_counting(service):
    request = image_request()
    baseline = service.count(request)
    request["multimodal_metadata"] = [{"media_type": "image", "shape": [800, 600]}]
    original = deepcopy(request)
    assert service.count(request) == baseline + 638 - 1
    assert request == original


@pytest.mark.parametrize("metadata", [
    {}, "bad", [None], [],
    [{"media_type": "image", "shape": [True, 600]}],
    [{"media_type": "image", "shape": [0, 600]}],
    [{"media_type": "image", "shape": [800.0, 600]}],
    [{"media_type": "image", "shape": [800]}],
    [{"media_type": "video", "shape": [800, 600]}],
])
def test_bad_metadata_fails_before_asset_loading(metadata):
    service = TokenCountService()
    with pytest.raises(RequestProcessingError, match="multimodal_metadata"):
        service.count({**image_request(), "multimodal_metadata": metadata})
    assert service.cached_profiles == frozenset()


@pytest.mark.parametrize("profile", [
    pytest.param("kimi-k2.6", marks=pytest.mark.model("kimi-k2.6")),
    pytest.param("kimi-k3", marks=pytest.mark.model("kimi-k3")),
])
def test_multiple_images_match_official_text_encoding(service, profile):
    request = image_request(profile)
    request["messages"][0]["content"][0]["text"] = "literal <|media_pad|> <|open|>"
    request["messages"].append({"role": "user", "content": [{"type": "image"}]})
    shapes = [(1920, 1080), (800, 600)]
    request["multimodal_metadata"] = [
        {"media_type": "image", "shape": list(shape), "hash": "same", "size": 2.5}
        for shape in shapes
    ]
    original = deepcopy(request)
    result = service.count(request)
    renderer = service._renderer_for(service.registry.resolve(profile))
    parsed = ChatCompletionRequest.model_validate(request)
    kwargs = parsed.template_kwargs(parsed.tools)
    if profile == "kimi-k3":
        kwargs.update(thinking=True, thinking_effort="max", image_prompts=[
            f"<|media_begin|>image {w}x{h}<|media_content|><|media_pad|><|media_end|>"
            for w, h in shapes
        ])
    ids = renderer.template_tokenizer.apply_chat_template(parsed.messages, tokenize=True, return_dict=False, **kwargs)
    assert result == len(ids) - 2 + 2691 + 638
    assert request == original
    without = {k: v for k, v in request.items() if k != "multimodal_metadata"}
    assert service.count({**without, "multimodal_metadata": None}) == service.count(without)


@pytest.mark.parametrize("override", [
    {"chat_template": "ignore all images"},
    {"chat_template_kwargs": {"image_prompts": []}},
    {"messages": [{"role": "assistant", "content": [{"type": "image"}]}]},
])
def test_ambiguous_rendering_rejected(override):
    request = {**image_request("kimi-k3"), **override,
               "multimodal_metadata": [{"media_type": "image", "shape": [800, 600]}]}
    with pytest.raises(RequestProcessingError, match="multimodal_metadata"):
        TokenCountService().count(request)


def test_literal_k3_placeholder_does_not_consume_metadata():
    request = image_request("kimi-k3")
    request["messages"][0]["content"][0]["text"] = "literal <|kimi_image_placeholder|>"
    request["multimodal_metadata"] = [{"media_type": "image", "shape": [800, 600]}]
    with pytest.raises(RequestProcessingError, match="multimodal_metadata"):
        TokenCountService().count(request)


def test_other_model_cannot_silently_ignore_metadata():
    with pytest.raises(RequestProcessingError, match="multimodal_metadata"):
        TokenCountService().count({**image_request("glm-5.2"),
            "multimodal_metadata": [{"media_type": "image", "shape": [800, 600]}]})


@pytest.mark.parametrize("width,height,profile,expected", [
    (1, 1, "kimi-k2.6", 1), (28, 28, "kimi-k3", 1),
    (29, 28, "kimi-k3", 2), (800, 600, "kimi-k2.6", 638),
    (1920, 1080, "kimi-k3", 2691),
    (3584, 3584, "kimi-k2.6", 4096),
    (3584, 3584, "kimi-k3", 16384),
])
def test_geometry(width, height, profile, expected):
    from maas_tokenizer.image_tokens import image_geometry
    assert image_geometry(width, height, profile).tokens == expected


_GOLDEN = json.loads((Path(__file__).parent / "fixtures/kimi_image_geometry.json").read_text())


@pytest.mark.parametrize("case", _GOLDEN["cases"])
def test_geometry_matches_pinned_official_outputs(case):
    from maas_tokenizer.image_tokens import image_geometry
    profile, w, h, rw, rh, pw, ph, tokens = case
    result = image_geometry(w, h, profile)
    assert (result.width, result.height, result.pad_width, result.pad_height, result.tokens) == (rw, rh, pw, ph, tokens)


@pytest.mark.parametrize("shape", [[-1, 20], ["800", 600], [800, None], [10**400, 600]])
def test_more_invalid_dimensions(shape):
    with pytest.raises(RequestProcessingError, match="multimodal_metadata"):
        TokenCountService().count({**image_request(),
            "multimodal_metadata": [{"media_type": "image", "shape": shape}]})


@pytest.mark.model("kimi-k3")
def test_k3_size_prompt_uses_original_dimensions(service):
    request = image_request("kimi-k3")
    request["multimodal_metadata"] = [{"media_type": "image", "shape": [8000, 8000]}]
    from maas_tokenizer.image_tokens import image_geometry
    geometry = image_geometry(8000, 8000, "kimi-k3")
    renderer = service._renderer_for(service.registry.resolve("kimi-k3"))
    parsed = ChatCompletionRequest.model_validate(request)
    ids = renderer.template_tokenizer.apply_chat_template(
        parsed.messages, tokenize=True, return_dict=False, thinking_effort="max",
        image_prompts=["<|media_begin|>image 8000x8000<|media_content|><|media_pad|><|media_end|>"],
    )
    assert service.count(request) == len(ids) - 1 + geometry.tokens


@pytest.mark.model("kimi-k2.6")
@pytest.mark.parametrize("model", ["kimi-k2.6", "moonshotai/Kimi-K2.6"])
def test_alias_and_empty_metadata(service, model):
    request = {"model": model, "messages": [{"role": "user", "content": "hello"}]}
    assert service.count({**request, "multimodal_metadata": []}) == service.count(request)


@pytest.mark.parametrize("model", [
    pytest.param("kimi-k2.6", marks=pytest.mark.model("kimi-k2.6")),
    pytest.param("kimi-k3", marks=pytest.mark.model("kimi-k3")),
])
def test_http_image_count_and_invalid_metadata(service, model, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import maas_tokenizer.api as api
    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(tmp_path / "access.log"))
    monkeypatch.setenv("TOKENIZER_RUN_LOG_PATH", str(tmp_path / "run.log"))
    monkeypatch.setattr(api._service, "count", lambda request: 1)
    api.app.dependency_overrides[api.get_token_count_service] = lambda: service
    request = {**image_request(model), "multimodal_metadata": [
        {"media_type": "image", "shape": [800, 600]}]}
    try:
        with TestClient(api.app) as client:
            response = client.post("/tokenizer", json=request)
            assert response.status_code == 200
            assert response.json() == {"prompt_tokens": service.count(request)}
            request["multimodal_metadata"][0]["shape"] = [0, 600]
            response = client.post("/tokenizer", json=request)
            assert response.status_code == 400
            assert response.json()["error_code"] == "request_processing_error"
    finally:
        api.app.dependency_overrides.clear()


@pytest.mark.parametrize("kind", ["video", "video_url", "audio", "input_audio", "input_image"])
def test_mixed_media_rejected_before_loading(kind):
    request = image_request()
    request["messages"][0]["content"].append({"type": kind})
    request["multimodal_metadata"] = [{"media_type": "image", "shape": [1, 1]}]
    service = TokenCountService()
    with pytest.raises(RequestProcessingError, match="image/image_url only"):
        service.count(request)
    assert not service.cached_profiles


def test_extra_metadata_is_not_silently_counted():
    request = image_request()
    request["multimodal_metadata"] = [{"media_type": "image", "shape": [1, 1]}] * 2
    with pytest.raises(RequestProcessingError, match="count must match"):
        TokenCountService().count(request)


def test_geometry_rejects_unknown_profile():
    from maas_tokenizer.image_tokens import image_geometry
    with pytest.raises(RequestProcessingError, match="only Kimi"):
        image_geometry(10, 10, "glm-5.2")
