from collections.abc import Mapping
from typing import Any

from fastapi.testclient import TestClient
import pytest

from maas_tokenizer.api import app, get_token_count_service
from maas_tokenizer.assets import AssetIntegrityError
from maas_tokenizer.errors import (
    ProcessorRequiredError,
    RequestProcessingError,
    UnknownModelError,
)


class FixedService:
    def count(self, request: Mapping[str, Any]) -> int:
        assert request["model"] == "glm-5.2"
        return 18


def test_token_count_endpoint_returns_only_count() -> None:
    app.dependency_overrides[get_token_count_service] = lambda: FixedService()
    try:
        response = TestClient(app).post(
            "/tokenizer",
            json={
                "model": "glm-5.2",
                "messages": [{"role": "user", "content": "你好"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == 18
    assert type(response.json()) is int


def test_old_token_count_endpoint_is_not_available() -> None:
    response = TestClient(app).post(
        "/v1/token-count",
        json={
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "你好"}],
        },
    )

    assert response.status_code == 404


class FailingService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def count(self, request: Mapping[str, Any]) -> int:
        raise self.error


@pytest.mark.parametrize(
    ("error", "status", "stage", "error_type"),
    [
        (UnknownModelError("unknown"), 404, "profile_resolution", "unknown_model"),
        (RequestProcessingError("bad request"), 400, "request_validation", "request_processing_error"),
        (ProcessorRequiredError("needs processor"), 501, "processor_required", "multimodal_processor_required"),
        (AssetIntegrityError("bad assets"), 500, "asset_integrity", "asset_integrity_error"),
        (RuntimeError("boom"), 500, "internal", "internal_error"),
    ],
)
def test_expected_errors_have_stable_http_mapping(
    error: Exception, status: int, stage: str, error_type: str
) -> None:
    app.dependency_overrides[get_token_count_service] = lambda: FailingService(error)
    try:
        response = TestClient(app, raise_server_exceptions=False).post(
            "/tokenizer",
            json={"model": "glm-5.2", "messages": [{"role": "user", "content": "hi"}]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status
    message = "internal server error" if error_type == "internal_error" else str(error)
    assert response.json() == {
        "detail": {"stage": stage, "type": error_type, "message": message}
    }


def test_malformed_json_uses_fastapi_422() -> None:
    response = TestClient(app).post(
        "/tokenizer", content="{", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
