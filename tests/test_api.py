from collections.abc import Mapping
from typing import Any

from fastapi.testclient import TestClient
import pytest

from maas_tokenizer.api import app, get_scheduler, get_token_count_service
from maas_tokenizer.assets import AssetIntegrityError
from maas_tokenizer.errors import (
    ProcessorRequiredError,
    RequestProcessingError,
    UnknownModelError,
)
from maas_tokenizer.scheduler import ExecutionResult, QueueFullError, QueueTimeoutError


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(tmp_path / "access.log"))
    with TestClient(app) as test_client:
        yield test_client, tmp_path / "access.log"
    app.dependency_overrides.clear()


class FixedService:
    def count(self, request: Mapping[str, Any]) -> int:
        assert request["model"] == "glm-5.2"
        return 18


def test_token_count_endpoint_returns_only_count_and_logs_access(client) -> None:
    test_client, log_path = client
    app.dependency_overrides[get_token_count_service] = lambda: FixedService()
    response = test_client.post(
        "/tokenizer",
        headers={"X-Span-Id": "span-123"},
        json={
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "你好"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == 18
    assert type(response.json()) is int
    log_line = log_path.read_text(encoding="utf-8").strip()
    assert "span_id=span-123" in log_line
    assert "model=glm-5.2" in log_line
    assert "status=success" in log_line
    assert "http_status=200" in log_line
    assert "queue_wait_ms=" in log_line
    assert "process_ms=" in log_line
    assert "total_ms=" in log_line


def test_old_token_count_endpoint_is_not_available(client) -> None:
    response = client[0].post(
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
    client, error: Exception, status: int, stage: str, error_type: str
) -> None:
    app.dependency_overrides[get_token_count_service] = lambda: FailingService(error)
    response = client[0].post(
        "/tokenizer",
        json={"model": "glm-5.2", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == status
    message = "internal server error" if error_type == "internal_error" else str(error)
    assert response.json() == {
        "detail": {"stage": stage, "type": error_type, "message": message}
    }


def test_malformed_json_uses_fastapi_422(client) -> None:
    response = client[0].post(
        "/tokenizer", content="{", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


class RejectingScheduler:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def submit(self, call) -> ExecutionResult:
        raise self.error


@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        (QueueFullError("tokenizer queue is full"), "queue_full"),
        (QueueTimeoutError("tokenizer queue wait timed out"), "queue_timeout"),
    ],
)
def test_admission_rejection_returns_429_with_retry_after(
    client, error: Exception, error_type: str
) -> None:
    app.dependency_overrides[get_scheduler] = lambda: RejectingScheduler(error)
    response = client[0].post(
        "/tokenizer",
        headers={"X-Span-Id": "rejected-span"},
        json={"model": "glm-5.2", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"
    assert response.json()["detail"]["type"] == error_type
    log_line = client[1].read_text(encoding="utf-8").strip()
    assert "span_id=rejected-span" in log_line
    assert "status=rejected" in log_line
    assert f"reason={error_type}" in log_line


def test_health_is_available_without_entering_scheduler(client) -> None:
    response = client[0].get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
