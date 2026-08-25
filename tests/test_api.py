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
    monkeypatch.setattr("maas_tokenizer.api._service.count", lambda request: 1)
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
    assert "request_body_bytes=" not in log_line
    assert "request_body=" not in log_line


def test_enabled_request_body_logging_records_compact_json(
    tmp_path, monkeypatch
) -> None:
    log_path = tmp_path / "access.log"
    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(log_path))
    monkeypatch.setenv("TOKENIZER_LOG_REQUEST_BODY", "true")
    monkeypatch.setattr("maas_tokenizer.api._service.count", lambda request: 7)
    body = {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "你好 world\nsecond"}],
    }

    with TestClient(app) as test_client:
        response = test_client.post("/tokenizer", json=body)

    expected = (
        '{"model":"glm-5.2","messages":'
        '[{"role":"user","content":"你好 world\\nsecond"}]}'
    )
    log_line = log_path.read_text(encoding="utf-8").strip()
    assert response.status_code == 200
    assert f"request_body_bytes={len(expected.encode('utf-8'))}" in log_line
    assert f"request_body={expected}" in log_line
    assert len(log_line.splitlines()) == 1


def test_oversized_request_body_logging_uses_omission_marker(
    tmp_path, monkeypatch
) -> None:
    log_path = tmp_path / "access.log"
    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(log_path))
    monkeypatch.setenv("TOKENIZER_LOG_REQUEST_BODY", "true")
    monkeypatch.setenv("TOKENIZER_LOG_REQUEST_BODY_MAX_BYTES", "1")
    monkeypatch.setattr("maas_tokenizer.api._service.count", lambda request: 7)

    with TestClient(app) as test_client:
        response = test_client.post(
            "/tokenizer", json={"model": "glm-5.2", "messages": []}
        )

    log_line = log_path.read_text(encoding="utf-8").strip()
    assert response.status_code == 200
    assert "request_body_bytes=" in log_line
    assert "request_body=<omitted_too_large>" in log_line
    assert 'request_body={"model"' not in log_line


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


def test_enabled_request_body_logging_is_kept_for_rejected_request(
    tmp_path, monkeypatch
) -> None:
    log_path = tmp_path / "access.log"
    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(log_path))
    monkeypatch.setenv("TOKENIZER_LOG_REQUEST_BODY", "true")
    monkeypatch.setattr("maas_tokenizer.api._service.count", lambda request: 1)
    app.dependency_overrides[get_scheduler] = lambda: RejectingScheduler(
        QueueFullError("tokenizer queue is full")
    )

    try:
        with TestClient(app) as test_client:
            response = test_client.post(
                "/tokenizer",
                json={"model": "glm-5.2", "messages": []},
            )
    finally:
        app.dependency_overrides.clear()

    log_line = log_path.read_text(encoding="utf-8").strip()
    assert response.status_code == 429
    assert "status=rejected" in log_line
    assert 'request_body={"model":"glm-5.2","messages":[]}' in log_line


def test_health_is_available_without_entering_scheduler(client) -> None:
    response = client[0].get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_warms_glm_5_2_before_serving(tmp_path, monkeypatch) -> None:
    calls: list[Mapping[str, Any]] = []
    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(tmp_path / "access.log"))
    monkeypatch.setattr(
        "maas_tokenizer.api._service.count",
        lambda request: calls.append(request) or 1,
    )

    with TestClient(app):
        assert calls == [
            {
                "model": "glm-5.2",
                "messages": [{"role": "user", "content": "warmup"}],
            }
        ]


class WarmupFailingScheduler:
    def __init__(self, **kwargs) -> None:
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def submit(self, call):
        call()
        raise AssertionError("unreachable")

    async def close(self) -> None:
        self.closed = True


def test_warmup_failure_prevents_startup_and_closes_scheduler(
    tmp_path, monkeypatch
) -> None:
    created: list[WarmupFailingScheduler] = []

    def scheduler_factory(**kwargs):
        scheduler = WarmupFailingScheduler(**kwargs)
        created.append(scheduler)
        return scheduler

    def fail(_request):
        raise RuntimeError("warmup failed")

    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(tmp_path / "access.log"))
    monkeypatch.setattr("maas_tokenizer.api.SerialScheduler", scheduler_factory)
    monkeypatch.setattr("maas_tokenizer.api._service.count", fail)

    with pytest.raises(RuntimeError, match="warmup failed"):
        with TestClient(app):
            pass

    assert created[0].started is True
    assert created[0].closed is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TOKENIZER_QUEUE_SIZE", "0"),
        ("TOKENIZER_QUEUE_TIMEOUT_SECONDS", "0"),
    ],
)
def test_startup_rejects_non_positive_queue_configuration(
    tmp_path, monkeypatch, name: str, value: str
) -> None:
    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(tmp_path / "access.log"))
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        with TestClient(app):
            pass
