from collections.abc import Mapping
from typing import Any

from fastapi.testclient import TestClient
import pytest

import maas_tokenizer.api as api_module
from maas_tokenizer.api import app, get_scheduler, get_token_count_service
from maas_tokenizer.assets import AssetIntegrityError
from maas_tokenizer.errors import (
    ProcessorRequiredError,
    RequestProcessingError,
    UnknownModelError,
)
from maas_tokenizer.scheduler import ExecutionResult, QueueFullError, QueueTimeoutError


@pytest.fixture(autouse=True)
def runtime_log_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOKENIZER_RUN_LOG_PATH", str(tmp_path / "run.log"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(tmp_path / "access.log"))
    monkeypatch.setenv("TOKENIZER_RUN_LOG_PATH", str(tmp_path / "run.log"))
    monkeypatch.setattr("maas_tokenizer.api._service.count", lambda request: 1)
    with TestClient(app) as test_client:
        yield test_client, tmp_path / "access.log"
    app.dependency_overrides.clear()


class FixedService:
    def count(self, request: Mapping[str, Any]) -> int:
        assert request["model"] == "glm-5.2"
        return 18


def test_token_count_endpoint_returns_object_and_logs_access(client) -> None:
    test_client, log_path = client
    app.dependency_overrides[get_token_count_service] = lambda: FixedService()
    response = test_client.post(
        "/tokenizer",
        headers={"X-Span-Id": "span-123", "X-Request-Id": "request-456"},
        json={
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "你好"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"token_count": 18}
    log_line = log_path.read_text(encoding="utf-8").strip()
    fields = log_line.split("|")
    assert len(fields) == 12
    assert fields[1] == "span-123"
    assert fields[2] == "request-456"
    assert fields[3] == "glm-5.2"
    assert int(fields[4]) == len(response.request.content)
    assert fields[5] == "18"
    assert fields[6:8] == ["", ""]
    assert fields[8] == "200"


def test_missing_request_ids_are_logged_as_empty_values(client) -> None:
    response = client[0].post(
        "/tokenizer",
        json={"model": "glm-5.2", "messages": []},
    )

    fields = client[1].read_text(encoding="utf-8").strip().split("|")
    assert response.status_code == 200
    assert fields[1:3] == ["", ""]


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
    fields = log_line.split("|")
    assert response.status_code == 200
    assert fields[12] == expected
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
    fields = log_line.split("|")
    assert response.status_code == 200
    assert fields[12] == "<omitted_too_large>"


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
    ("error", "status", "error_type"),
    [
        (UnknownModelError("unknown"), 404, "unknown_model"),
        (RequestProcessingError("bad request"), 400, "request_processing_error"),
        (ProcessorRequiredError("needs processor"), 501, "multimodal_processor_required"),
        (AssetIntegrityError("bad assets"), 500, "asset_integrity_error"),
        (RuntimeError("boom"), 500, "internal_error"),
    ],
)
def test_expected_errors_have_stable_http_mapping(
    client, error: Exception, status: int, error_type: str
) -> None:
    app.dependency_overrides[get_token_count_service] = lambda: FailingService(error)
    response = client[0].post(
        "/tokenizer",
        json={"model": "glm-5.2", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == status
    message = "internal server error" if error_type == "internal_error" else str(error)
    assert response.json() == {"error_code": error_type, "error_msg": message}
    fields = client[1].read_text(encoding="utf-8").strip().split("|")
    assert fields[5] == ""
    assert fields[6] == error_type
    assert fields[7] == message
    if error_type == "internal_error":
        run_log = (client[1].parent / "run.log").read_text(encoding="utf-8")
        assert "event=request_failed" in run_log
        assert "RuntimeError: boom" in run_log


def test_malformed_json_uses_fastapi_422(client) -> None:
    response = client[0].post(
        "/tokenizer", content="{", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json() == {
        "error_code": "request_validation_error",
        "error_msg": "invalid request body",
    }
    fields = client[1].read_text(encoding="utf-8").strip().split("|")
    assert fields[6] == "request_validation_error"
    assert fields[7] == "invalid request body"


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
    assert response.json() == {
        "error_code": error_type,
        "error_msg": str(error),
    }
    log_line = client[1].read_text(encoding="utf-8").strip()
    fields = log_line.split("|")
    assert fields[1] == "rejected-span"
    assert fields[5] == ""
    assert fields[6] == error_type
    assert fields[7] == str(error)


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
    fields = log_line.split("|")
    assert response.status_code == 429
    assert fields[6] == "queue_full"
    assert fields[12] == '{"model":"glm-5.2","messages":[]}'


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


def test_queue_timeout_defaults_to_200_milliseconds(monkeypatch) -> None:
    monkeypatch.delenv("TOKENIZER_QUEUE_TIMEOUT_MS", raising=False)

    assert api_module._queue_timeout_seconds() == pytest.approx(0.2)


def test_queue_timeout_converts_milliseconds_to_seconds(monkeypatch) -> None:
    monkeypatch.setenv("TOKENIZER_QUEUE_TIMEOUT_MS", "275")

    assert api_module._queue_timeout_seconds() == pytest.approx(0.275)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TOKENIZER_QUEUE_SIZE", "0"),
        ("TOKENIZER_QUEUE_TIMEOUT_MS", "0"),
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
