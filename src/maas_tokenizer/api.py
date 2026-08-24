"""FastAPI adapter for the token count service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
import os
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request

from .access_logging import (
    AccessLogConfig,
    AccessRecord,
    configure_access_logger,
    log_access,
)
from .service import TokenCountService
from .assets import AssetIntegrityError
from .errors import ProcessorRequiredError, RequestProcessingError, UnknownModelError
from .scheduler import QueueFullError, QueueTimeoutError, SerialScheduler


_service = TokenCountService()
_WARMUP_REQUEST: dict[str, Any] = {
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": "warmup"}],
}


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@asynccontextmanager
async def lifespan(application: FastAPI):
    scheduler = SerialScheduler(
        queue_size=_positive_int("TOKENIZER_QUEUE_SIZE", 100),
        queue_timeout_seconds=_positive_float(
            "TOKENIZER_QUEUE_TIMEOUT_SECONDS", 2.0
        ),
    )
    logger = configure_access_logger(AccessLogConfig.from_env())
    application.state.scheduler = scheduler
    application.state.access_logger = logger
    await scheduler.start()
    try:
        await scheduler.submit(lambda: _service.count(_WARMUP_REQUEST))
        yield
    finally:
        await scheduler.close()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


app = FastAPI(title="MaaS Tokenizer", lifespan=lifespan)


def get_token_count_service() -> TokenCountService:
    return _service


def get_scheduler(request: Request) -> SerialScheduler:
    return request.app.state.scheduler


@app.middleware("http")
async def tokenizer_access_log(request: Request, call_next):
    if request.url.path != "/tokenizer":
        return await call_next(request)
    started_at = perf_counter()
    request.state.span_id = request.headers.get("X-Span-Id") or str(uuid4())
    request.state.model = "-"
    request.state.access_status = "failed"
    request.state.access_reason = "-"
    request.state.queue_wait_ms = 0.0
    request.state.process_ms = 0.0
    response = await call_next(request)
    if response.status_code >= 400 and request.state.access_reason == "-":
        request.state.access_reason = f"http_{response.status_code}"
    log_access(
        request.app.state.access_logger,
        AccessRecord(
            timestamp=datetime.now(UTC),
            span_id=request.state.span_id,
            model=request.state.model,
            status=request.state.access_status,
            reason=request.state.access_reason,
            http_status=response.status_code,
            queue_wait_ms=request.state.queue_wait_ms,
            process_ms=request.state.process_ms,
            total_ms=(perf_counter() - started_at) * 1000,
        ),
    )
    return response


@app.post("/tokenizer", response_model=int)
async def token_count(
    request_body: dict[str, Any],
    request: Request,
    service: TokenCountService = Depends(get_token_count_service),
    scheduler: SerialScheduler = Depends(get_scheduler),
) -> int:
    model = request_body.get("model")
    request.state.model = model if isinstance(model, str) else "-"
    try:
        result = await scheduler.submit(lambda: service.count(request_body))
        request.state.queue_wait_ms = result.queue_wait_ms
        request.state.process_ms = result.process_ms
        request.state.access_status = "success"
        return result.value
    except QueueFullError as error:
        _capture_error_timings(request, error)
        request.state.access_status = "rejected"
        request.state.access_reason = "queue_full"
        raise _http_error(
            429, "admission_control", "queue_full", error, headers={"Retry-After": "1"}
        ) from error
    except QueueTimeoutError as error:
        _capture_error_timings(request, error)
        request.state.access_status = "rejected"
        request.state.access_reason = "queue_timeout"
        raise _http_error(
            429,
            "admission_control",
            "queue_timeout",
            error,
            headers={"Retry-After": "1"},
        ) from error
    except UnknownModelError as error:
        _capture_error_timings(request, error)
        request.state.access_reason = "unknown_model"
        raise _http_error(404, "profile_resolution", "unknown_model", error) from error
    except ProcessorRequiredError as error:
        _capture_error_timings(request, error)
        request.state.access_reason = "multimodal_processor_required"
        raise _http_error(
            501,
            "processor_required",
            "multimodal_processor_required",
            error,
        ) from error
    except RequestProcessingError as error:
        _capture_error_timings(request, error)
        request.state.access_reason = "request_processing_error"
        raise _http_error(
            400, "request_validation", "request_processing_error", error
        ) from error
    except AssetIntegrityError as error:
        _capture_error_timings(request, error)
        request.state.access_reason = "asset_integrity_error"
        raise _http_error(
            500, "asset_integrity", "asset_integrity_error", error
        ) from error
    except Exception as error:
        _capture_error_timings(request, error)
        request.state.access_reason = "internal_error"
        raise HTTPException(
            status_code=500,
            detail={
                "stage": "internal",
                "type": "internal_error",
                "message": "internal server error",
            },
        ) from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _capture_error_timings(request: Request, error: Exception) -> None:
    request.state.queue_wait_ms = getattr(error, "queue_wait_ms", 0.0)
    request.state.process_ms = getattr(error, "process_ms", 0.0)


def _http_error(
    status_code: int,
    stage: str,
    error_type: str,
    error: Exception,
    *,
    headers: dict[str, str] | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"stage": stage, "type": error_type, "message": str(error)},
        headers=headers,
    )
