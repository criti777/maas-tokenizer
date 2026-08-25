"""FastAPI adapter for the token count service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
import os
from time import perf_counter
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .access_logging import (
    AccessLogConfig,
    AccessRecord,
    configure_access_logger,
    log_access,
    prepare_request_body_for_log,
    sanitize_log_value,
)
from .service import TokenCountService
from .assets import AssetIntegrityError
from .errors import ProcessorRequiredError, RequestProcessingError, UnknownModelError
from .scheduler import QueueFullError, QueueTimeoutError, SerialScheduler
from .run_logging import (
    RunLogConfig,
    configure_process_file_logging,
    configure_run_logger,
)


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
    run_logger = configure_run_logger(RunLogConfig.from_env())
    configure_process_file_logging(run_logger)
    run_logger.info("event=service_starting")
    try:
        scheduler = SerialScheduler(
            queue_size=_positive_int("TOKENIZER_QUEUE_SIZE", 100),
            queue_timeout_seconds=_positive_float(
                "TOKENIZER_QUEUE_TIMEOUT_SECONDS", 2.0
            ),
        )
        access_log_config = AccessLogConfig.from_env()
        access_logger = configure_access_logger(access_log_config)
        application.state.scheduler = scheduler
        application.state.access_logger = access_logger
        application.state.access_log_config = access_log_config
        application.state.run_logger = run_logger
        await scheduler.start()
        warmup_started = perf_counter()
        run_logger.info("event=warmup_started|model=glm-5.2")
        await scheduler.submit(lambda: _service.count(_WARMUP_REQUEST))
        run_logger.info(
            "event=warmup_succeeded|model=glm-5.2|duration_ms=%.2f",
            (perf_counter() - warmup_started) * 1000,
        )
        run_logger.info("event=service_ready")
        yield
    except BaseException:
        run_logger.exception("event=service_lifecycle_failed")
        raise
    finally:
        run_logger.info("event=service_stopping")
        if "scheduler" in locals():
            await scheduler.close()
        if "access_logger" in locals():
            for handler in list(access_logger.handlers):
                handler.close()
                access_logger.removeHandler(handler)


app = FastAPI(title="MaaS Tokenizer", lifespan=lifespan)


class TokenCountResponse(BaseModel):
    token_count: int


class ErrorResponse(BaseModel):
    error_code: str
    error_msg: str


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        error_code: str,
        error_msg: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.error_msg = error_msg
        self.headers = headers


@app.exception_handler(APIError)
async def api_error_handler(request: Request, error: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"error_code": error.error_code, "error_msg": error.error_msg},
        headers=error.headers,
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    error_code = "request_validation_error"
    error_msg = "invalid request body"
    request.state.error_code = error_code
    request.state.error_message = error_msg
    return JSONResponse(
        status_code=422,
        content={"error_code": error_code, "error_msg": error_msg},
    )


def get_token_count_service() -> TokenCountService:
    return _service


def get_scheduler(request: Request) -> SerialScheduler:
    return request.app.state.scheduler


@app.middleware("http")
async def tokenizer_access_log(request: Request, call_next):
    if request.url.path != "/tokenizer":
        return await call_next(request)
    started_at = perf_counter()
    request.state.span_id = request.headers.get("X-Span-Id", "")
    request.state.request_id = request.headers.get("X-Request-Id", "")
    request.state.content_length = request.headers.get("Content-Length", "")
    request.state.model = ""
    request.state.token_count = None
    request.state.error_code = ""
    request.state.error_message = ""
    request.state.queue_wait_ms = 0.0
    request.state.process_ms = 0.0
    request.state.request_body = None
    response = await call_next(request)
    if response.status_code >= 400 and request.state.error_code == "":
        request.state.error_code = f"http_{response.status_code}"
        request.state.error_message = (
            f"request failed with HTTP {response.status_code}"
        )
    log_access(
        request.app.state.access_logger,
        AccessRecord(
            timestamp=datetime.now(UTC),
            span_id=request.state.span_id,
            request_id=request.state.request_id,
            model=request.state.model,
            content_length=request.state.content_length,
            token_count=request.state.token_count,
            error_code=request.state.error_code,
            error_message=request.state.error_message,
            http_status=response.status_code,
            queue_wait_ms=request.state.queue_wait_ms,
            process_ms=request.state.process_ms,
            total_ms=(perf_counter() - started_at) * 1000,
            request_body=request.state.request_body,
        ),
    )
    return response


@app.post(
    "/tokenizer",
    response_model=TokenCountResponse,
    responses={
        status: {"model": ErrorResponse}
        for status in (400, 404, 422, 429, 500, 501)
    },
)
async def token_count(
    request_body: dict[str, Any],
    request: Request,
    service: TokenCountService = Depends(get_token_count_service),
    scheduler: SerialScheduler = Depends(get_scheduler),
) -> TokenCountResponse:
    model = request_body.get("model")
    request.state.model = model if isinstance(model, str) else ""
    access_log_config: AccessLogConfig = request.app.state.access_log_config
    if access_log_config.log_request_body:
        prepared_body = prepare_request_body_for_log(
            request_body,
            max_bytes=access_log_config.request_body_max_bytes,
        )
        request.state.request_body = prepared_body.value
    try:
        result = await scheduler.submit(lambda: service.count(request_body))
        request.state.queue_wait_ms = result.queue_wait_ms
        request.state.process_ms = result.process_ms
        request.state.token_count = result.value
        return TokenCountResponse(token_count=result.value)
    except QueueFullError as error:
        _capture_error(request, error, "queue_full")
        raise _api_error(
            429, "queue_full", error, headers={"Retry-After": "1"}
        ) from error
    except QueueTimeoutError as error:
        _capture_error(request, error, "queue_timeout")
        raise _api_error(
            429,
            "queue_timeout",
            error,
            headers={"Retry-After": "1"},
        ) from error
    except UnknownModelError as error:
        _capture_error(request, error, "unknown_model")
        raise _api_error(404, "unknown_model", error) from error
    except ProcessorRequiredError as error:
        _capture_error(request, error, "multimodal_processor_required")
        raise _api_error(501, "multimodal_processor_required", error) from error
    except RequestProcessingError as error:
        _capture_error(request, error, "request_processing_error")
        raise _api_error(400, "request_processing_error", error) from error
    except AssetIntegrityError as error:
        _capture_error(request, error, "asset_integrity_error")
        _log_request_exception(request, error, "asset_integrity_error")
        raise _api_error(500, "asset_integrity_error", error) from error
    except Exception as error:
        _capture_error(
            request,
            error,
            "internal_error",
            message="internal server error",
        )
        _log_request_exception(request, error, "internal_error")
        raise APIError(500, "internal_error", "internal server error") from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _capture_error(
    request: Request,
    error: Exception,
    error_code: str,
    *,
    message: str | None = None,
) -> None:
    request.state.queue_wait_ms = getattr(error, "queue_wait_ms", 0.0)
    request.state.process_ms = getattr(error, "process_ms", 0.0)
    request.state.error_code = error_code
    request.state.error_message = str(error) if message is None else message


def _api_error(
    status_code: int,
    error_code: str,
    error: Exception,
    *,
    headers: dict[str, str] | None = None,
) -> APIError:
    return APIError(
        status_code,
        error_code,
        str(error),
        headers=headers,
    )


def _log_request_exception(
    request: Request, error: Exception, error_code: str
) -> None:
    request.app.state.run_logger.error(
        "event=request_failed|x_span_id=%s|x_request_id=%s|model=%s|error_code=%s",
        sanitize_log_value(request.state.span_id),
        sanitize_log_value(request.state.request_id),
        sanitize_log_value(request.state.model),
        error_code,
        exc_info=(type(error), error, error.__traceback__),
    )
