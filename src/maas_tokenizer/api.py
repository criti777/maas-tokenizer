"""FastAPI adapter for the token count service."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from .service import TokenCountService
from .assets import AssetIntegrityError
from .errors import ProcessorRequiredError, RequestProcessingError, UnknownModelError


app = FastAPI(title="MaaS Tokenizer")
_service = TokenCountService()


def get_token_count_service() -> TokenCountService:
    return _service


@app.post("/tokenizer", response_model=int)
def token_count(
    request: dict[str, Any],
    service: TokenCountService = Depends(get_token_count_service),
) -> int:
    try:
        count = service.count(request)
    except UnknownModelError as error:
        raise _http_error(404, "profile_resolution", "unknown_model", error) from error
    except ProcessorRequiredError as error:
        raise _http_error(
            501,
            "processor_required",
            "multimodal_processor_required",
            error,
        ) from error
    except RequestProcessingError as error:
        raise _http_error(
            400, "request_validation", "request_processing_error", error
        ) from error
    except AssetIntegrityError as error:
        raise _http_error(
            500, "asset_integrity", "asset_integrity_error", error
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "stage": "internal",
                "type": "internal_error",
                "message": "internal server error",
            },
        ) from error
    return count


def _http_error(
    status_code: int, stage: str, error_type: str, error: Exception
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"stage": stage, "type": error_type, "message": str(error)},
    )
