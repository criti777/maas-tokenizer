"""FastAPI adapter for the token count service."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, ConfigDict

from .service import TokenCountService


class TokenCountResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_count: int


app = FastAPI(title="MaaS Tokenizer")
_service = TokenCountService()


def get_token_count_service() -> TokenCountService:
    return _service


@app.post("/v1/token-count", response_model=TokenCountResponse)
def token_count(
    request: dict[str, Any],
    service: TokenCountService = Depends(get_token_count_service),
) -> TokenCountResponse:
    return TokenCountResponse(token_count=service.count(request))
