from collections.abc import Mapping
from typing import Any

from fastapi.testclient import TestClient

from maas_tokenizer.api import app, get_token_count_service


class FixedService:
    def count(self, request: Mapping[str, Any]) -> int:
        assert request["model"] == "glm-5.2"
        return 18


def test_token_count_endpoint_returns_only_count() -> None:
    app.dependency_overrides[get_token_count_service] = lambda: FixedService()
    try:
        response = TestClient(app).post(
            "/v1/token-count",
            json={
                "model": "glm-5.2",
                "messages": [{"role": "user", "content": "你好"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"token_count": 18}
