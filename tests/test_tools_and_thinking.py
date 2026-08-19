from pathlib import Path

import pytest

from maas_tokenizer.service import TokenCountService


@pytest.mark.model("deepseek-v3.2")
def test_v32_tool_declaration_matches_pinned_count() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    assert service.count(
        {
            "model": "deepseek-v3.2",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
    ) == 273

