from types import SimpleNamespace
from typing import Any

import pytest

import maas_tokenizer.renderers as renderers
from maas_tokenizer.renderers import HFRenderer


class _UnusedEncoder:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        raise AssertionError("encode is not part of this renderer test")


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [(True, "enabled"), (False, "disabled")],
)
def test_hf_renderer_maps_boolean_thinking_to_named_mode(
    monkeypatch: pytest.MonkeyPatch, thinking: bool, expected: str
) -> None:
    captured: dict[str, Any] = {}

    def fake_render_chat(**kwargs: Any) -> tuple[None, str, None]:
        captured.update(kwargs["template_kwargs"])
        return None, "rendered", None

    monkeypatch.setattr(renderers, "render_chat", fake_render_chat)
    renderer = HFRenderer(
        SimpleNamespace(), _UnusedEncoder(), template_thinking_mode=True
    )
    parsed = SimpleNamespace(
        messages=[],
        tools=None,
        chat_template=None,
        chat_template_content_format="auto",
        add_special_tokens=False,
        template_kwargs=lambda _tools: {
            "thinking": thinking,
            "enable_thinking": thinking,
        },
    )

    renderer.render(parsed)

    assert captured["thinking_mode"] == expected
