from pathlib import Path

import pytest

from maas_tokenizer.service import TokenCountService


@pytest.mark.model("kimi-k2.6")
def test_kimi_skips_image_payload_and_counts_text_parts_only() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    with_media = service.count(
        {
            "model": "kimi-k2.6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image_url", "image_url": {"url": "https://invalid.example/x.png"}},
                        {"type": "text", "text": "after"},
                    ],
                }
            ],
        }
    )
    text_only = service.count(
        {
            "model": "kimi-k2.6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "text", "text": "after"},
                    ],
                }
            ],
        }
    )
    assert with_media == text_only


@pytest.mark.parametrize(
    "profile",
    [
        pytest.param("deepseek-v3", marks=pytest.mark.model("deepseek-v3")),
        pytest.param("glm-5.2", marks=pytest.mark.model("glm-5.2")),
        pytest.param("glm-5.3", marks=pytest.mark.model("glm-5.3")),
        pytest.param("glm-5.3-flash", marks=pytest.mark.model("glm-5.3-flash")),
        pytest.param("glm-5.1", marks=pytest.mark.model("glm-5.1")),
        pytest.param("deepseek-v3.2", marks=pytest.mark.model("deepseek-v3.2")),
        pytest.param("deepseek-v4-flash", marks=pytest.mark.model("deepseek-v4-flash")),
        pytest.param("deepseek-v4-pro", marks=pytest.mark.model("deepseek-v4-pro")),
        pytest.param(
            "deepseek-v4.1-flash", marks=pytest.mark.model("deepseek-v4.1-flash")
        ),
        pytest.param("kimi-k2.6", marks=pytest.mark.model("kimi-k2.6")),
        pytest.param("kimi-k3", marks=pytest.mark.model("kimi-k3")),
        pytest.param("minimax-m2.7", marks=pytest.mark.model("minimax-m2.7")),
        pytest.param("minimax-m3", marks=pytest.mark.model("minimax-m3")),
    ],
)
def test_multimodal_parts_are_skipped_for_text_only_counting(profile: str) -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    with_media = service.count(
        {
            "model": profile,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,x"},
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "base64"},
                        },
                        {
                            "type": "video",
                            "video_url": "https://invalid.example/video",
                        },
                        {"type": "text", "text": "after"},
                    ],
                }
            ],
        }
    )
    text_only = service.count(
        {
            "model": profile,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "text", "text": "after"},
                    ],
                }
            ],
        }
    )
    assert with_media == text_only


@pytest.mark.model("deepseek-v4.1-flash")
def test_multimodal_blocks_are_skipped_before_v41_rendering() -> None:
    service = TokenCountService(assets_root=Path("model_assets"))
    with_media = service.count(
        {
            "model": "deepseek-v4.1-flash",
            "messages": [
                {
                    "role": "user",
                    "content": "",
                    "content_blocks": [
                        {"type": "text", "text": "caption"},
                        {"type": "image", "image": "opaque payload"},
                        {
                            "type": "tool_result",
                            "content": [
                                {"type": "text", "text": "tool text"},
                                {"type": "video_frame", "data": "opaque payload"},
                            ],
                        },
                    ],
                }
            ],
        }
    )
    text_only = service.count(
        {
            "model": "deepseek-v4.1-flash",
            "messages": [
                {
                    "role": "user",
                    "content": "",
                    "content_blocks": [
                        {"type": "text", "text": "caption"},
                        {"type": "tool_result", "content": "tool text"},
                    ],
                }
            ],
        }
    )
    assert with_media == text_only
