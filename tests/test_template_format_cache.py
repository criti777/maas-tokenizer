import transformers.utils.chat_template_utils as hf_chat_utils

from vendor.vllm.extracted.template_format import detect_content_format


def test_content_format_detection_is_cached_per_template(monkeypatch) -> None:
    compile_template = hf_chat_utils._compile_jinja_template
    calls = 0

    def counted_compile(template: str):
        nonlocal calls
        calls += 1
        return compile_template(template)

    monkeypatch.setattr(hf_chat_utils, "_compile_jinja_template", counted_compile)
    detect_content_format.cache_clear()
    try:
        string_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
        parts_template = (
            "{% for message in messages %}"
            "{% for part in message.content %}{{ part.text }}{% endfor %}"
            "{% endfor %}"
        )
        assert detect_content_format(string_template) == "string"
        assert detect_content_format(string_template) == "string"
        assert calls == 1
        assert detect_content_format(parts_template) == "openai"
        assert detect_content_format(parts_template) == "openai"
        assert calls == 2
    finally:
        detect_content_format.cache_clear()
