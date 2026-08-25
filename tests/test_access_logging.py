from dataclasses import replace
from datetime import UTC, datetime
import logging
from pathlib import Path

import pytest

from maas_tokenizer.access_logging import (
    AccessLogConfig,
    AccessRecord,
    configure_access_logger,
    log_access,
    prepare_request_body_for_log,
)


def _record(span_id: str = "span-1") -> AccessRecord:
    return AccessRecord(
        timestamp=datetime(2026, 8, 24, 7, 20, 31, tzinfo=UTC),
        span_id=span_id,
        request_id="request-1",
        model="glm-5.2",
        content_length=82,
        token_count=18,
        error_code="",
        error_message="",
        http_status=200,
        queue_wait_ms=1.25,
        process_ms=2.5,
        total_ms=4.0,
    )


def test_access_log_is_written_to_stdout_and_file(tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "logs" / "access.log"
    logger = configure_access_logger(
        AccessLogConfig(log_path=log_path, max_bytes=10_000, backup_count=2)
    )

    log_access(logger, _record())
    for handler in logger.handlers:
        handler.flush()

    stdout_line = capsys.readouterr().out.strip()
    file_line = log_path.read_text(encoding="utf-8").strip()
    assert stdout_line == file_line
    assert file_line == (
        "2026-08-24 15:20:31.000"
        "|span-1"
        "|request-1"
        "|glm-5.2"
        "|82"
        "|18"
        "||"
        "|200"
        "|1.25"
        "|2.50"
        "|4.00"
        "|"
    )


def test_access_log_encodes_delimiter_and_crlf(tmp_path: Path) -> None:
    log_path = tmp_path / "access.log"
    logger = configure_access_logger(
        AccessLogConfig(log_path=log_path, max_bytes=10_000, backup_count=1)
    )

    log_access(logger, _record("good|value\r\nstatus=forged"))
    for handler in logger.handlers:
        handler.flush()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "|good%7Cvalue\\r\\nstatus=forged|" in lines[0]


def test_access_log_preserves_spaces_inside_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "access.log"
    logger = configure_access_logger(
        AccessLogConfig(log_path=log_path, max_bytes=10_000, backup_count=1)
    )

    log_access(logger, replace(_record(), model="glm 5.2 preview"))
    for handler in logger.handlers:
        handler.flush()

    line = log_path.read_text(encoding="utf-8").strip()
    assert "|glm 5.2 preview|" in line


def test_access_log_rotates(tmp_path: Path) -> None:
    log_path = tmp_path / "access.log"
    logger = configure_access_logger(
        AccessLogConfig(log_path=log_path, max_bytes=200, backup_count=2)
    )

    for number in range(10):
        log_access(logger, _record(f"span-{number}"))
    for handler in logger.handlers:
        handler.flush()

    assert log_path.exists()
    assert (tmp_path / "access.log.1").exists()


def test_request_body_logging_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TOKENIZER_LOG_REQUEST_BODY", raising=False)
    monkeypatch.delenv("TOKENIZER_LOG_REQUEST_BODY_MAX_BYTES", raising=False)

    config = AccessLogConfig.from_env()

    assert config.log_request_body is False
    assert config.request_body_max_bytes == 64 * 1024


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
def test_request_body_logging_accepts_enabled_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("TOKENIZER_LOG_REQUEST_BODY", value)

    assert AccessLogConfig.from_env().log_request_body is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off"])
def test_request_body_logging_accepts_disabled_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("TOKENIZER_LOG_REQUEST_BODY", value)

    assert AccessLogConfig.from_env().log_request_body is False


def test_request_body_logging_rejects_invalid_boolean(monkeypatch) -> None:
    monkeypatch.setenv("TOKENIZER_LOG_REQUEST_BODY", "sometimes")

    with pytest.raises(ValueError, match="TOKENIZER_LOG_REQUEST_BODY"):
        AccessLogConfig.from_env()


def test_request_body_logging_rejects_non_positive_limit(monkeypatch) -> None:
    monkeypatch.setenv("TOKENIZER_LOG_REQUEST_BODY_MAX_BYTES", "0")

    with pytest.raises(ValueError, match="TOKENIZER_LOG_REQUEST_BODY_MAX_BYTES"):
        AccessLogConfig.from_env()


def test_request_body_uses_compact_json_and_utf8_byte_size() -> None:
    body = {"model": "glm-5.2", "messages": [{"content": "你好 world"}]}

    prepared = prepare_request_body_for_log(body, max_bytes=1024)

    expected = '{"model":"glm-5.2","messages":[{"content":"你好 world"}]}'
    assert prepared.value == expected
    assert prepared.byte_count == len(expected.encode("utf-8"))


def test_oversized_request_body_is_not_logged() -> None:
    prepared = prepare_request_body_for_log({"content": "你好"}, max_bytes=1)

    assert prepared.byte_count > 1
    assert prepared.value == "<omitted_too_large>"


def test_request_body_access_log_stays_on_one_physical_line(tmp_path: Path) -> None:
    log_path = tmp_path / "access.log"
    logger = configure_access_logger(
        AccessLogConfig(log_path=log_path, max_bytes=10_000, backup_count=1)
    )
    prepared = prepare_request_body_for_log(
        {"messages": [{"content": "first|second\nthird\tfourth"}]},
        max_bytes=1024,
    )
    record = replace(
        _record(),
        request_body=prepared.value,
    )

    log_access(logger, record)
    for handler in logger.handlers:
        handler.flush()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert (
        '|{"messages":[{"content":"first%7Csecond\\nthird\\tfourth"}]}'
        in lines[0]
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [("TOKENIZER_LOG_MAX_BYTES", "0"), ("TOKENIZER_LOG_BACKUP_COUNT", "0")],
)
def test_access_log_config_rejects_non_positive_values(
    monkeypatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        AccessLogConfig.from_env()


def teardown_module() -> None:
    logger = logging.getLogger("maas_tokenizer.access")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
