from datetime import UTC, datetime
import logging
from pathlib import Path

from maas_tokenizer.access_logging import (
    AccessLogConfig,
    AccessRecord,
    configure_access_logger,
    log_access,
)


def _record(span_id: str = "span-1") -> AccessRecord:
    return AccessRecord(
        timestamp=datetime(2026, 8, 24, 7, 20, 31, tzinfo=UTC),
        span_id=span_id,
        model="glm-5.2",
        status="success",
        reason="-",
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
    assert "span_id=span-1" in file_line
    assert "model=glm-5.2" in file_line
    assert "status=success" in file_line
    assert "http_status=200" in file_line
    assert "queue_wait_ms=1.25" in file_line
    assert "process_ms=2.50" in file_line
    assert "total_ms=4.00" in file_line


def test_access_log_sanitizes_newlines(tmp_path: Path) -> None:
    log_path = tmp_path / "access.log"
    logger = configure_access_logger(
        AccessLogConfig(log_path=log_path, max_bytes=10_000, backup_count=1)
    )

    log_access(logger, _record("good\nstatus=forged"))
    for handler in logger.handlers:
        handler.flush()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "span_id=good\\nstatus=forged" in lines[0]


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


def teardown_module() -> None:
    logger = logging.getLogger("maas_tokenizer.access")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
