from pathlib import Path

from maas_tokenizer.run_logging import RunLogConfig, configure_run_logger


def test_run_log_is_written_only_to_rotating_file(tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "logs" / "run.log"
    logger = configure_run_logger(
        RunLogConfig(log_path=log_path, max_bytes=10_000, backup_count=2)
    )

    logger.info("event=service_started")
    for handler in logger.handlers:
        handler.flush()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    line = log_path.read_text(encoding="utf-8").strip()
    assert "|INFO|event=service_started" in line


def test_run_log_keeps_traceback_on_one_physical_line(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    logger = configure_run_logger(
        RunLogConfig(log_path=log_path, max_bytes=10_000, backup_count=1)
    )

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logger.exception("event=request_failed")
    for handler in logger.handlers:
        handler.flush()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "RuntimeError: boom" in lines[0]
    assert "\\n" in lines[0]


def test_run_log_rotates(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    logger = configure_run_logger(
        RunLogConfig(log_path=log_path, max_bytes=120, backup_count=2)
    )

    for number in range(10):
        logger.info("event=test|number=%d|payload=%s", number, "x" * 40)
    for handler in logger.handlers:
        handler.flush()

    assert log_path.exists()
    assert (tmp_path / "run.log.1").exists()
