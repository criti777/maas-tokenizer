"""File-only rotating runtime logging for the tokenizer service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from threading import Lock
from zoneinfo import ZoneInfo


_RUN_LOG_TIMEZONE = ZoneInfo("Asia/Shanghai")
_LOGGER_NAME = "maas_tokenizer.run"


@dataclass(frozen=True)
class RunLogConfig:
    log_path: Path = Path("/opt/cloud/logs/maas-tokenizer/run.log")
    max_bytes: int = 100 * 1024 * 1024
    backup_count: int = 5
    level: str = "INFO"

    @classmethod
    def from_env(cls) -> "RunLogConfig":
        max_bytes = int(
            os.getenv("TOKENIZER_RUN_LOG_MAX_BYTES", str(100 * 1024 * 1024))
        )
        backup_count = int(os.getenv("TOKENIZER_RUN_LOG_BACKUP_COUNT", "5"))
        level = os.getenv("TOKENIZER_RUN_LOG_LEVEL", "INFO").upper()
        if max_bytes <= 0:
            raise ValueError("TOKENIZER_RUN_LOG_MAX_BYTES must be positive")
        if backup_count <= 0:
            raise ValueError("TOKENIZER_RUN_LOG_BACKUP_COUNT must be positive")
        if level not in logging.getLevelNamesMapping():
            raise ValueError("TOKENIZER_RUN_LOG_LEVEL must be a valid logging level")
        return cls(
            log_path=Path(
                os.getenv(
                    "TOKENIZER_RUN_LOG_PATH",
                    "/opt/cloud/logs/maas-tokenizer/run.log",
                )
            ),
            max_bytes=max_bytes,
            backup_count=backup_count,
            level=level,
        )


class _SingleLineFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        timestamp = datetime.fromtimestamp(record.created, _RUN_LOG_TIMEZONE)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def formatException(self, exc_info) -> str:
        return super().formatException(exc_info).replace("\r", "\\r").replace(
            "\n", "\\n"
        )

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record).replace("\r", "\\r").replace("\n", "\\n")


def configure_run_logger(config: RunLogConfig) -> logging.Logger:
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(config.level)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    handler = RotatingFileHandler(
        config.log_path,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(_SingleLineFormatter("%(asctime)s|%(levelname)s|%(message)s"))
    logger.addHandler(handler)
    return logger


def configure_process_file_logging(logger: logging.Logger) -> None:
    """Route process and Uvicorn logs to run.log without console handlers."""
    handler = logger.handlers[0]
    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        existing.close()
        root_logger.removeHandler(existing)
    root_logger.handlers = [handler]
    root_logger.setLevel(logger.level)
    logging.captureWarnings(True)

    uvicorn_logger = logging.getLogger("uvicorn")
    for existing in list(uvicorn_logger.handlers):
        existing.close()
        uvicorn_logger.removeHandler(existing)
    uvicorn_logger.handlers = [handler]
    uvicorn_logger.setLevel(logger.level)
    uvicorn_logger.propagate = False

    for name in ("uvicorn.error",):
        child = logging.getLogger(name)
        child.handlers.clear()
        child.propagate = True
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False
    access_logger.disabled = True


class _LoggerStream:
    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level
        self.encoding = "utf-8"
        self._buffer = ""
        self._lock = Lock()

    def write(self, value: str) -> int:
        with self._lock:
            self._buffer += value
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line:
                    self.logger.log(self.level, "event=stdio|message=%s", line)
        return len(value)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                self.logger.log(
                    self.level, "event=stdio|message=%s", self._buffer
                )
                self._buffer = ""

    def isatty(self) -> bool:
        return False


def redirect_standard_streams(logger: logging.Logger) -> None:
    sys.stdout = _LoggerStream(logger, logging.INFO)
    sys.stderr = _LoggerStream(logger, logging.ERROR)


def get_run_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)
