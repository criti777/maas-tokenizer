"""Single-line access logging for tokenizer requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys


@dataclass(frozen=True)
class AccessLogConfig:
    log_path: Path = Path("/opt/cloud/logs/access.log")
    max_bytes: int = 100 * 1024 * 1024
    backup_count: int = 5

    @classmethod
    def from_env(cls) -> "AccessLogConfig":
        max_bytes = int(os.getenv("TOKENIZER_LOG_MAX_BYTES", str(100 * 1024 * 1024)))
        backup_count = int(os.getenv("TOKENIZER_LOG_BACKUP_COUNT", "5"))
        if max_bytes <= 0:
            raise ValueError("TOKENIZER_LOG_MAX_BYTES must be positive")
        if backup_count <= 0:
            raise ValueError("TOKENIZER_LOG_BACKUP_COUNT must be positive")
        return cls(
            log_path=Path(
                os.getenv("TOKENIZER_LOG_PATH", "/opt/cloud/logs/access.log")
            ),
            max_bytes=max_bytes,
            backup_count=backup_count,
        )


@dataclass(frozen=True)
class AccessRecord:
    timestamp: datetime
    span_id: str
    model: str
    status: str
    reason: str
    http_status: int
    queue_wait_ms: float
    process_ms: float
    total_ms: float


def sanitize_log_value(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace(" ", "_")
    )


def configure_access_logger(config: AccessLogConfig) -> logging.Logger:
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("maas_tokenizer.access")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        config.log_path,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return logger


def log_access(logger: logging.Logger, record: AccessRecord) -> None:
    values = {
        "timestamp": record.timestamp.isoformat(timespec="milliseconds"),
        "span_id": record.span_id,
        "model": record.model,
        "status": record.status,
        "reason": record.reason,
        "http_status": record.http_status,
        "queue_wait_ms": f"{record.queue_wait_ms:.2f}",
        "process_ms": f"{record.process_ms:.2f}",
        "total_ms": f"{record.total_ms:.2f}",
    }
    logger.info(
        " ".join(f"{key}={sanitize_log_value(value)}" for key, value in values.items())
    )
