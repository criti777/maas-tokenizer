"""Production entrypoint with file-only Uvicorn logging."""

from __future__ import annotations

import uvicorn

from .run_logging import (
    RunLogConfig,
    configure_process_file_logging,
    configure_run_logger,
    redirect_standard_streams,
)


def main() -> None:
    logger = configure_run_logger(RunLogConfig.from_env())
    configure_process_file_logging(logger)
    redirect_standard_streams(logger)
    uvicorn.run(
        "maas_tokenizer.api:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
        access_log=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
