# Pod Queue and Access Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded per-Pod FIFO that serializes tokenizer computation and add one dual-destination access log line per `/tokenizer` request.

**Architecture:** FastAPI owns an async scheduler with a 100-item `asyncio.Queue`; one consumer submits work to a dedicated `ThreadPoolExecutor(max_workers=1)`. Middleware records request outcome and timing to a logger configured with stdout and a rotating file handler.

**Tech Stack:** Python 3.11, FastAPI, asyncio, concurrent.futures, standard-library logging, pytest/TestClient.

## Global Constraints

- One Uvicorn worker and one CPU per Pod.
- One running tokenizer computation and at most 100 waiting requests.
- Reject a full queue immediately and reject a queued request that has not started within 2 seconds.
- Return HTTP 429 with `Retry-After: 1` for both admission failures.
- Write access logs to stdout and `/opt/cloud/logs/access.log`, rotating at 100 MB and retaining 5 backups.
- Never log request content; sanitize all untrusted log values.
- Do not route `/health` through the tokenizer queue.

---

### Task 1: Bounded Serial Scheduler

**Files:**
- Create: `src/maas_tokenizer/scheduler.py`
- Create: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: synchronous callable `Callable[[], int]`.
- Produces: `SerialScheduler(queue_size: int, queue_timeout_seconds: float)`, async `start()`, `close()`, and `submit(call: Callable[[], int]) -> ExecutionResult`; `QueueFullError` and `QueueTimeoutError`; `ExecutionResult(value, queue_wait_ms, process_ms)`.

- [ ] Write async tests using blocking events to assert maximum concurrent calls is one, FIFO start order, immediate full-queue rejection, timeout-before-start, and skipped cancelled work.
- [ ] Run `.venv/bin/pytest tests/test_scheduler.py -q` and confirm imports fail because the module does not exist.
- [ ] Implement one `asyncio.Queue`, one consumer task, a single-thread executor, per-job started/completion futures, and cancellation markers. Measure wait and processing time with `time.perf_counter()`.
- [ ] Run `.venv/bin/pytest tests/test_scheduler.py -q` and confirm all scheduler tests pass.
- [ ] Commit `scheduler.py` and `test_scheduler.py` as `feat: add bounded serial tokenizer scheduler`.

### Task 2: Dual-Destination Access Logger

**Files:**
- Create: `src/maas_tokenizer/access_logging.py`
- Create: `tests/test_access_logging.py`

**Interfaces:**
- Produces: immutable `AccessLogConfig.from_env()`, `configure_access_logger(config) -> logging.Logger`, `sanitize_log_value(value: object) -> str`, and `log_access(logger, AccessRecord) -> None`.
- `AccessRecord` contains timestamp, span ID, model, status, reason, HTTP status, queue wait, process, and total milliseconds.

- [ ] Write tests with a temporary file and captured stream for identical single-line output, field presence, newline sanitization, and rotation with a small test threshold.
- [ ] Run `.venv/bin/pytest tests/test_access_logging.py -q` and confirm imports fail.
- [ ] Implement environment parsing, validation, directory creation, non-propagating named logger, stdout handler, `RotatingFileHandler`, and key-value line formatting.
- [ ] Run `.venv/bin/pytest tests/test_access_logging.py -q` and confirm all logging tests pass.
- [ ] Commit logger code and tests as `feat: add rotating tokenizer access logs`.

### Task 3: FastAPI Lifecycle and Admission Control

**Files:**
- Modify: `src/maas_tokenizer/api.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Endpoint remains `POST /tokenizer -> int`.
- FastAPI lifespan owns scheduler and logger instances stored on `app.state`.
- Handler awaits `scheduler.submit(lambda: service.count(request))` and maps queue errors to stable 429 responses.
- Middleware reads `X-Span-Id`, captures final HTTP status, and logs one `AccessRecord` after every `/tokenizer` response.

- [ ] Add API tests for 429 `queue_full`, 429 `queue_timeout`, `Retry-After`, direct integer success, span/model/status/timings in one log record, business errors, malformed JSON, and responsive `/health` during a blocked calculation.
- [ ] Run `.venv/bin/pytest tests/test_api.py -q` and confirm the new tests fail for missing lifespan/scheduler behavior.
- [ ] Convert the endpoint to async, add lifespan setup/teardown, scheduler dependency, `/health`, queue error mapping, request state timings, and logging middleware. Ensure generated span IDs use UUID4 and header values are sanitized by the logger.
- [ ] Run `.venv/bin/pytest tests/test_api.py tests/test_scheduler.py tests/test_access_logging.py -q` and confirm all pass.
- [ ] Commit API integration as `feat: serialize tokenizer requests with bounded admission`.

### Task 4: Deployment Contract and Regression

**Files:**
- Modify: `README.md`
- Modify: `Dockerfile` only if it is intentionally brought under version control by the user; otherwise document required directory ownership without staging the existing untracked file.

**Interfaces:**
- Documents the five environment variables and mandatory `--workers 1` deployment constraint.
- Documents `/health`, both 429 reasons, log fields, rotation, and writable `/opt/cloud/logs` volume requirement.

- [ ] Update README startup and Kubernetes guidance, including `mkdir -p /opt/cloud/logs` and ownership by runtime UID 1000.
- [ ] Run `.venv/bin/pytest -q`, then run each supported model selection with `.venv/bin/pytest --model <profile> -q`.
- [ ] Run `git diff --check` and search current source/README for stale concurrency or endpoint statements.
- [ ] Inspect `git status --short` and ensure the user's pre-existing untracked Docker files remain untouched and unstaged.
- [ ] Commit only planned tracked files as `docs: describe queue and access log operations`.
