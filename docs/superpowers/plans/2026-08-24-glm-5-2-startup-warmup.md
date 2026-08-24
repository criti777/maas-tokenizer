# GLM-5.2 Startup Warmup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Warm GLM-5.2 through the production token-count path before FastAPI reports application startup complete.

**Architecture:** The existing FastAPI lifespan starts the bounded serial scheduler, submits one fixed GLM-5.2 request through that scheduler, and waits for its result before yielding. The existing `TokenCountService` renderer cache retains both the Hugging Face template tokenizer and Gigatoken encoder; no second cache is introduced. Startup failure propagates while the existing cleanup path closes the scheduler and access-log handlers.

**Tech Stack:** Python 3.11, FastAPI lifespan, asyncio, pytest, FastAPI TestClient

## Global Constraints

- Preload only `glm-5.2`.
- Do not change `POST /tokenizer`, `GET /health`, or their response contracts.
- Do not modify the Dockerfile or run warmup during image construction.
- Other six models remain lazy-loaded on first request.
- Warmup must execute through `SerialScheduler.submit()` on the dedicated tokenizer thread.
- Warmup failure must prevent application startup.

---

### Task 1: Add fail-fast GLM-5.2 lifespan warmup

**Files:**
- Modify: `tests/test_api.py`
- Modify: `src/maas_tokenizer/api.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `SerialScheduler.submit(call: Callable[[], int]) -> ExecutionResult` and `TokenCountService.count(request: Mapping[str, Any]) -> int`.
- Produces: module constant `_WARMUP_REQUEST: dict[str, Any]`; FastAPI lifespan warms GLM-5.2 before `yield` and always closes startup resources.

- [ ] **Step 1: Isolate existing API tests from real model warmup**

Update the existing `client` fixture so every API test replaces the global startup service's `count` method before entering `TestClient`:

```python
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(tmp_path / "access.log"))
    monkeypatch.setattr("maas_tokenizer.api._service.count", lambda request: 1)
    with TestClient(app) as test_client:
        yield test_client, tmp_path / "access.log"
    app.dependency_overrides.clear()
```

- [ ] **Step 2: Write failing startup warmup tests**

Add a test that records the request passed to the global service and proves it runs before the `TestClient` context becomes usable:

```python
def test_startup_warms_glm_5_2_before_serving(tmp_path, monkeypatch) -> None:
    calls: list[Mapping[str, Any]] = []
    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(tmp_path / "access.log"))
    monkeypatch.setattr(
        "maas_tokenizer.api._service.count",
        lambda request: calls.append(request) or 1,
    )

    with TestClient(app):
        assert calls == [
            {
                "model": "glm-5.2",
                "messages": [{"role": "user", "content": "warmup"}],
            }
        ]
```

Add a minimal recording scheduler and a test proving warmup failure propagates and closes the scheduler:

```python
class WarmupFailingScheduler:
    def __init__(self, **kwargs) -> None:
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def submit(self, call):
        call()
        raise AssertionError("unreachable")

    async def close(self) -> None:
        self.closed = True


def test_warmup_failure_prevents_startup_and_closes_scheduler(
    tmp_path, monkeypatch
) -> None:
    created: list[WarmupFailingScheduler] = []

    def scheduler_factory(**kwargs):
        scheduler = WarmupFailingScheduler(**kwargs)
        created.append(scheduler)
        return scheduler

    def fail(_request):
        raise RuntimeError("warmup failed")

    monkeypatch.setenv("TOKENIZER_LOG_PATH", str(tmp_path / "access.log"))
    monkeypatch.setattr("maas_tokenizer.api.SerialScheduler", scheduler_factory)
    monkeypatch.setattr("maas_tokenizer.api._service.count", fail)

    with pytest.raises(RuntimeError, match="warmup failed"):
        with TestClient(app):
            pass

    assert created[0].started is True
    assert created[0].closed is True
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/pytest \
  tests/test_api.py::test_startup_warms_glm_5_2_before_serving \
  tests/test_api.py::test_warmup_failure_prevents_startup_and_closes_scheduler \
  -q
```

Expected: both tests fail because the lifespan does not yet call `_service.count()` during startup; the failure-cleanup test observes no warmup exception.

- [ ] **Step 4: Implement the minimal lifespan warmup**

Add the fixed request near the global service:

```python
_service = TokenCountService()
_WARMUP_REQUEST: dict[str, Any] = {
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": "warmup"}],
}
```

Move the scheduler startup inside the cleanup-protected section, submit the real service call before `yield`, and keep all existing cleanup:

```python
    application.state.scheduler = scheduler
    application.state.access_logger = logger
    await scheduler.start()
    try:
        await scheduler.submit(lambda: _service.count(_WARMUP_REQUEST))
        yield
    finally:
        await scheduler.close()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_api.py -q
```

Expected: all API tests pass, including startup warmup and failure cleanup.

- [ ] **Step 6: Update operating documentation**

Replace the README statement that all models load on first request with this exact behavior:

```text
服务启动阶段会通过正式计数链路预热 `glm-5.2`，预热成功后才开始接收流量；其他模型第一次收到请求时才校验并加载其 tokenizer/template。所有已加载模型之后都在进程内按 profile 缓存。
```

- [ ] **Step 7: Run targeted and full regression verification**

Run:

```bash
.venv/bin/pytest tests/test_api.py tests/test_scheduler.py tests/test_cache.py -q
.venv/bin/pytest -q
```

Expected: both commands exit 0; the default suite satisfies the configured 85% coverage floor when run with the project's normal coverage command if coverage is not enabled by default.

- [ ] **Step 8: Inspect and commit only warmup changes**

Run:

```bash
git diff --check
git status --short
git diff -- src/maas_tokenizer/api.py tests/test_api.py README.md
git add src/maas_tokenizer/api.py tests/test_api.py README.md
git commit -m "feat: warm GLM-5.2 during application startup"
```

Do not add the pre-existing untracked `.dockerignore` or `Dockerfile`.
