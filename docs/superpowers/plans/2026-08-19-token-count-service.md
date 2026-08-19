# MaaS Token Count Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI service that accepts an OpenAI-style chat request at `POST /v1/token-count` and returns only the selected model's prompt token count.

**Architecture:** A small HTTP adapter delegates to a `TokenCountService`. The service resolves a fixed model profile, lazily builds and caches one renderer per profile, runs the extracted vLLM normalization/render path, and returns `len(token_ids)`. Pinned local tokenizer/template assets are verified before loading; datasets, JSONL, hashes, oracle result envelopes, and weights are excluded.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, Pydantic 2.12.5, Transformers 5.13.1, Tokenizers 0.22.2, Jinja 3.1.6, tiktoken 0.12.0, pytest 8.4.2, pytest-cov, HTTPX.

## Global Constraints

- Expose only `POST /v1/token-count`; success is exactly `{"token_count": N}`.
- Use an OpenAI Chat Completions-style request body and select the model from `model`.
- Support only the seven profiles registered in `models/profiles.json`; never silently fall back.
- Lazily load and process-cache tokenizer/template renderers with concurrency-safe first construction.
- Do not load model weights, PyTorch, CUDA, the vLLM wheel, datasets, JSONL helpers, or PyArrow.
- Do not return rendered text, token IDs, hashes, case IDs, or internal diagnostics.
- Preserve the extracted vLLM SPDX headers and one concise third-party attribution notice.
- Tests use FastAPI's in-process client and never bind `127.0.0.1:8080`.

---

### Task 1: Package skeleton and public HTTP contract

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/maas_tokenizer/__init__.py`
- Create: `src/maas_tokenizer/api.py`
- Create: `src/maas_tokenizer/service.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Produces: `maas_tokenizer.api.app: FastAPI`.
- Produces: `TokenCountService.count(request: Mapping[str, Any]) -> int`.
- Produces: `POST /v1/token-count -> {"token_count": int}`.

- [ ] Write an HTTP test that posts `{"model":"glm-5.2","messages":[{"role":"user","content":"你好"}]}` through `TestClient`, injects a fake service returning `18`, and asserts the response equals `{"token_count":18}` with no extra keys.
- [ ] Run `.venv/bin/pytest tests/test_api.py -q` and verify collection fails because `maas_tokenizer.api` does not exist.
- [ ] Add pinned runtime/test dependencies, package discovery, a service dependency provider, the FastAPI endpoint, and the minimal `TokenCountResponse` Pydantic model.
- [ ] Run `.venv/bin/pytest tests/test_api.py -q` and verify the HTTP contract passes.
- [ ] Commit with `feat: add token count HTTP contract`.

### Task 2: Model registry, assets, and extracted renderers

**Files:**
- Create: `src/maas_tokenizer/errors.py`
- Create: `src/maas_tokenizer/registry.py`
- Create: `src/maas_tokenizer/assets.py`
- Create: `src/maas_tokenizer/protocol.py`
- Create: `src/maas_tokenizer/renderers.py`
- Create: `vendor/__init__.py`
- Create: `vendor/vllm/__init__.py`
- Create: `vendor/vllm/extracted/*.py`
- Create: `models/profiles.json`
- Create: `models/manifests/*.json`
- Create: `model_assets/**`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `tests/test_registry.py`
- Create: `tests/test_asset_integrity.py`

**Interfaces:**
- Produces: `ModelRegistry.resolve(name: str) -> ModelProfile`.
- Produces: `verify_asset_directory(profile: ModelProfile, assets_root: Path) -> Path`.
- Produces: `build_renderer(profile: ModelProfile, asset_path: Path) -> Renderer` where `Renderer.render(request) -> list[int]`.

- [ ] Write tests for all seven canonical profiles, alias resolution, unknown-model rejection, manifest size/SHA-256 verification, and rejection of missing/untracked Python assets.
- [ ] Run the registry/asset tests and verify they fail because the modules and assets are absent.
- [ ] Migrate only the fixed profile metadata, manifests, required tokenizer/template assets, and extracted vLLM files; remove oracle hashing/result dependencies and adapt renderer output to token IDs only.
- [ ] Add the minimal third-party notice and retain SPDX headers in extracted files.
- [ ] Run the registry/asset tests and verify they pass without network access or model weights.
- [ ] Commit with `feat: add pinned model render assets`.

### Task 3: Token count service and lazy cache

**Files:**
- Modify: `src/maas_tokenizer/service.py`
- Modify: `src/maas_tokenizer/protocol.py`
- Modify: `src/maas_tokenizer/renderers.py`
- Create: `tests/test_service.py`
- Create: `tests/test_cache.py`

**Interfaces:**
- Produces: `TokenCountService(assets_root: Path, registry_path: Path)`.
- Produces: `TokenCountService.count(request: Mapping[str, Any]) -> int`.
- Produces: `TokenCountService.cached_profiles -> frozenset[str]` for observability/tests only.

- [ ] Write service tests for exact GLM-5.2 count, model/request mismatch, invalid role, and processor-required media input.
- [ ] Write a concurrency test that sends simultaneous first calls for one profile and asserts renderer construction occurs once, then write a failure test proving failed construction is not cached.
- [ ] Run the service/cache tests and verify the missing implementation failures.
- [ ] Implement strict profile resolution, Pydantic request validation, multimodal boundary checks, renderer execution, and per-profile lock-protected lazy cache.
- [ ] Run service/cache tests and verify they pass.
- [ ] Commit with `feat: count tokens with lazy model cache`.

### Task 4: HTTP error mapping and health endpoint behavior

**Files:**
- Modify: `src/maas_tokenizer/api.py`
- Modify: `src/maas_tokenizer/errors.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: typed service exceptions carrying `stage`, `type`, and `message`.
- Produces: HTTP 400/404/501/500 mappings under FastAPI's `detail` envelope.

- [ ] Add parameterized HTTP tests for unknown model (404), invalid request/render failure (400), processor required (501), malformed JSON (422), and asset/internal failure (500).
- [ ] Run the tests and verify current uncaught errors fail the expected status assertions.
- [ ] Implement one FastAPI exception handler per typed service error family; do not expose tracebacks or filesystem paths.
- [ ] Run `tests/test_api.py` and verify all status/response assertions pass.
- [ ] Commit with `feat: map token count service errors`.

### Task 5: Seven-model behavioral coverage

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_hf_models.py`
- Create: `tests/test_deepseek_models.py`
- Create: `tests/test_tools_and_thinking.py`
- Create: `tests/test_multimodal_boundary.py`

**Interfaces:**
- Produces: `pytest --model <profile>` and `pytest --model all` selection.
- Consumes: the same `TokenCountService.count()` used by HTTP.

- [ ] Add pytest model selection and golden counts for DeepSeek V3, Kimi K2.6, GLM 5.1, GLM 5.2, and MiniMax M2.7.
- [ ] Add DeepSeek V3.2/V4 tests for basic text, tools, thinking, tool-result ordering, and reasoning-effort mapping.
- [ ] Add Kimi media-placeholder success plus processor-required failures for profiles that need a multimodal processor.
- [ ] Run each profile independently, then run `pytest --model all -q`; fix only migration defects revealed by exact token assertions.
- [ ] Commit with `test: cover supported model render paths`.

### Task 6: Coverage gate, documentation, and deployment verification

**Files:**
- Create: `README.md`
- Modify: `pyproject.toml`
- Create: `tests/test_packaging.py`

**Interfaces:**
- Produces: `uvicorn maas_tokenizer.api:app --host 0.0.0.0 --port 8080` deployment command.
- Produces: reproducible install, test, model-selective test, and coverage commands.

- [ ] Add a packaging test that asserts excluded directories/modules (`datasets`, `tools`, `jsonl`, hashing/result contracts) are absent and model weights are absent.
- [ ] Run the packaging test and verify it detects any accidental migration residue.
- [ ] Write concise README sections for install, endpoint request/response, supported aliases, lazy-cache semantics, error codes, model asset requirements, tests, coverage, and port configuration.
- [ ] Run `pytest --model all -q` and `pytest --cov=maas_tokenizer --cov=vendor.vllm.extracted --cov-report=term-missing`; add focused tests for uncovered production branches rather than excluding them from coverage.
- [ ] Start Uvicorn on an automatically selected non-8080 localhost port, issue one real HTTP request, verify `{"token_count":N}`, and stop only that test process.
- [ ] Run `git diff --check`, confirm no weights/datasets/JSONL/hash code, review the final diff, commit with `docs: document token count service`, and push `main`.
