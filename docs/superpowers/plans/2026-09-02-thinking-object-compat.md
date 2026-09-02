# Thinking Object Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `/tokenizer` to accept object-form Thinking and legacy gateway fields while preserving existing default token counts and rejecting only explicit conflicts or uninterpretable types.

**Architecture:** Keep compatibility parsing in `request_compat.py`, where all external spellings collapse into a boolean-or-unspecified value and small copied containers. Extend the extracted request model only with fields required to deliver normalized auxiliary parameters to renderers; keep model-specific interpretation in the service/renderer boundary.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, Hugging Face templates, extracted vLLM renderers, Gigatoken.

## Global Constraints

- Do not mutate caller-owned request data or copy large `messages` content.
- Missing Thinking and `type=auto` must preserve existing model defaults.
- Accept every valid form described in the approved design.
- Reject explicit `True`/`False` conflicts with `conflicting thinking options`.
- Do not copy generation-only sampling rewrites.
- Keep the existing two-field HTTP error response.

---

### Task 1: Normalize object and legacy Thinking inputs

**Files:**
- Modify: `src/maas_tokenizer/request_compat.py`
- Test: `tests/test_request_compat.py`

**Interfaces:**
- Consumes: `normalize_compatibility_fields(request: Mapping[str, Any], *, minimal_disables_thinking: bool = False) -> dict[str, Any]`.
- Produces: a copied request whose explicit effective switch is mirrored to top-level `thinking` and both chat-template boolean names; object-only auxiliary values become top-level `clear_thinking`.

- [ ] **Step 1: Write failing table-driven tests**

Cover `thinking=None`, `{}`, object types `enabled/disabled/auto`, top-level `enable_thinking`, consistent unions, `auto` plus an explicit boolean, object `clear_thinking`, invalid container/type values, and explicit conflicts.

```python
@pytest.mark.parametrize(
    ("thinking", "expected"),
    [({}, True), ({"type": "enabled"}, True), ({"type": "disabled"}, False)],
)
def test_object_thinking_is_normalized(thinking, expected):
    normalized = normalize_compatibility_fields(
        {"model": "glm-5.2", "messages": [], "thinking": thinking}
    )
    assert normalized["thinking"] is expected
    assert normalized["chat_template_kwargs"]["enable_thinking"] is expected
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_request_compat.py`

Expected: object-form and top-level legacy-field cases fail under the current boolean-only implementation.

- [ ] **Step 3: Implement minimal normalization**

Add focused helpers equivalent to:

```python
def _parse_thinking(value: Any) -> tuple[bool | None, bool | None]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return value, None
    if not isinstance(value, Mapping):
        raise RequestProcessingError("thinking must be a boolean, object, or null")
    type_value = value.get("type", "enabled")
    if type_value == "auto":
        enabled = None
    elif type_value in {"enabled", "disabled"}:
        enabled = type_value == "enabled"
    else:
        raise RequestProcessingError("thinking.type must be enabled, disabled, or auto")
    clear = value.get("clear_thinking")
    if clear is not None:
        clear = _require_bool("thinking.clear_thinking", clear)
    return enabled, clear
```

Merge only explicit boolean candidates; let `None` abstain from conflict resolution. Validate `preserve_thinking` and legacy `enable_thinking` when present. Copy `chat_template_kwargs` before mutation.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/pytest -q tests/test_request_compat.py`

Expected: all request compatibility tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/maas_tokenizer/request_compat.py tests/test_request_compat.py
git commit -m "feat(api): accept object-form Thinking options"
```

### Task 2: Deliver auxiliary Thinking parameters to renderers

**Files:**
- Modify: `vendor/vllm/extracted/protocol.py`
- Modify: `src/maas_tokenizer/service.py`
- Modify: `src/maas_tokenizer/renderers.py`
- Test: `tests/test_service.py`
- Test: `tests/test_renderers.py`
- Test: `tests/test_deepseek_models.py`
- Test: `tests/test_kimi_k3.py`

**Interfaces:**
- Consumes: normalized top-level `clear_thinking`, `preserve_thinking`, `thinking`, `enable_thinking`, and `reasoning_effort`.
- Produces: `ChatCompletionRequest.template_kwargs()` containing only explicit switches plus auxiliary template values; model renderers receive their established native representation.

- [ ] **Step 1: Write failing propagation and default-preservation tests**

```python
def test_template_kwargs_include_thinking_auxiliary_fields():
    parsed = ChatCompletionRequest.model_validate(
        {"messages": [], "clear_thinking": True, "preserve_thinking": True}
    )
    kwargs = parsed.template_kwargs(None)
    assert kwargs["clear_thinking"] is True
    assert kwargs["preserve_thinking"] is True
```

Add renderer-capture tests proving explicit booleans reach both template names, while omission injects neither. Lock current fixed counts for requests with no Thinking fields.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_service.py tests/test_renderers.py tests/test_deepseek_models.py tests/test_kimi_k3.py --model all`

Expected: auxiliary propagation cases fail; existing default-count cases remain green.

- [ ] **Step 3: Implement request-model and renderer propagation**

Add typed optional fields and include non-`None` values in template kwargs:

```python
clear_thinking: bool | None = None
preserve_thinking: bool | None = None
```

HF renderers pass these kwargs through. DeepSeek V3.2/V4 continue converting the explicit boolean to `thinking_mode`; Kimi K3 continues passing a concrete boolean only when supplied and otherwise uses its official default. Do not add sampling parameters.

- [ ] **Step 4: Verify GREEN**

Run the same focused command and expect all selected tests to pass.

- [ ] **Step 5: Commit**

```bash
git add vendor/vllm/extracted/protocol.py src/maas_tokenizer/service.py src/maas_tokenizer/renderers.py tests/test_service.py tests/test_renderers.py tests/test_deepseek_models.py tests/test_kimi_k3.py
git commit -m "feat(renderers): propagate normalized Thinking options"
```

### Task 3: Add GLM-5.2 minimal-effort override and full regression

**Files:**
- Modify: `src/maas_tokenizer/service.py`
- Modify: `README.md`
- Test: `tests/test_hf_models.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: resolved model profile and normalized request.
- Produces: GLM-5.2-only policy flag where `none/minimal` participates in normalization as an explicit disabled switch before Pydantic validation/rendering.

- [ ] **Step 1: Write failing GLM/API tests**

```python
def test_glm_5_2_minimal_effort_disables_thinking():
    service = TokenCountService()
    request = {"model": "glm-5.2", "messages": [{"role": "user", "content": "hi"}]}
    assert service.count({**request, "reasoning_effort": "minimal"}) == service.count(
        {**request, "thinking": False}
    )
```

Add API tests for successful object-form input and stable conflict/invalid-type errors.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_hf_models.py tests/test_api.py --model glm-5.2`

Expected: the GLM minimal-effort equality and object-form endpoint tests fail.

- [ ] **Step 3: Implement the scoped override and documentation**

Read and validate the raw model first, resolve its profile, then call:

```python
request_dict = normalize_compatibility_fields(
    request,
    minimal_disables_thinking=profile.profile_id == "glm-5.2",
)
```

The normalizer maps `minimal` to `False` only under this policy, so `thinking=true + reasoning_effort=minimal` follows the reference gateway's forced-disable behavior instead of raising a false conflict. Document accepted Thinking forms and conflict behavior in `README.md`.

- [ ] **Step 4: Run focused, fast, and all-model verification**

```bash
.venv/bin/pytest -q tests/test_request_compat.py tests/test_api.py tests/test_hf_models.py tests/test_deepseek_models.py tests/test_kimi_k3.py --model all
.venv/bin/pytest -q
.venv/bin/pytest -q --model all --cov=maas_tokenizer --cov-report=term-missing --cov-fail-under=85
git diff --check
```

Expected: all tests pass, coverage is at least 85%, and no whitespace errors are reported.

- [ ] **Step 5: Commit**

```bash
git add src/maas_tokenizer/service.py README.md tests/test_hf_models.py tests/test_api.py
git commit -m "feat(models): align GLM Thinking effort compatibility"
```
