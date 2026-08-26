# Request Compatibility Union Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a strict, non-destructive union of equivalent Thinking and continuation request fields before tokenizer rendering.

**Architecture:** Add a focused compatibility module that deep-copies and normalizes the raw request mapping before Pydantic validation. `TokenCountService` consumes the normalized mapping, while existing `RequestProcessingError` handling preserves the public HTTP 400 error contract.

**Tech Stack:** Python 3.11, Pydantic v2, FastAPI, pytest.

## Global Constraints

- Preserve all existing equivalent fields and fill missing representations; never delete a representation.
- Reject contradictory explicit values instead of applying precedence.
- Do not mutate the caller-owned mapping or nested objects.
- Do not change fixed model templates, renderer selection, or DeepSeek V4 continuation behavior.
- Use the existing `request_processing_error` HTTP 400 response contract.

---

### Task 1: Strict compatibility normalizer

**Files:**
- Create: `src/maas_tokenizer/request_compat.py`
- Create: `tests/test_request_compat.py`

**Interfaces:**
- Consumes: `collections.abc.Mapping[str, Any]` and `RequestProcessingError`.
- Produces: `normalize_compatibility_fields(request: Mapping[str, Any]) -> dict[str, Any]`.

- [ ] **Step 1: Write failing Thinking tests**

Add parameterized tests proving that each single upstream representation (`thinking`, `chat_template_kwargs.thinking`, or `chat_template_kwargs.enable_thinking`) produces all three equal fields, preserves unrelated kwargs, and does not mutate the input. Add failure cases for non-boolean switches, contradictory direct switches, and contradiction with `reasoning_effort` (`none` means false; every other supported non-null value means true).

- [ ] **Step 2: Run the Thinking tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_request_compat.py -v
```

Expected: collection/import failure because `maas_tokenizer.request_compat` does not exist.

- [ ] **Step 3: Implement minimal Thinking normalization**

Create:

```python
def normalize_compatibility_fields(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = deepcopy(dict(request))
    # Collect explicit booleans, reject disagreement, then fill:
    # normalized["thinking"]
    # normalized["chat_template_kwargs"]["thinking"]
    # normalized["chat_template_kwargs"]["enable_thinking"]
    return normalized
```

Raise `RequestProcessingError("conflicting thinking options")` for disagreement and a field-specific `RequestProcessingError` when a direct switch is not boolean.

- [ ] **Step 4: Run the Thinking tests and verify GREEN**

Run the same focused pytest command and confirm all Thinking cases pass.

- [ ] **Step 5: Write failing continuation tests**

Add tests proving:

- final assistant `prefix=true` fills `continue_final_message=true` and `add_generation_prompt=false`;
- the exact top-level continuation pair fills final assistant `prefix=true`;
- the union remains unchanged and preserves unrelated message fields;
- `prefix=false` alone does not synthesize top-level fields;
- contradictory explicit fields fail;
- continuation without a final assistant fails;
- non-final `prefix=true` and non-boolean Prefix/control values fail;
- the original nested messages remain unchanged.

- [ ] **Step 6: Run continuation tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_request_compat.py -v
```

Expected: continuation cases fail because only Thinking normalization exists.

- [ ] **Step 7: Implement minimal continuation normalization**

Inspect raw field presence before protocol defaults are applied. Establish positive continuation only from final assistant `prefix=true` or the exact `continue_final_message=true` / `add_generation_prompt=false` pair. Fill all three fields, reject contradictions with `conflicting message continuation options`, and reject an invalid final role with `continuation requires a final assistant message`.

- [ ] **Step 8: Run normalizer tests and verify GREEN**

Run the focused test file and confirm all compatibility cases pass.

- [ ] **Step 9: Commit the isolated normalizer**

```bash
git add src/maas_tokenizer/request_compat.py tests/test_request_compat.py
git commit -m "feat: normalize tokenizer compatibility fields"
```

### Task 2: Service integration and API contract

**Files:**
- Modify: `src/maas_tokenizer/service.py`
- Modify: `tests/test_service.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `normalize_compatibility_fields(request)` from Task 1.
- Produces: all renderer paths receive the normalized request; compatibility failures retain HTTP 400 `request_processing_error`.

- [ ] **Step 1: Write failing service integration test**

Inject a recording registry/renderer or equivalent lightweight test fixture and prove `TokenCountService.count()` sends the complete Thinking/continuation union into protocol parsing and rendering rather than the upstream-reduced form.

- [ ] **Step 2: Write failing API error test**

POST a request containing contradictory Thinking or continuation values and assert:

```json
{
  "error_code": "request_processing_error",
  "error_msg": "conflicting thinking options"
}
```

with HTTP 400.

- [ ] **Step 3: Run integration tests and verify RED**

```bash
.venv/bin/pytest tests/test_service.py tests/test_api.py -v
```

Expected: new integration assertions fail because the service does not call the normalizer.

- [ ] **Step 4: Integrate the normalizer**

In `TokenCountService.count()`, call:

```python
request_dict = normalize_compatibility_fields(request)
```

before model resolution and `ChatCompletionRequest.model_validate()`.

- [ ] **Step 5: Run focused integration tests and verify GREEN**

Run the service/API tests and confirm the existing and new cases pass.

- [ ] **Step 6: Run model-specific and full regression suites**

```bash
.venv/bin/pytest tests/test_request_compat.py tests/test_service.py tests/test_api.py tests/test_tools_and_thinking.py tests/test_extracted_paths.py -v
.venv/bin/pytest -q
```

Expected: all default tests pass; tests requiring unavailable optional environments remain skipped under their existing markers.

- [ ] **Step 7: Review and commit integration**

Run `git diff --check`, inspect the complete diff, then commit:

```bash
git add src/maas_tokenizer/service.py tests/test_service.py tests/test_api.py
git commit -m "feat: restore request compatibility union"
```
