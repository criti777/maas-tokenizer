# MaaS Token Count Service Design

## Goal

Build a small production service that accepts an OpenAI-style chat completion
request, reproduces the selected vLLM text preprocessing path, and returns only
the number of prompt token IDs.

The service does not load model weights, run inference, process datasets, or
return rendered text, token IDs, hashes, case IDs, or oracle diagnostics.

## HTTP contract

The service exposes one endpoint:

```http
POST /v1/token-count
Content-Type: application/json
```

The request body is an OpenAI Chat Completions-style request. At minimum it
contains `model` and `messages`; supported optional fields include `tools`,
`tool_choice`, `reasoning_effort`, `add_generation_prompt`,
`continue_final_message`, `documents`, `chat_template`,
`chat_template_kwargs`, and `chat_template_content_format`.

Example:

```json
{
  "model": "glm-5.2",
  "messages": [{"role": "user", "content": "你好"}],
  "reasoning_effort": "high"
}
```

Success response:

```json
{"token_count": 18}
```

No other success fields are returned.

Errors use FastAPI's `detail` envelope:

```json
{
  "detail": {
    "stage": "request_validation",
    "type": "validation_error",
    "message": "..."
  }
}
```

Status mapping:

- `400`: request, message, template, render, or encode error;
- `404`: unknown model profile;
- `422`: malformed JSON or FastAPI body validation failure;
- `501`: media input requires a multimodal processor and cannot be counted by
  this text-only service;
- `500`: missing/corrupt local assets or an unexpected internal error.

## Processing pipeline

Every request follows one public path:

```text
FastAPI request
  -> resolve model profile
  -> get or lazily construct cached model processor
  -> validate the OpenAI-style request
  -> normalize messages with the extracted vLLM behavior
  -> route to the profile renderer
  -> render prompt text
  -> encode with the pinned tokenizer
  -> return len(token_ids)
```

The caller never selects a renderer directly.

Renderer routing is fixed by the model registry:

- DeepSeek V3, Kimi K2.6, GLM 5.1, GLM 5.2, and MiniMax M2.7 use the extracted
  Hugging Face renderer path;
- DeepSeek V3.2 uses the extracted DeepSeek V3.2 renderer;
- DeepSeek V4 uses the extracted DeepSeek V4 renderer.

The service supports only registered model names and aliases. There is no
silent fallback for unknown models.

## Model loading and caching

Tokenizer and template assets are loaded lazily on the first request for a
model. The resulting processor is cached in memory for the life of the process.

Cache creation is concurrency-safe: concurrent first requests for the same
model construct one processor and then share it. Different model profiles may
be loaded independently. Failed construction is not cached, so a corrected
asset deployment can succeed on a later request.

The service default port may be configured as `8080`. Tests use FastAPI's
in-process client and never bind that port.

## Multimodal boundary

This is a text token-count service. It reproduces only text produced before a
model-specific multimodal processor.

- A profile whose official text template can deterministically render media
  placeholders may count those textual placeholders.
- A profile that needs a processor to determine its input sequence returns
  HTTP `501` with stage `processor_required`.
- The service never downloads images, decodes media, computes pixel values,
  runs a vision/audio encoder, or estimates expanded media tokens.
- Media content is never silently removed or replaced with an invented generic
  separator.

## Repository scope

The production repository contains:

```text
pyproject.toml
README.md
src/maas_tokenizer/
vendor/vllm/extracted/
models/profiles.json
models/manifests/
model_assets/
tests/
THIRD_PARTY_NOTICES.md
```

It excludes:

- JSONL helpers;
- request/result datasets;
- batch generators and verifiers;
- rendered-text and token-ID hashing;
- oracle result contracts, case IDs, and dataset diagnostics;
- the complete vLLM source tree;
- model weights and inference dependencies;
- development-history documents unrelated to operating the service.

The copied vLLM-derived files retain their SPDX headers. One concise
`THIRD_PARTY_NOTICES.md` records the Apache-2.0 upstream attribution. Repeated
model-repository license files are not part of the runtime package.

## Python package and dependencies

The package requires Python 3.11 or newer. Direct runtime dependencies are
limited to FastAPI/Uvicorn and the libraries required by the pinned tokenizer
and template paths: Transformers, Tokenizers, Jinja, Pydantic, and tiktoken.
No PyTorch, CUDA, vLLM wheel, model weights, dataset library, or PyArrow is
required.

Exact versions are pinned in `pyproject.toml`. Test-only dependencies include
pytest, pytest-cov, and the HTTP client used by FastAPI's in-process test
client.

## Testing and coverage

Implementation follows test-driven development. Tests cover the public HTTP
contract and the production branches required to meet pipeline coverage:

- success response contains only `token_count`;
- unknown model and malformed request handling;
- tools, thinking/reasoning, continuation, and developer-role behavior;
- all five Hugging Face profiles;
- DeepSeek V3.2 and V4 specialized renderers;
- multimodal placeholder/processor-required boundary;
- asset integrity failures;
- lazy cache reuse and concurrency-safe first load;
- model-selective pytest execution so CI can run one model or all models.

Golden assertions compare exact token counts and, inside renderer unit tests,
exact token IDs where necessary to prevent equal-length-but-different-encoding
regressions. The public API still returns only the count.

The primary commands are:

```bash
pytest -q
pytest --model glm-5.2 -q
pytest --model all -q
pytest --cov=maas_tokenizer --cov=vendor.vllm.extracted --cov-report=term-missing
```

## Deployment

The ASGI application is exposed as `maas_tokenizer.api:app`. A normal process
can be started with:

```bash
uvicorn maas_tokenizer.api:app --host 0.0.0.0 --port 8080
```

The service assumes `models/` and `model_assets/` are available in the deployed
artifact. Asset paths may be overridden by configuration for company packaging
without changing the HTTP contract.
