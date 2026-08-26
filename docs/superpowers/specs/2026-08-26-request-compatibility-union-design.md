# Request Compatibility Union Design

## Objective

Restore equivalent request fields removed by upstream XDS/Fabric and vLLM normalization so every tokenizer renderer receives the representation it understands. The service must preserve one unambiguous semantic value, add equivalent representations, and reject contradictory inputs instead of guessing.

## Scope

The compatibility layer covers:

- Thinking controls:
  - top-level `thinking`
  - `chat_template_kwargs.thinking`
  - `chat_template_kwargs.enable_thinking`
  - top-level `reasoning_effort` when it implies Thinking on or off
- Continuation controls:
  - `messages[-1].prefix`
  - top-level `continue_final_message`
  - top-level `add_generation_prompt`

It does not change model templates, tokenizer assets, renderer selection, or DeepSeek V4 continuation behavior.

## Architecture

Add one model-independent request-normalization function before `ChatCompletionRequest.model_validate()` and before renderer dispatch. The function returns a copy and never mutates the caller-owned mapping or nested values.

The service continues to resolve the model and validate/render the normalized request through the existing paths. Compatibility errors are raised as `RequestProcessingError` and therefore return HTTP 400 with the existing error envelope.

## Thinking Normalization

Collect explicit semantic values from:

- top-level `thinking`, if present;
- `chat_template_kwargs.thinking`, if present;
- `chat_template_kwargs.enable_thinking`, if present;
- `reasoning_effort`, where `none` means false and every other supported non-null value means true.

Rules:

1. Each present Thinking field must have the expected type. The three direct switches must be booleans.
2. If the collected values disagree, reject the request with `conflicting thinking options`.
3. If no Thinking value is present, leave the request unchanged.
4. If one unambiguous value exists, preserve existing fields and fill all three direct representations with the same boolean:
   - top-level `thinking`;
   - `chat_template_kwargs.thinking`;
   - `chat_template_kwargs.enable_thinking`.
5. Preserve `reasoning_effort`; do not synthesize it from a boolean because it carries more detail than an on/off switch.

This is safe for the pinned renderers: Kimi reads `thinking`, GLM reads `enable_thinking`, DeepSeek V3.2/V4 use their logical OR, and templates that do not recognize the variables ignore them.

## Continuation Normalization

A positive continuation semantic is represented by either:

- `messages[-1].prefix == true`; or
- `continue_final_message == true` together with `add_generation_prompt == false`.

Rules:

1. Only the final message may establish continuation, and it must have role `assistant`.
2. When the final assistant message has `prefix=true`, fill the top-level pair with `continue_final_message=true` and `add_generation_prompt=false`.
3. When the top-level pair has exactly those values, fill `prefix=true` on the final assistant message.
4. Preserve all three fields after normalization.
5. Do not infer normal generation from `prefix=false`; absent/default fields remain governed by the existing protocol defaults.
6. If explicit fields contradict the continuation semantic, reject the request with `conflicting message continuation options`.
7. If the top-level pair requests continuation but the final message is absent or is not an assistant message, reject the request with `continuation requires a final assistant message`.

HF renderers consume the top-level pair, while DeepSeek V3.2 consumes `prefix`. DeepSeek V4 currently ignores these three representations; adding the union therefore does not alter its current behavior.

## Error Contract

All normalization failures use the existing response form:

```json
{
  "error_code": "request_processing_error",
  "error_msg": "<specific compatibility error>"
}
```

No new HTTP status code or error response schema is introduced.

## Testing

Follow test-driven development:

1. Add focused unit tests for XDS/Fabric Thinking input, vLLM `thinking`, and vLLM `enable_thinking` input.
2. Verify all three direct Thinking fields are present and equal after normalization.
3. Verify contradictory Thinking values and conflicts with `reasoning_effort` fail.
4. Add unit tests for Prefix-only input and top-level continuation-only input.
5. Verify the three continuation fields are present after normalization.
6. Verify contradictory continuation values and a non-assistant final message fail.
7. Verify ordinary requests remain unchanged and input mappings are not mutated.
8. Add service/API integration coverage proving compatibility failures map to HTTP 400.
9. Run the focused tests and then the complete test suite.

## Non-goals

- Adding Qwen to the supported model registry.
- Changing DeepSeek V4 to interpret Prefix as `wo_eos`.
- Changing fixed model templates or token vocabularies.
- Silently selecting a precedence when callers provide contradictory values.
