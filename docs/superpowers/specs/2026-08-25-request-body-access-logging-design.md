# Request Body Access Logging Design

## Goal

Allow operators to opt in to logging parsed `/tokenizer` request bodies in the
existing access log during integration testing, without changing the API or
adding unbounded log volume in production.

## Configuration

- `TOKENIZER_LOG_REQUEST_BODY` controls request-body logging and defaults to
  `false`.
- `TOKENIZER_LOG_REQUEST_BODY_MAX_BYTES` controls the maximum logged UTF-8 JSON
  size and defaults to `65536` bytes (64 KiB).
- The maximum must be positive. Invalid configuration prevents application
  startup, consistent with the existing access-log settings.

## Logging Behavior

FastAPI's parsed request mapping is serialized as compact, UTF-8 JSON. No
request headers are included and fields are not redacted.

When logging is disabled, the current access-log format remains unchanged and
the request body is not serialized for logging.

When enabled and the compact JSON is at most the configured limit, the access
line gains:

```text
request_body_bytes=286 request_body={"model":"glm-5.2","messages":[...]}
```

When enabled and the body exceeds the limit, the line gains:

```text
request_body_bytes=10000286 request_body=<omitted_too_large>
```

The existing sanitizer keeps each access record on one physical line. Invalid
JSON is rejected by FastAPI before a parsed mapping reaches the endpoint, so
its full body is not recorded.

## Data Flow

1. Application startup loads the request-body logging configuration together
   with the existing access-log configuration.
2. The `/tokenizer` endpoint records a log-safe representation on request
   state before scheduler admission.
3. The existing middleware emits the representation in the same access record
   for success, rejection, or handled failure.
4. The existing stdout and rotating-file handlers receive the unchanged
   single-line record.

## Security and Operations

The feature is disabled by default because message content may be sensitive and
access records are written both to stdout and to the rotating file. Operators
must enable it only where retaining request content is acceptable. Existing
100 MiB file rotation and five-backup defaults remain unchanged.

## Verification

Tests cover the disabled default, enabled small-body logging, oversized-body
omission, UTF-8 byte measurement, one-line sanitization, and invalid environment
configuration. The full existing test suite must remain green.
