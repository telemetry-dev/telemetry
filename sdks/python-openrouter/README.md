# telemetry-dev-openrouter

OpenRouter SDK instrumentation for telemetry.dev. It wraps the official `openrouter` Python SDK
and emits telemetry.dev generation and embedding spans through `telemetry-dev`.

## Install

```sh
pip install telemetry-dev-openrouter
```

Initialize the core SDK first:

```py
import telemetry_dev

telemetry_dev.init(
    api_key="td_live_...",
    service_name="my-service",
)
```

## Per-client wrapping

```py
from openrouter import OpenRouter
from telemetry_dev_openrouter import wrap_open_router

client = wrap_open_router(OpenRouter(api_key="sk-or-v1-..."))

client.chat.send(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a joke about OpenTelemetry"}],
)
```

Use this when you want explicit control over which clients are instrumented. The same wrapped
client supports synchronous methods and their `*_async` counterparts.

## Global instrumentation

```py
from openrouter import OpenRouter
from telemetry_dev_openrouter import instrument_openrouter, uninstrument_openrouter

instrument_openrouter()
client = OpenRouter(api_key="sk-or-v1-...")

try:
    client.responses.send(model="openai/gpt-4o-mini", input="Explain OpenTelemetry briefly")
finally:
    uninstrument_openrouter()
```

Use this as the app-wide one-liner at startup when all OpenRouter clients should be instrumented.
Repeated wrapping or global instrumentation is safe, and `uninstrument_openrouter()` restores the
original SDK methods.

## Instrumented surfaces

Synchronous and asynchronous variants are covered:

- `client.chat.send(...)` and `client.chat.send_async(...)`
- `client.responses.send(...)` and `client.responses.send_async(...)`
- `client.embeddings.generate(...)` and `client.embeddings.generate_async(...)`

All spans use `gen_ai.provider.name = "openrouter"`. Generation spans capture the request model and
input, supported sampling parameters, response model and ID, finish reasons, output messages, and
token usage. Embedding spans capture the request input, request and response models, and token
usage; embedding vectors are intentionally not captured as output.

When OpenRouter returns cost information, the integration records `gen_ai.usage.cost` from
`usage.cost`. If that field is absent, it falls back to
`usage.cost_details.upstream_inference_cost`. The cost attribute is omitted when neither value is
available.

## Streaming

Chat and Responses API streams are traced for sync and async callers. The integration accumulates
captured output, records time to first chunk, and maps terminal usage and cost when OpenRouter
provides them. Terminal Responses API events close the span, including completed, failed,
incomplete, and error events.

Responses stream output can include reconstructed output items followed by bounded
`{"type": "telemetry.dev.response_stream_event", "event_type": ..., "payload": ...}` entries for
consumed image-generation lifecycle and partial-image events, apply-patch diff events, fusion
lifecycle, analysis, and panel events, web-search lifecycle events, and debug timing events.
OpenRouter debug upstream request-body echoes are not retained. These provider-event entries share
the stream capture budget and set
`telemetry.dev.capture.truncated = true` when only a prefix can be retained.

Requests are passed to OpenRouter unchanged. The integration does not add or modify
`stream_options`, so streaming usage and cost are available only when they are present in the
stream returned by OpenRouter.

## Failure behavior

Instrumentation is fail-open when `telemetry_dev.init()` has not been called: OpenRouter requests
still execute and return their normal SDK values without exporting telemetry.

## Limitations

- Unconsumed streams end their spans only when the stream is exhausted, errors, or is closed.
- Streaming output is retained only up to the core SDK capture budget; terminal metadata and usage
  can still be recorded after the budget is reached. A span with partial captured output includes
  `telemetry.dev.capture.truncated = true`.
- Only the SDK methods listed above are instrumented in this version.

## Development

From `sdks/python-openrouter`:

```sh
uv sync
uv run ruff format .
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run pyright
```
