# telemetry-dev-openai

OpenAI SDK instrumentation for telemetry.dev. It wraps the official `openai` Python SDK and emits telemetry.dev generation and embedding spans through `telemetry-dev`.

## Install

```sh
pip install telemetry-dev-openai
```

Initialize the core SDK first:

```py
import telemetry_dev

telemetry_dev.init(
    api_key="td_live_...",
    base_url="http://localhost:4318",
    service_name="my-service",
)
```

## Per-client wrapping

```py
from openai import OpenAI
from telemetry_dev_openai import wrap_openai

client = wrap_openai(OpenAI())

client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a joke about OpenTelemetry"}],
)
```

Use this when you want explicit control over which sync or async clients are instrumented.

## Global instrumentation

```py
from openai import OpenAI
from telemetry_dev_openai import instrument_openai, uninstrument_openai

instrument_openai()
client = OpenAI()

try:
    client.responses.create(model="gpt-4o-mini", input="Tell me a joke about OpenTelemetry")
finally:
    uninstrument_openai()
```

Use this as the app-wide one-liner at startup when all OpenAI clients should be instrumented.

## Instrumented surfaces

Sync and async variants are covered:

- `client.chat.completions.create(...)`
- `client.chat.completions.parse(...)`
- `client.chat.completions.stream(...)`
- `client.responses.create(...)`
- `client.responses.parse(...)`
- `client.responses.stream(...)` when starting a new response
- `client.embeddings.create(...)`

The integration maps native OpenAI request/response shapes directly into telemetry.dev fields. It does not normalize messages into another schema.

## Streaming

Chat completion streams are traced. Requests are sent unchanged by default, so token usage is only captured when the caller sets `stream_options={"include_usage": True}` themselves. Pass `inject_stream_usage=True` to `wrap_openai` or `instrument_openai` to inject it automatically; the synthetic usage-only chunk is then hidden from the caller. Injection is opt-in because some providers reject `stream_options` — for example Azure OpenAI "on your data" (`data_sources`) returns 400 for it while plain `stream=True` works.

```py
client = wrap_openai(OpenAI(), inject_stream_usage=True)
```

Responses API streams are traced through `responses.create(stream=True)`; terminal `response.completed`, `response.failed`, and `response.incomplete` events close the span.

## Embeddings

Embedding calls emit `gen_ai.operation.name = "embeddings"`, request model/input, response model, and token usage. Embedding vectors are intentionally not captured as output.

## Provider detection

`wrap_openai(AzureOpenAI(...))` and `wrap_openai(AsyncAzureOpenAI(...))` record provider `azure.ai.openai`. Global class instrumentation detects Azure from the resource client when available.

OpenAI clients configured with `https://openrouter.ai/api/v1` as their base URL record provider `openrouter`. The same detection applies to OpenRouter subdomains; unrelated hosts containing `openrouter.ai` are not matched.

## Limitations

- `with_raw_response` snapshots bound methods on first access in the OpenAI Python SDK. Call `wrap_openai()` or `instrument_openai()` before accessing `with_raw_response` if those methods need instrumentation.
- `responses.stream(response_id=...)` resumes an existing response through `retrieve()`, which is not instrumented in this version.
- Unconsumed streams end their spans only when the stream is exhausted, errors, or is closed.
