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
- `client.images.generate(...)`, `edit(...)`, and `create_variation(...)`
- `client.audio.speech.create(...)`, `transcriptions.create(...)`, and `translations.create(...)`
- `client.batches.create(...)`, `retrieve(...)`, and `cancel(...)`

The integration maps native OpenAI request/response shapes directly into telemetry.dev fields. It does not normalize messages into another schema.

## Streaming

Chat completion streams are traced. Requests are sent unchanged by default, so token usage is only captured when the caller sets `stream_options={"include_usage": True}` themselves. Pass `inject_stream_usage=True` to `wrap_openai` or `instrument_openai` to inject it automatically; the synthetic usage-only chunk is then hidden from the caller. Injection is opt-in because some providers reject `stream_options` — for example Azure OpenAI "on your data" (`data_sources`) returns 400 for it while plain `stream=True` works.

Streamed chat output and finish reasons use separate budgets of up to 48 KiB and 1,000 items; The configured `max_attribute_length` does not shrink these retention bounds; the core applies it to the exported attribute after the mask runs, with its `...[truncated]` marker, without setting `telemetry.dev.capture.truncated`. When present, `gen_ai.output.messages` contains bounded partial output. If output and/or finish-reason capture is incomplete, including when a choice ends without `finish_reason`, `telemetry.dev.capture.truncated` is `true`. With a mask configured, incomplete output is omitted because the mask cannot inspect the complete value. When output capture is disabled or omitted, response IDs and models, finish reasons, usage, timing, and errors may still be recorded.

Capture flags do not gate stop sequences, caller-supplied metadata or raw attributes, or exception messages and stack traces; redact those separately when needed.

```py
client = wrap_openai(OpenAI(), inject_stream_usage=True)
```

Responses API streams are traced through `responses.create(stream=True)` and `responses.stream(response_id=...)`; terminal `response.completed`, `response.failed`, and `response.incomplete` events close the span.

Responses and transcription streams follow the same rule. A Responses stream that ends without a terminal snapshot, ends with an `error` event, fails to map an event, or has a terminal snapshot over the limit sets `telemetry.dev.capture.truncated`; it keeps the last snapshot that fit, unless a mask is configured, in which case output is omitted. Transcription text has its own retention limit of up to 64 KiB with no item limit. A complete non-streamed transcript is passed to the core whole. A transcription stream that ends before `transcript.text.done` (an error, early close, or end of stream) or whose text exceeds the limit is flagged the same way and keeps its bounded partial text only when no mask is configured.

## Embeddings

Embedding calls emit `gen_ai.operation.name = "embeddings"`, request model/input, response model, and token usage. Embedding vectors are intentionally not captured as output.

## Provider detection

`wrap_openai(AzureOpenAI(...))` and `wrap_openai(AsyncAzureOpenAI(...))` record provider `azure.ai.openai`. Global class instrumentation detects Azure from the resource client when available.

OpenAI clients configured with `https://openrouter.ai/api/v1` as their base URL record provider `openrouter`. The same detection applies to OpenRouter subdomains; unrelated hosts containing `openrouter.ai` are not matched.

OpenAI-compatible URLs on Groq, xAI, DeepSeek, Together, and Fireworks domains (including their subdomains) record the corresponding provider. Other compatible endpoints retain the `openai` default.

## Limitations

- `with_raw_response` and `with_streaming_response` snapshot bound methods on first access in the OpenAI Python SDK. Call `wrap_openai()` or `instrument_openai()` before accessing either helper if those methods need instrumentation.
- Unconsumed streams end their spans only when the stream is exhausted, errors, or is closed.
- Binary image and audio bodies are never captured, including binary fields nested in Responses API input and output. Speech download/streaming response spans cover request creation, not later byte consumption. Transcription stream spans remain open through stream consumption and record terminal or bounded partial transcript text and usage.
- Realtime is not wrapped: OpenAI Python 2.x exposes an async WebSocket connection (`realtime.connect`), not an event emitter with a stable listener lifecycle, so an explicit event-emitter wrapper would be misleading.
- Videos (`client.videos`) are not instrumented yet; video create and retrieve calls produce no spans. The TypeScript integration traces them.
- Batch list pagination is not instrumented so the SDK's synchronous `AsyncPaginator` remains directly usable with `async for`.
