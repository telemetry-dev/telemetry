# telemetry-dev-anthropic

Anthropic Claude SDK instrumentation for telemetry.dev. It wraps the official `anthropic` Python SDK and emits telemetry.dev generation spans through `telemetry-dev`.

## Install

```sh
pip install telemetry-dev-anthropic
```

Initialize the core SDK first:

```python
import telemetry_dev

telemetry_dev.init(
    api_key="td_live_...",
    service_name="my-service",
)
```

## Per-client wrapping

```python
from anthropic import Anthropic
from telemetry_dev_anthropic import wrap_anthropic

client = wrap_anthropic(Anthropic())
message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a joke about OpenTelemetry"}],
)
```

`wrap_anthropic` also supports `AsyncAnthropic`, `AnthropicBedrock`, `AsyncAnthropicBedrock`, `AnthropicVertex`, and `AsyncAnthropicVertex` clients.

## Global instrumentation

```python
from anthropic import Anthropic
from telemetry_dev_anthropic import instrument_anthropic, uninstrument_anthropic

instrument_anthropic()
try:
    client = Anthropic()
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )
finally:
    uninstrument_anthropic()
```

## Instrumented surfaces

- `client.messages.create(...)`
- `client.messages.create(..., stream=True)`
- `client.messages.stream(...)` context managers, sync and async

The integration maps native Anthropic request and response shapes directly into telemetry.dev fields. It does not normalize messages into another schema.

## Streaming

Native Anthropic stream events pass through unmodified. The span records time to first chunk on the first received event, merges usage from `message_start` and `message_delta`, aggregates text, tool-use JSON, and thinking blocks, and ends on stream exhaustion, close, context-manager exit, or error.

`messages.stream()` starts the span when the context manager is entered, because that is when the Anthropic SDK opens the HTTP stream.

## Bedrock and Vertex

Class instrumentation covers Bedrock and Vertex clients because the Anthropic SDK reuses the same `Messages` and `AsyncMessages` resource classes. Provider attribution is recorded as `aws.bedrock` or `gcp.vertex_ai` when the client class identifies those runtimes.

## Limitations

- `client.beta.messages` is not instrumented.
- `messages.parse()` and `messages.count_tokens()` are not instrumented.
- `with_raw_response` snapshots bound methods; wrap or instrument clients before creating raw-response wrappers.
- Unconsumed streams end spans only on exhaustion, close, context-manager exit, or error.
