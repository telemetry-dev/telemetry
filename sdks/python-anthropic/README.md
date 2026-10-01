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

`wrap_anthropic` also supports `AsyncAnthropic` and the provider clients `AnthropicBedrock`, `AnthropicBedrockMantle`, `AnthropicVertex`, `AnthropicAWS`, and `AnthropicFoundry`, with their `Async` variants.

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
- `client.messages.parse(...)`, sync and async
- `client.beta.messages.create(...)`, `stream(...)`, and `parse(...)`, sync and async
- `client.beta.messages.tool_runner(...)`: each model request in the tool loop records its own generation span

The integration maps native Anthropic request and response shapes directly into telemetry.dev fields. It does not normalize messages into another schema.

## Streaming

Native Anthropic stream events pass through unmodified. The span records time to first chunk on the first received event, merges usage from `message_start` and `message_delta`, aggregates text, tool-use JSON, and thinking blocks, and ends on stream exhaustion, close, context-manager exit, or error.

Reconstructed output is bounded to 48 KiB and 1,000 items. The configured `max_attribute_length` does not shrink these retention bounds; the core applies it to the exported attribute after the mask runs, with its `...[truncated]` marker, without setting `telemetry.dev.capture.truncated`. A stream that ends without `message_stop`, encounters a mapping failure,
or exceeds a capture bound sets `telemetry.dev.capture.truncated`. With a mask configured,
incomplete output is omitted because the mask cannot inspect the complete value. When output
capture is disabled, response IDs and models, finish reasons, usage, timing, and errors may still
be recorded.
Capture flags do not gate stop sequences, caller-supplied metadata or raw attributes, or exception
messages and stack traces; redact those separately when needed. Request tools and `tool_choice` are part of the captured input, so `capture_input=False` removes them.

`messages.stream()` starts the span when the context manager is entered, because that is when the Anthropic SDK opens the HTTP stream.

## Provider clients

Class instrumentation covers the provider clients. Their `messages` resources reuse the stable `Messages` and `AsyncMessages` classes. The Bedrock and Vertex `beta.messages` resources are separate provider classes, which are patched too: Bedrock exposes `create()` and Vertex exposes `create()` and `stream()`. Provider attribution comes from the client class, including subclasses: `aws.bedrock` for `AnthropicBedrock` and `AnthropicBedrockMantle`, and `gcp.vertex_ai` for `AnthropicVertex`. `AnthropicAWS` (Claude Platform on AWS) and `AnthropicFoundry` (Microsoft Foundry) serve the Anthropic API, so they are recorded as `anthropic`.

## Limitations

- `messages.count_tokens()`, `beta.messages.count_tokens()`, and batches are not instrumented.
- `with_raw_response` snapshots bound methods; wrap or instrument clients before creating raw-response wrappers.
- Unconsumed streams end spans only on exhaustion, close, context-manager exit, or error.
