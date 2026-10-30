# telemetry-dev

telemetry.dev SDK for Python — OpenTelemetry-native GenAI tracing, logs, and metrics. Thin
ergonomic functions over OTel spans (`gen_ai.*` semantic conventions), exported as OTLP
protobuf to the telemetry.dev ingest.

Docs: https://telemetry.dev/docs/sdk/python

## Install

```sh
pip install telemetry-dev
# or
uv add telemetry-dev
```

Requires Python >= 3.10.

## Quickstart

```python
import telemetry_dev
from telemetry_dev import log, observe, propagate_attributes, start_span, update_current_span

telemetry_dev.init()  # reads TELEMETRY_DEV_API_KEY from the environment

@observe  # arguments -> input, return value -> output, errors captured + re-raised
def lookup_weather(city: str) -> dict:
    return {"forecast": "sunny"}

with propagate_attributes(user_id="user_123", session_id="session_456"):
    with start_span(
        "chat gpt-4o",
        type="generation",
        model="gpt-4o",
        provider="openai",
        input=[{"role": "user", "content": "Plan a day trip"}],
    ):
        log("calling the model")
        update_current_span(
            output=[{"role": "assistant", "content": "Here you go..."}],
            usage={"input_tokens": 11, "output_tokens": 7},
            finish_reason="stop",
        )

    lookup_weather("Kyoto")

telemetry_dev.flush()
```

The SDK fails open: without an API key every call is a silent no-op, and internal errors are
routed to the `on_error` hook / `telemetry_dev` logger — never raised into your code.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELEMETRY_DEV_API_KEY` | — | Ingest key (`td_live_...`). Absent = SDK is a no-op. |
| `TELEMETRY_DEV_BASE_URL` | `https://api.telemetry.dev` | Base URL for telemetry export and `get_prompt` (trailing slashes stripped). |
| `TELEMETRY_DEV_ENVIRONMENT` | `production` | Deployment environment label. |
| `OTEL_SERVICE_NAME` | `unknown_service` | Service name on every trace. |
| `OTEL_TRACES_SAMPLER` | `parentbased_always_on` | OTel trace sampling policy, which includes session roots. |
| `OTEL_TRACES_SAMPLER_ARG` | `1.0` | Probability for `traceidratio` / `parentbased_traceidratio`. |

Explicit `init()` arguments take precedence over environment variables.

Earlier versions sent data to `https://ingest.telemetry.dev` by default. That hostname reaches the
same service and stays supported: if a firewall allows only it, set `TELEMETRY_DEV_BASE_URL` or
`base_url` to it.

Root spans remain independent by default. `telemetry_dev.init(session_mode="process")` groups
otherwise uncorrelated spans and logs using a fresh opaque UUID for each enabled SDK
initialization. Explicit and propagated session IDs take precedence, and valid parent trace
context is preserved. The generated session ends when the client is shut down or replaced; a
later `init()` generates a new ID. It is not persisted across processes.

## API reference

| Name | Description |
| --- | --- |
| `init(**options) -> Client` | Initialize the SDK (see options below). Calling again replaces the previous client. |
| `@observe` / `@observe(name=, type=, capture_input=, capture_output=, attributes=)` | Wrap a sync/async function (or generator) in a span. Arguments become `input` (param-name dict, `self`/`cls` dropped), the return value becomes `output`, exceptions are captured and re-raised. |
| `start_span(name, *, type="span", ...) -> SpanHandle` | Start a span. `with` activates it in the current context; without `with` it is a detached handle you must `.end()`. |
| `SpanHandle.update(**fields)` / `.end(**fields, end_time=)` / `.record_output_chunk(timestamp_ms=None)` / `.traceparent()` | Update attributes, end, record arrival of a non-empty output chunk, or read the W3C traceparent. |
| `SpanHandle.serialize_content(value, key) -> str \| None` | Prepare content an integration writes outside `input` and `output`, such as an error message or a raw attribute, the way `input` and `output` are prepared: the configured `mask` with `key`, JSON serialization (strings pass through), then truncation to `max_attribute_length`. Returns `None` when the mask raises (reported through `on_error`) or the value cannot be serialized; write nothing then. Does not check `capture_input` / `capture_output`. Integrations that support older cores should treat a missing method as a configured mask. |
| `update_current_span(**fields)` | Apply the update field set to the currently active span (no-op without one). |
| `propagate_attributes(*, user_id=, session_id=, metadata=)` | Context manager stamping `user.id` / `gen_ai.conversation.id` / `td.metadata.*` on every span and log record started inside (threads/asyncio included via contextvars). |
| `log(message, *, level="info", event_name=None, attributes=None)` | Emit an OTLP log record to `/v1/logs`, correlated with the current trace. Levels: `debug`/`info`/`warn`/`error` (`"warning"` is accepted as an alias of `warn`). |
| `get_traceparent() -> str \| None` | W3C traceparent of the current context. |
| `get_prompt(name, *, label=, version=, type="chat", cache_ttl_seconds=60, fallback=, api_key=, base_url=)` / `get_prompt_async(...)` | Fetch a prompt by label or version; see [Prompts](#prompts). |
| `with_prompt(prompt)` | Context manager recording the prompt on each model call started inside. |
| `flush(timeout_s=10.0)` / `shutdown(timeout_s=10.0)` | Force-flush / tear down traces + logs + metrics. Shutdown also runs atexit unless `disable_atexit=True`. |
| `TelemetrySpanProcessor` / `telemetry_dev.otel.create_telemetry_span_exporter` | Bring-your-own-OTel helpers (below). |
| `MaskContext`, `Usage`, `SpanHandle`, `Client`, `NOT_GIVEN` | Supporting types. |

`record_output_chunk()` measures intervals at the point the consumer pulls each chunk. If a
consumer waits between pulls, that delay is included and cannot be separated from provider latency.
The exact `gen_ai.client.operation.time_per_output_chunk` histogram is supported by the built-in
OTLP metric exporter. OpenTelemetry Python does not expose an aggregate-input API for arbitrary
metric readers, so supplying `metric_reader=` disables this histogram. Duration, token usage, and
time-to-first-chunk metrics remain available; the first accepted streamed span with two or more
recorded chunks reports this limitation once through `on_error` and the SDK logger.

Provider integrations feature-detect `record_output_chunk`. Older core SDKs without that method
continue delivering streams and tracing normally, but do not emit chunk-interval metrics. Upgrade
the core SDK with provider integrations to enable the new metric.

### Span types

`type=` maps to `gen_ai.operation.name`:

| `type` | operation | input / output attributes |
| --- | --- | --- |
| `"span"` (default) | `function` | `gen_ai.input.messages` / `gen_ai.output.messages` |
| `"generation"` | `chat` | `gen_ai.input.messages` / `gen_ai.output.messages` |
| `"tool"` | `execute_tool` | `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` |
| `"agent"` | `invoke_agent` | `gen_ai.input.messages` / `gen_ai.output.messages` |
| `"embedding"` | `embeddings` | `gen_ai.input.messages` / `gen_ai.output.messages` |

### Span fields (start/update/end)

`input`, `output`, `model`, `provider`, `system_instructions`, `response_model`, `response_id`,
`output_type`, `finish_reason`, `usage` (dict with `input_tokens`, `output_tokens`,
`total_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`,
`reasoning_output_tokens`, `text_input_tokens`, `text_output_tokens`, `image_input_tokens`,
`image_output_tokens`, `audio_input_tokens`, `audio_output_tokens`,
`text_cache_read_input_tokens`, `image_cache_read_input_tokens`, and
`audio_cache_read_input_tokens`), `cost_usd`, `temperature`, `top_p`, `top_k`, `max_tokens`,
`stop_sequences`, `seed`, `frequency_penalty`, `presence_penalty`, `time_to_first_chunk_ms`,
`tool_name`, `tool_call_id`, `tool_description`, `agent_name`, `agent_id`, `metadata`
(→ `td.metadata.*`, this span only), `attributes` (raw escape hatch, merged last), `error`.
`start_span` additionally accepts `parent` (traceparent string, OTel `Context`, or
`SpanContext`), `start_time`, and per-call `capture_input` / `capture_output` overrides;
`end()` additionally accepts `end_time`.

Core SDK error recording keeps failure status, error type, and an exception event. It includes
`exception.message` and `exception.stacktrace` only when both effective `capture_input` and
`capture_output` are true, including per-span overrides: exceptions can echo either inputs or
outputs. The policy is per span: when an error propagates to an enclosing span whose capture is on,
that span records its message and stack. Included fields pass through `mask` with their exception
attribute keys, then serialization and truncation. A truncated `exception.stacktrace` keeps its
end, where Python normally puts the innermost frame and the exception line. Mask and serialization failures drop the affected field and report generic errors
to `on_error` and SDK diagnostics without the original message, traceback, or exception chain.
The original application error is raised unchanged. These controls do not scrub error type names,
metadata, identifiers, resource attributes, or events written directly through raw OpenTelemetry APIs.

### init() options

| Option | Default | Purpose |
| --- | --- | --- |
| `api_key`, `base_url`, `environment`, `service_name` | env vars | Connection + resource settings. |
| `enabled` | `True` | `False` = hard kill switch (tests). |
| `session_mode` | `"explicit"` | `"process"` groups otherwise uncorrelated spans and logs under one generated session ID per `init()` (see [Environment variables](#environment-variables)). Other values call `on_error` and install a disabled client. |
| `register_global` | `False` | Also register the tracer provider globally. Applies a default export filter (only `telemetry_dev`-scoped spans are exported); pass `span_filter=lambda s: True` to export everything. |
| `export_mode` | `"batched"` | `"immediate"` exports synchronously per span/log (serverless). |
| `log_level` | `"warn"` | SDK diagnostics level: `debug`/`info`/`warn`/`error`/`silent`. |
| `capture_input`, `capture_output` | `True` | Global content-capture defaults. |
| `mask` | `None` | `Callable[[Any, MaskContext], Any]` redaction hook, runs before JSON serialization on input/output/log messages and `SpanHandle.serialize_content` (`MaskContext.key` is the attribute being written). Not applied to correlation identifiers. |
| `max_attribute_length` | `65536` | Non-negative integer per-content-attribute cap. Invalid values call `on_error`, do not raise into application code, and install a disabled client; truncated values use the ASCII `...[truncated]` marker, or its prefix when the cap is shorter than the marker. The cap also applies, with the marker, to string metadata and log attributes; any other string attribute is cut at the cap without a marker, except that a cap of `0` leaves those at 65536. |
| `span_filter` | `None` | Export predicate `Callable[[ReadableSpan], bool]`. |
| `sampler` | OTel environment configuration | OTel `Sampler` that overrides environment settings. Session roots use its root policy. Real parents keep their sampling decisions. |
| `on_error` | `None` | Receives internal SDK errors; repeated instrumentation failures from one stream may be coalesced into one report. The SDK never raises them into application code. A series of failed exports, for example during an ingest outage, is one report to `on_error` and the `telemetry_dev` logger. After an export reaches the ingest, the next failure is a new report. |
| `disable_atexit` | `False` | Skip the automatic atexit shutdown. |
| `timeout` | `10.0` | OTLP HTTP timeout in seconds. |
| `span_exporter`, `log_exporter`, `metric_reader` | `None` | Test seams / offline mode; any of them enables the client without an API key. |

`capture_input=False` and `capture_output=False` omit captured content. Provider integrations may
still record non-content metadata such as response IDs and models, finish reasons, usage, timing,
and error types. Capture flags do not gate stop sequences, tool descriptions and definitions
(`gen_ai.tool.description`, `gen_ai.tool.definitions`), caller-supplied metadata or raw attributes;
redact those separately when needed.

`mask` receives input and output content, system instructions, log messages, captured exception
messages and stack traces, and content that
integrations pass to `SpanHandle.serialize_content`. `MaskContext.key` names the attribute being
written. It does not receive span or propagated `metadata`, raw `attributes`, or log attributes;
redact those separately. Log masking is independent of span capture flags.

Session roots share trace IDs from the API key and session ID. The hash matches the TypeScript SDK.
The SDK uses the configured OTel sampling policy, not the synthetic parent's flags.
For example, `init(sampler=ParentBased(TraceIdRatioBased(0.1)))` samples sessions with OTel's ratio sampler.
These classes come from `opentelemetry.sdk.trace.sampling`.
The built-in Python and TypeScript ratio samplers can select different sessions because the OTel algorithms differ.

## Prompts

`get_prompt` fetches a prompt that you manage in telemetry.dev. Fetch it by its label, so that you
deploy a version when you move the label, without a change to the code:

```python
import telemetry_dev
from openai import OpenAI
from telemetry_dev_openai import wrap_openai

telemetry_dev.init()
client = wrap_openai(OpenAI())

prompt = telemetry_dev.get_prompt("support/reply", label="production")
messages = prompt.compile(customer_message=customer_message, tone=tone)

with telemetry_dev.with_prompt(prompt):
    response = client.chat.completions.create(
        model=prompt.model["name"] if prompt.model else "gpt-4o", messages=messages
    )

# Pin one version, as in a test:
pinned = telemetry_dev.get_prompt("support/reply", version=13)
```

- Without `label` and `version`, `get_prompt` fetches the `production` label. A text prompt needs
  `type="text"`; `get_prompt` raises for a prompt of the other type.
- `compile(variables)` or `compile(**variables)` puts the values in place of `{{name}}`. A missing
  value or `None` becomes an empty text, and a value that is not a string becomes JSON. In a chat
  prompt, a placeholder message becomes the list of messages in its variable. A text prompt
  compiles to a string.
- The prompt also has `model`, `parameters`, `tools`, and `response_format` of its version. The
  keys inside them are the API's (`maxTokens`, `topP`, ...).
- `get_prompt` keeps each prompt. When the copy is older than `cache_ttl_seconds` (default 60), a
  call returns it at once and fetches a new copy in a background thread. Only a call with no copy
  waits for a fetch, for 10 seconds at most, also when the network stops. Calls for the same
  prompt share one fetch. `get_prompt_async` runs the same function in a worker thread.
- If a fetch fails, `get_prompt` returns the last copy that it fetched and reports the error
  through `on_error`. With no copy, it returns the `fallback` (`version` is `None` and
  `is_fallback` is `True`). With no fallback, it raises `PromptError` whose `code` tells why, for
  example `prompt_not_found`, `label_not_found`, `insufficient_scope`, `missing_api_key`, or
  `timeout`. A response with a different status and no API error gives `http_<status>`, for
  example `http_302` for a redirect. `get_prompt` does not send the request to the location of a
  redirect, thus your key does not go to a different host. With no copy, the calls of the next 10
  seconds (or the TTL, if it is shorter) get the same result without a fetch. After a 429 or 503
  response with a `Retry-After` header, `get_prompt` does not fetch that prompt again before that
  time (one hour at most). This is the one SDK function that can raise: your app cannot run
  without its prompt.
- `get_prompt` uses the API key and base URL of an enabled `init()`, or `TELEMETRY_DEV_API_KEY`
  and `TELEMETRY_DEV_BASE_URL`, or its own `api_key` and `base_url`. It reads the same host that
  telemetry goes to, `https://api.telemetry.dev` by default. The key must have Read prompts in
  its Access on the API keys page (the `prompts:read` scope).
- When a label sends part of the traffic to a second version, each session keeps one version: the
  choice uses the session id of `propagate_attributes` (or of `session_mode="process"`). Without a
  session id, each call chooses at random.

`with_prompt(prompt)` records the prompt on each model call that starts inside it:
`td.prompt.name`, `td.prompt.version`, and `td.prompt.label` (only for a prompt fetched by label).
When input capture is on, `td.prompt.variables` holds the JSON of the variables of the last
`compile()` of that prompt object before `with_prompt` starts, or of the last `compile()` inside
it, after `mask` and truncation. If threads or tasks that run at the same time share one prompt
object, call `compile()` in `with_prompt`, or call `get_prompt` in each thread or task.
`get_prompt` returns a new object from the cache. If you do not, a thread can record the variables
of a different thread. Inside one `with_prompt`, an asyncio task records the last `compile()` that
ran in it, or before it started in the code that started it; with none, it records the last
`compile()` in the block. A `compile()` in a worker thread, as with `asyncio.to_thread`, counts only
for code that has not compiled in its own task. Model calls are the spans of type `generation`
(operations `chat`, `text_completion`, and `generate_content`), including the spans of the provider
integrations: OpenAI, Anthropic, Amazon Bedrock, Google GenAI, LiteLLM, and OpenRouter. Spans from
other OpenTelemetry tracers do not get the attributes. A model call outside `with_prompt` does not
get the attributes, even when it sends messages from `compile()`; with input capture on, the call
still records those messages as its input.

## Auto-metrics

Ended spans automatically record histograms (DELTA temporality, exported every 60s):

- `gen_ai.client.operation.duration` (unit `s`) for `chat`, `invoke_agent`, `embeddings`,
  `execute_tool`
- `gen_ai.client.token.usage` (unit `{token}`, attribute `gen_ai.token.type=input|output`) for
  `chat`, `invoke_agent`, `embeddings`
- `gen_ai.client.operation.time_to_first_chunk` (unit `s`) for `chat` when a finite, non-negative
  first-chunk timing is available.
- `gen_ai.client.operation.time_per_output_chunk` (unit `s`) for `chat` with two or more recorded
  output chunks, using the built-in OTLP exporter. Filtering applies before either streaming metric
  is emitted, including streams interrupted after receiving output.

Plain spans (`function`) record no metrics. Quiet intervals produce zero metric requests.

## Serverless

Use `export_mode="immediate"` and/or call `telemetry_dev.flush()` before the runtime freezes:

```python
telemetry_dev.init(export_mode="immediate")
...
telemetry_dev.flush()  # force-flush traces + logs + metrics
```

## Bring your own OTel (`telemetry_dev.otel`)

If you already run an OpenTelemetry SDK, attach the telemetry.dev processor to your provider
instead of calling `init()`:

```python
from telemetry_dev.otel import TelemetrySpanProcessor

your_tracer_provider.add_span_processor(TelemetrySpanProcessor())  # reads TELEMETRY_DEV_API_KEY
```

If you need to wire your own span processor stack, create the telemetry.dev exporter directly:

```python
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from telemetry_dev.otel import create_telemetry_span_exporter

span_exporter = create_telemetry_span_exporter()
your_tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
```

`TelemetrySpanProcessor` remains available from the package root for compatibility:

```python
from telemetry_dev import TelemetrySpanProcessor
```

| Option | Default | Purpose |
| --- | --- | --- |
| `api_key` | `TELEMETRY_DEV_API_KEY` | Ingest key. Absent with no `span_exporter` = inert no-op. |
| `base_url` | `TELEMETRY_DEV_BASE_URL` or `https://api.telemetry.dev` | Base URL; trailing slashes are stripped. |
| `export_mode` | `"batched"` | `"batched"` or `"immediate"` span export. |
| `max_export_batch_size` | `64` | BatchSpanProcessor export batch size. |
| `schedule_delay_millis` | `1000` | BatchSpanProcessor schedule delay. |
| `max_queue_size` | `2048` | BatchSpanProcessor queue size. |
| `export_timeout_millis` | `30000` | BatchSpanProcessor export timeout. |
| `span_filter` | `None` | Export predicate `Callable[[ReadableSpan], bool]`; failures export the span. |
| `metrics` | `True` | Auto-record GenAI duration/token histograms for exported spans when an API key is available. |
| `service_name` | `OTEL_SERVICE_NAME` or `unknown_service` | `service.name` on the metrics resource. |
| `environment` | `TELEMETRY_DEV_ENVIRONMENT` or `production` | `deployment.environment.name` on the metrics resource. |
| `on_error` | `None` | Receives export errors of the telemetry.dev exporters and internal processor errors; errors are never raised into your code. A series of failed exports is one report. After an export reaches the ingest, the next failure is a new report. |
| `span_exporter` | `None` | Advanced/test seam replacing the telemetry.dev OTLP trace exporter. |

Without an API key and without `span_exporter`, `TelemetrySpanProcessor()` is an inert no-op:
safe to attach unconditionally, with only a debug log. Auto-metrics are on by default for exported
GenAI spans, use the `service_name` / `environment` resource settings above, and are skipped
without an API key.

### Native Claude SDK spans

The Python `anthropic` SDK 1.14.0 and later emits OpenTelemetry spans natively. The Claude SDK uses the global tracer provider of the application.
To send these spans to telemetry.dev, register a global tracer provider with `TelemetrySpanProcessor`:

```sh
pip install telemetry-dev "anthropic>=1.14.0"
```

```python
from anthropic import Anthropic
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from telemetry_dev.otel import TelemetrySpanProcessor

provider = TracerProvider()
provider.add_span_processor(TelemetrySpanProcessor())
trace.set_tracer_provider(provider)

client = Anthropic()
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    messages=[{"role": "user", "content": "Say hello."}],
)

provider.shutdown()
```

If you use `telemetry_dev.init(register_global=True)`, also pass `span_filter=lambda span: True`. The default filter of that mode exports only the spans of this SDK.

By default, the spans contain metadata only. To also send prompts and responses, set `ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE=content`. The client reads this variable one time, when you make the client.

Do not use the native spans and `telemetry-dev-anthropic` together. If you use the two, each request has two spans.

You can also send the native spans without this package. Use an OTLP/HTTP protobuf exporter:

```sh
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

```python
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

exporter = OTLPSpanExporter(
    endpoint="https://api.telemetry.dev/v1/traces",
    headers={"Authorization": f"Bearer {os.environ['TELEMETRY_DEV_API_KEY']}"},
)
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
```

The Claude API can also send server-side spans to telemetry.dev through a Console destination. Set the endpoint to `https://api.telemetry.dev`. Set the `Authorization` header to `Bearer <telemetry.dev API key>`. For more information, refer to [Claude Platform](https://telemetry.dev/docs/integrations/claude-platform).

When the SDK and the Claude API send the spans of a request to the same telemetry.dev project, telemetry.dev counts the tokens of the request one time, from the API span. The SDK span stays in the trace with no tokens. telemetry.dev matches the two spans by the request ID or the response ID, so the API span does not have to be a child of the SDK span. If the Console destination uses an API key that is scoped to one environment, its API spans take over only the SDK spans of that environment.

## Limitations

- `register_global=True` cannot be undone on `shutdown()`: OpenTelemetry Python has no public
  API to unregister a global `TracerProvider`, so a later `init(register_global=True)` in the
  same process cannot reclaim the global slot. Prefer the isolated default (or the BYO
  processor) for processes that re-initialize.

## Development

Uses [uv](https://docs.astral.sh/uv/):

```sh
uv sync                 # install (writes uv.lock)
uv run pytest           # tests
uv run ruff format .    # format
uv run ruff check .     # lint
uv run pyright          # type-check
uv build                # build wheel + sdist
```

`examples/quickstart.py` is runnable against a real ingest:
`TELEMETRY_DEV_API_KEY=td_live_... uv run examples/quickstart.py`.

## License

MIT. See [LICENSE](https://github.com/telemetry-dev/telemetry/blob/main/sdks/python/LICENSE).
