# telemetry-dev

telemetry.dev SDK for Python — OpenTelemetry-native GenAI tracing, logs, and metrics. Thin
ergonomic functions over OTel spans (`gen_ai.*` semantic conventions), exported as OTLP
protobuf to the telemetry.dev ingest.

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
| `TELEMETRY_DEV_BASE_URL` | `https://ingest.telemetry.dev` | Ingest base URL (trailing slashes stripped). |
| `TELEMETRY_DEV_ENVIRONMENT` | `production` | Deployment environment label. |
| `OTEL_SERVICE_NAME` | `unknown_service` | Service name on every trace. |
| `OTEL_TRACES_SAMPLER` | `parentbased_always_on` | OTel trace sampling policy, which includes session roots. |
| `OTEL_TRACES_SAMPLER_ARG` | `1.0` | Probability for `traceidratio` / `parentbased_traceidratio`. |

Explicit `init()` arguments take precedence over environment variables.

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
| `SpanHandle.update(**fields)` / `.end(**fields, end_time=)` / `.traceparent()` | Update attributes, end (accepts the full update field set), or read the W3C traceparent. |
| `update_current_span(**fields)` | Apply the update field set to the currently active span (no-op without one). |
| `propagate_attributes(*, user_id=, session_id=, metadata=)` | Context manager stamping `user.id` / `gen_ai.conversation.id` / `td.metadata.*` on every span and log record started inside (threads/asyncio included via contextvars). |
| `log(message, *, level="info", event_name=None, attributes=None)` | Emit an OTLP log record to `/v1/logs`, correlated with the current trace. Levels: `debug`/`info`/`warn`/`error` (`"warning"` is accepted as an alias of `warn`). |
| `get_traceparent() -> str \| None` | W3C traceparent of the current context. |
| `flush(timeout_s=10.0)` / `shutdown(timeout_s=10.0)` | Force-flush / tear down traces + logs + metrics. Shutdown also runs atexit unless `disable_atexit=True`. |
| `TelemetrySpanProcessor` / `telemetry_dev.otel.create_telemetry_span_exporter` | Bring-your-own-OTel helpers (below). |
| `MaskContext`, `Usage`, `SpanHandle`, `Client`, `NOT_GIVEN` | Supporting types. |

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
`output_type`, `finish_reason`, `usage` (dict with exactly `input_tokens`, `output_tokens`,
`total_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`,
`reasoning_output_tokens`), `cost_usd`, `temperature`, `top_p`, `top_k`, `max_tokens`,
`stop_sequences`, `seed`, `frequency_penalty`, `presence_penalty`, `time_to_first_chunk_ms`,
`tool_name`, `tool_call_id`, `tool_description`, `agent_name`, `agent_id`, `metadata`
(→ `td.metadata.*`, this span only), `attributes` (raw escape hatch, merged last), `error`.
`start_span` additionally accepts `parent` (traceparent string, OTel `Context`, or
`SpanContext`), `start_time`, and per-call `capture_input` / `capture_output` overrides;
`end()` additionally accepts `end_time`.

### init() options

| Option | Default | Purpose |
| --- | --- | --- |
| `api_key`, `base_url`, `environment`, `service_name` | env vars | Connection + resource settings. |
| `enabled` | `True` | `False` = hard kill switch (tests). |
| `register_global` | `False` | Also register the tracer provider globally. Applies a default export filter (only `telemetry_dev`-scoped spans are exported); pass `span_filter=lambda s: True` to export everything. |
| `export_mode` | `"batched"` | `"immediate"` exports synchronously per span/log (serverless). |
| `log_level` | `"warn"` | SDK diagnostics level: `debug`/`info`/`warn`/`error`/`silent`. |
| `capture_input`, `capture_output` | `True` | Global content-capture defaults. |
| `mask` | `None` | `Callable[[Any, MaskContext], Any]` redaction hook, runs before JSON serialization on input/output/log messages (`MaskContext.key` is the attribute being written). Not applied to correlation identifiers. |
| `max_attribute_length` | `65536` | Per-content-attribute cap; truncated values get an ASCII `...[truncated]` marker appended. |
| `span_filter` | `None` | Export predicate `Callable[[ReadableSpan], bool]`. |
| `sampler` | OTel environment configuration | OTel `Sampler` that overrides environment settings. Session roots use its root policy. Real parents keep their sampling decisions. |
| `on_error` | `None` | Receives every internal SDK error; the SDK never raises. |
| `disable_atexit` | `False` | Skip the automatic atexit shutdown. |
| `timeout` | `10.0` | OTLP HTTP timeout in seconds. |
| `span_exporter`, `log_exporter`, `metric_reader` | `None` | Test seams / offline mode; any of them enables the client without an API key. |

Session roots share trace IDs from the API key and session ID. The hash matches the TypeScript SDK.
The SDK uses the configured OTel sampling policy, not the synthetic parent's flags.
For example, `init(sampler=ParentBased(TraceIdRatioBased(0.1)))` samples sessions with OTel's ratio sampler.
These classes come from `opentelemetry.sdk.trace.sampling`.
The built-in Python and TypeScript ratio samplers can select different sessions because the OTel algorithms differ.

## Auto-metrics

Ended spans automatically record two histograms (DELTA temporality, exported every 60s):

- `gen_ai.client.operation.duration` (unit `s`) for `chat`, `invoke_agent`, `embeddings`,
  `execute_tool`
- `gen_ai.client.token.usage` (unit `{token}`, attribute `gen_ai.token.type=input|output`) for
  `chat`, `invoke_agent`, `embeddings`

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
| `base_url` | `TELEMETRY_DEV_BASE_URL` or `https://ingest.telemetry.dev` | Ingest base URL; trailing slashes are stripped. |
| `export_mode` | `"batched"` | `"batched"` or `"immediate"` span export. |
| `max_export_batch_size` | `64` | BatchSpanProcessor export batch size. |
| `schedule_delay_millis` | `1000` | BatchSpanProcessor schedule delay. |
| `max_queue_size` | `2048` | BatchSpanProcessor queue size. |
| `export_timeout_millis` | `30000` | BatchSpanProcessor export timeout. |
| `span_filter` | `None` | Export predicate `Callable[[ReadableSpan], bool]`; failures export the span. |
| `metrics` | `True` | Auto-record GenAI duration/token histograms for exported spans when an API key is available. |
| `service_name` | `OTEL_SERVICE_NAME` or `unknown_service` | `service.name` on the metrics resource. |
| `environment` | `TELEMETRY_DEV_ENVIRONMENT` or `production` | `deployment.environment.name` on the metrics resource. |
| `on_error` | `None` | Receives internal processor errors; errors are never raised into your code. |
| `span_exporter` | `None` | Advanced/test seam replacing the telemetry.dev OTLP trace exporter. |

Without an API key and without `span_exporter`, `TelemetrySpanProcessor()` is an inert no-op:
safe to attach unconditionally, with only a debug log. Auto-metrics are on by default for exported
GenAI spans, use the `service_name` / `environment` resource settings above, and are skipped
without an API key.

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
