# telemetry-dev-litellm

LiteLLM integration for the telemetry.dev Python SDK. It instruments `litellm.completion`, `litellm.acompletion`, `litellm.embedding`, `litellm.aembedding`, streaming `CustomStreamWrapper` responses, and `litellm.Router` calls by emitting spans through the public `telemetry_dev.start_span` API.

## Install

```sh
pip install telemetry-dev telemetry-dev-litellm
# or
uv add telemetry-dev telemetry-dev-litellm
```

Requires Python `>=3.10,<3.14`.

## Initialize telemetry.dev

```python
import telemetry_dev
from telemetry_dev_litellm import instrument_litellm

telemetry_dev.init()  # reads TELEMETRY_DEV_API_KEY and OTLP settings
instrument_litellm()
```

The core SDK fails open: without an API key, LiteLLM calls still run and telemetry is a no-op.

## Per-call drop-ins

Use the exported functions as drop-in replacements when you do not want global monkey-patching:

```python
from telemetry_dev_litellm import completion

response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hi"}],
)
```

Available drop-ins: `completion`, `acompletion`, `embedding`, `aembedding`. Each resolves the current `litellm.<function>` at call time. If global instrumentation is already active, it delegates to the globally wrapped function so the call produces exactly one span.

## Global instrumentation

```python
import litellm
from telemetry_dev_litellm import instrument_litellm, uninstrument_litellm

instrument_litellm()
response = litellm.completion(model="gpt-4o-mini", messages=[...])
uninstrument_litellm()
```

`instrument_litellm()` patches exactly four package attributes:

- `litellm.completion`
- `litellm.acompletion`
- `litellm.embedding`
- `litellm.aembedding`

It is thread-safe and idempotent. `uninstrument_litellm()` restores the original objects and is also idempotent.

Import-order limitation: references captured before instrumentation, such as `from litellm import completion`, keep pointing at the original function and are not traced. Instrument at process startup or use the `telemetry_dev_litellm` drop-ins.

## Router wrapping

Global instrumentation traces Router deployment attempts because LiteLLM Router dispatches through the patched package attributes at call time:

```python
import litellm
from telemetry_dev_litellm import instrument_litellm

instrument_litellm()
router = litellm.Router(model_list=[...])
router.completion(model="my-group", messages=[...])  # one span per deployment attempt
```

Use `wrap_router(router)` for an enclosing Router-level span:

```python
import litellm
from telemetry_dev_litellm import instrument_litellm, wrap_router

instrument_litellm()
router = wrap_router(litellm.Router(model_list=[...]))
router.completion(model="my-group", messages=[...])
```

With both `wrap_router()` and `instrument_litellm()` active, the Router span is the parent and each deployment attempt is a child span. Router-injected LiteLLM metadata such as `model_group` and `deployment` is forwarded as `td.metadata.*` on inner attempt spans when LiteLLM provides it.

Internal LiteLLM `num_retries` retries happen inside a single wrapped call, so they are represented as one span whose duration covers all attempts. Router retries and fallbacks are separate deployment calls and can produce separate child spans under global instrumentation.

## Captured fields

Chat spans use `type="generation"` and emit `gen_ai.operation.name="chat"`:

- request model, provider, messages, sampling parameters (including `top_k`), stop sequences, seed, frequency and presence penalties
- `response_format` as `gen_ai.output.type` (`json` for `json_object`/`json_schema`/pydantic models)
- response model, response id, output messages, finish reasons, usage, cost when available
- `metadata={...}` passed to LiteLLM as `td.metadata.*`
- errors via the core SDK's `error.type` and exception event mapping

Embedding spans use `type="embedding"` and emit `gen_ai.operation.name="embeddings"`:

- request model, provider, input, metadata
- response model, usage, cost when available
- no embedding vectors as output

Usage keys follow the core SDK contract: `input_tokens`, `output_tokens`, `total_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, and `reasoning_output_tokens`.

## Streaming

`stream=True` chat calls return a proxy around LiteLLM's `CustomStreamWrapper` that supports both sync and async iteration, context managers, `close()`, `aclose()`, and attribute delegation to the wrapped stream.

The integration never mutates request arguments. It does not inject `stream_options.include_usage`. On stream completion, early close, or mid-stream error, it rebuilds the best available response with `litellm.stream_chunk_builder(...)` and ends the span once. `time_to_first_chunk_ms` is recorded on the first chunk. Stream usage is best-effort: LiteLLM aggregates usage chunks when present, otherwise it estimates usage from chunks and request messages.

## Provider attribution

LiteLLM provider names are mapped to OpenTelemetry well-known `gen_ai.provider.name` values where one exists:

| LiteLLM provider | Emitted provider |
| --- | --- |
| `openai` | `openai` |
| `azure`, `azure_text` | `azure.ai.openai` |
| `azure_ai` | `azure.ai.inference` |
| `anthropic`, `anthropic_text` | `anthropic` |
| `bedrock` | `aws.bedrock` |
| `vertex_ai`, `vertex_ai_beta` | `gcp.vertex_ai` |
| `gemini` | `gcp.gemini` |
| `mistral` | `mistral_ai` |
| `groq` | `groq` |
| `deepseek` | `deepseek` |
| `xai` | `x_ai` |
| `cohere`, `cohere_chat` | `cohere` |
| `perplexity` | `perplexity` |
| `watsonx`, `watsonx_text` | `ibm.watsonx.ai` |

Unlisted providers pass through unchanged, for example `ollama`, `openrouter`, `together_ai`, and `fireworks_ai`.

## Cost

LiteLLM's synchronous response metadata is used when available: `_hidden_params.response_cost` maps to `gen_ai.usage.cost`. If that field is absent, the integration tries `litellm.completion_cost(completion_response=...)` and silently omits cost if LiteLLM cannot price the model.

## Limitations

- Instrumented surfaces are limited to `completion`, `acompletion`, `embedding`, `aembedding`, and the same four methods on `Router` via `wrap_router`. `text_completion`, `litellm.responses`, image/audio/rerank/batch APIs, and provider-specific APIs are out of scope for this package version.
- References imported from LiteLLM before `instrument_litellm()` are not patched.
- Internal `num_retries` retries are not separate spans; Router deployment attempts are.
- OTel's built-in-tools scenario is not portable across LiteLLM providers and is not covered by the examples.
- Multimodal output is not covered; LiteLLM chat returns text/tool calls, while image and audio generation use APIs outside this integration.

## Development

```sh
uv sync
uv run pytest
uv run ruff format .
uv run ruff check .
uv run pyright
```

Repository-level commands run this package together with the core Python SDK:

```sh
pnpm run py:sync
pnpm run py:check
pnpm run py:test
```
