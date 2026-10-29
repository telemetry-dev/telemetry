# telemetry-dev-litellm

LiteLLM integration for the telemetry.dev Python SDK. It instruments LiteLLM completion, Responses, embedding, and rerank APIs, including their async and streaming variants, and `litellm.Router` methods when available.

Docs: https://telemetry.dev/docs/integrations/litellm

## Install

```sh
pip install telemetry-dev telemetry-dev-litellm
# or
uv add telemetry-dev telemetry-dev-litellm
```

Requires Python 3.10 to 3.13 and `litellm` 1.90.2 or a later 1.x release.

## Initialize telemetry.dev

```python
import telemetry_dev
from telemetry_dev_litellm import instrument_litellm

telemetry_dev.init()  # reads TELEMETRY_DEV_API_KEY and OTLP settings
instrument_litellm()
```

Versions of `telemetry-dev` before 0.3.0 sent data to `https://ingest.telemetry.dev` by default.
That hostname reaches the same service and stays supported: if a firewall allows only it, set
`TELEMETRY_DEV_BASE_URL` to it.

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

Available drop-ins: `completion`, `acompletion`, `responses`, `aresponses`, `embedding`, `aembedding`, `rerank`, `arerank`. Each resolves the current `litellm.<function>` at call time. If global instrumentation is already active, it delegates to the globally wrapped function so the call produces exactly one span.

## Global instrumentation

```python
import litellm
from telemetry_dev_litellm import instrument_litellm, uninstrument_litellm

instrument_litellm()
response = litellm.completion(model="gpt-4o-mini", messages=[...])
uninstrument_litellm()
```

`instrument_litellm()` patches these package attributes when LiteLLM provides them:

- `litellm.completion`
- `litellm.acompletion`
- `litellm.embedding`
- `litellm.aembedding`
- `litellm.rerank`
- `litellm.arerank`
- `litellm.responses`
- `litellm.aresponses`

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
- `metadata={...}` passed to LiteLLM as `td.metadata.*` (see [Metadata](#metadata))
- errors via the core SDK's `error.type` and exception event mapping

Embedding spans use `type="embedding"` and emit `gen_ai.operation.name="embeddings"`:

- request model, provider, input, metadata
- response model, usage, cost when available
- no embedding vectors as output

Rerank spans emit `gen_ai.operation.name="rerank"`:

- request model, provider, query, documents, ranking options, and metadata
- response id, ranked results and scores, result count, provider search units, token usage, and cost when available

Responses spans emit `gen_ai.operation.name="chat"` and capture request input, instructions, sampling options, metadata, response output/status/id, usage, cost, and provider. Streaming Responses events are handled separately from chat chunks; terminal failure and error events mark the span as failed, and early close records the partial output collected so far.

Streamed chat chunks and Responses reconstruction are retained up to 48 KiB and 1,000 items. Non-streamed Responses input and output and rerank input and results are retained up to 64 KiB and 1,024 items. The configured `max_attribute_length` does not shrink these retention bounds; the core applies it to the exported attribute after the mask runs. A chat stream that is cut at its bound, closed early, or fails, and a Responses stream that is cut at its bound, ends without a terminal response, or ends with an `error` event set `telemetry.dev.capture.truncated`. With a mask configured, the incomplete reconstructed output is omitted because the mask cannot inspect the complete value.

Aggregate usage keys are `input_tokens`, `output_tokens`, `total_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, and `reasoning_output_tokens`. The core contract also includes modality fields; Responses usage emits text, image, and audio input/output token details when LiteLLM exposes them.

## Metadata

On a LiteLLM or Router call made outside the LiteLLM proxy, each key of the call's `litellm_metadata` argument, or of `metadata` when `litellm_metadata` is absent or empty, is exported as `td.metadata.<key>`, except the keys listed below. That includes keys LiteLLM's Router adds to the argument, such as `model_group`, `deployment`, `model_info` and `api_base`. Rerank spans also add `result_count` and `search_units`.

Strings, numbers, booleans and non-empty sequences of only strings, only booleans or only numbers are exported as they are; other values, including empty sequences, are JSON-encoded. Each string, including each string in a sequence, and each JSON value is truncated to `max_attribute_length`. Metadata values are not passed to the `mask` callback, and metadata is exported whatever `capture_input` and `capture_output` are set to.

These top-level keys are always dropped:

- `headers`, `requester_metadata`, `requester_ip_address`, `user_agent`, `endpoint`, `user_api_end_user_max_budget`, and every key that starts with `user_api_key`. LiteLLM's proxy stores the incoming request's headers, client IP address, user agent and URL under these names, along with the authenticated key, user, team and end user, including the user's email and JWT claims.
- `previous_models`, where the Router keeps copies of the arguments and metadata of earlier failed calls on a retry or fallback.
- `userId`, `sessionId`, `user_id` and `session_id`, which the core SDK reserves. Use `propagate_attributes(user_id=..., session_id=...)` instead.

Don't use these names for your own metadata. Other keys, including nested values, are exported as they are, so don't put credentials or personal data in metadata.

### Calls made by the LiteLLM proxy

When LiteLLM's proxy server handles a request, the call it makes through its Router carries a `proxy_server_request` argument. For every call with that argument, the integration reads metadata differently. The proxy and its guardrails and hooks write their own data into both metadata arguments, such as the request's headers, the authenticated key and user, or Presidio's `pii_tokens`. So only part of one argument is read: `litellm_metadata` for Responses calls, and `metadata` for chat, embedding and rerank calls. These are the arguments the proxy and Router write into for those calls. Two parts of it are exported:

- `requester_metadata`, the proxy's copy of the request body's `metadata` object. The proxy takes this copy before its guardrails and hooks run, but after authentication and request preparation have added data to the body's metadata, such as the `x-litellm-tags` header and the key's tags. So these keys of the copy are not exported, even when the client sent them in the body: `tags`, `spend_logs_metadata`, `agent_id`, `trace_id`, `session_id`, `used_client_oauth_token` and `litellm_session_id_omitted`, plus the keys dropped for every call listed above. If the body has no `metadata` object, nothing from `requester_metadata` is exported.
- The Router's `model_group`, `model_group_alias`, `model_group_size`, `deployment`, `deployment_model_name`, `attempted_retries` and `max_retries`, when that argument holds a string, number or boolean under them. Objects, lists and `null` under these names are not exported. The Router's `model_info` and `api_base` are not exported, because `model_info` holds any fields an admin configured for the deployment and an `api_base` URL can contain credentials.

Anything else is not exported, including keys the client sends in the body's `litellm_metadata`. Some limits remain:

- The integration has no copy of the request of its own. A guardrail or hook that rewrites `requester_metadata`, or writes one of the Router's key names into that argument, changes what is exported. The Router's key names are read whoever wrote them, so a scalar value a client or hook wrote under one of those names can appear when the Router wrote none.
- `/queue` routes skip the proxy's usual request preparation and record no `requester_metadata`. Only the Router's keys are exported, unless the body's `metadata` itself contains a `requester_metadata` object, which is then exported as the client sent it.
- The list of request keys left out of `requester_metadata` matches LiteLLM 1.90.2 through 1.105.1. A later release that adds other keys there before taking its copy would have them exported.

## Streaming

`stream=True` chat calls return a proxy around LiteLLM's `CustomStreamWrapper` that supports both sync and async iteration, context managers, `close()`, `aclose()`, and attribute delegation to the wrapped stream.

The integration never mutates request arguments. It does not inject `stream_options.include_usage`. On stream completion, early close, or mid-stream error, it rebuilds the best available response with `litellm.stream_chunk_builder(...)` and ends the span once. `time_to_first_chunk_ms` is recorded on the first chunk. Stream usage is best-effort: LiteLLM aggregates usage chunks when the provider sends them and otherwise may estimate counts from the chunks and request messages. Some LiteLLM versions instead expose zero usage for streams without provider usage, as observed with 1.104.2. When a stream produced output and its usage is zero input and zero output tokens with no other positive token count, the integration records no usage or cost instead of zero. To get the provider's own counts, pass `stream_options={"include_usage": True}` to providers that support it.

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

- Instrumented surfaces are limited to `completion`, `acompletion`, `responses`, `aresponses`, `embedding`, `aembedding`, `rerank`, `arerank`, and the corresponding methods available on `Router` via `wrap_router`. `text_completion`, image/audio/batch APIs, video APIs such as `video_generation` and `video_status`, and provider-specific APIs are out of scope for this package version.
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

## License

MIT. See [LICENSE](https://github.com/telemetry-dev/telemetry/blob/main/sdks/python-litellm/LICENSE).
