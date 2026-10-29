# SDK conformance: the cross-language contract

`@telemetry-dev/sdk` (TypeScript, `packages/sdk`) and `telemetry-dev` (Python, `sdks/python`)
must emit byte-compatible OTLP for the same logical input. This document defines that contract.
The ingestion assertions and database-backed e2e harness live in the separate private product
repository, not this SDK workspace. Paths below refer to that product repository:

- `apps/ingest/tests/sdk_otlp_e2e_test.ts` — TypeScript SDK, capture-fetch → replay through
  `handleOtlpRequest`.
- `apps/ingest/tests/python_sdk_otlp_e2e_test.ts` — Python SDK, real HTTP against a `Bun.serve`
  ingest on an ephemeral port, scenario driven by `sdks/python/tests/e2e_scenarios.py`
  (skipped with a warning when `uv` is missing; CI must not skip it).

## Attribute contract

| SDK field (TS / Py)                                                                                                                                                                                                                                                                                                                                                                                                   | Wire attribute                                                                                                                       | Notes                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `type: "span"`                                                                                                                                                                                                                                                                                                                                                                                                        | `gen_ai.operation.name = "function"`                                                                                                 | Always explicit: a propagated `gen_ai.conversation.id` would otherwise trip the ingest's `hasGenAi` → `"llm"` heuristic |
| `type: "generation"`                                                                                                                                                                                                                                                                                                                                                                                                  | `gen_ai.operation.name = "chat"`                                                                                                     |                                                                                                                         |
| `type: "tool"`                                                                                                                                                                                                                                                                                                                                                                                                        | `gen_ai.operation.name = "execute_tool"`                                                                                             |                                                                                                                         |
| `type: "agent"`                                                                                                                                                                                                                                                                                                                                                                                                       | `gen_ai.operation.name = "invoke_agent"`                                                                                             |                                                                                                                         |
| `type: "embedding"`                                                                                                                                                                                                                                                                                                                                                                                                   | `gen_ai.operation.name = "embeddings"`                                                                                               |                                                                                                                         |
| `input` / `output`                                                                                                                                                                                                                                                                                                                                                                                                    | `gen_ai.input.messages` / `gen_ai.output.messages` (`gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` for tools)              | mask → JSON stringify → truncate; gated by capture flags                                                                |
| `attributes["telemetry.dev.capture.truncated"]`                                                                                                                                                                                                                                                                                                                                                                       | `telemetry.dev.capture.truncated = true`                                                                                             | captured telemetry is incomplete                                                                                        |
| `model`, `provider`                                                                                                                                                                                                                                                                                                                                                                                                   | `gen_ai.request.model`, `gen_ai.provider.name`                                                                                       |                                                                                                                         |
| `systemInstructions` / `system_instructions`                                                                                                                                                                                                                                                                                                                                                                          | `gen_ai.system_instructions`                                                                                                         | gated by captureInput                                                                                                   |
| `responseModel`, `responseId`, `outputType`                                                                                                                                                                                                                                                                                                                                                                           | `gen_ai.response.model`, `gen_ai.response.id`, `gen_ai.output.type`                                                                  |                                                                                                                         |
| `usage.{inputTokens,outputTokens,totalTokens,cacheReadInputTokens,cacheCreationInputTokens,reasoningOutputTokens}`                                                                                                                                                                                                                                                                                                    | `gen_ai.usage.{input_tokens,output_tokens,total_tokens,cache_read.input_tokens,cache_creation.input_tokens,reasoning.output_tokens}` | aggregate input tokens include cache reads and cache writes                                                             |
| `usage.{textInputTokens,textOutputTokens,textCacheReadInputTokens,imageInputTokens,imageOutputTokens,imageCacheReadInputTokens,audioInputTokens,audioOutputTokens,audioCacheReadInputTokens}` / `usage.{text_input_tokens,text_output_tokens,text_cache_read_input_tokens,image_input_tokens,image_output_tokens,image_cache_read_input_tokens,audio_input_tokens,audio_output_tokens,audio_cache_read_input_tokens}` | `gen_ai.usage.{text,image,audio}.{input_tokens,output_tokens,cache_read.input_tokens}`                                               | flat TypeScript / Python modality fields; SDK usage objects drop unknown keys                                           |
| `costUsd` / `cost_usd`                                                                                                                                                                                                                                                                                                                                                                                                | `gen_ai.usage.cost` (number)                                                                                                         | optional override; server pricing (`apps/ingest/src/pricing.ts`) is the default                                         |
| `finishReason`                                                                                                                                                                                                                                                                                                                                                                                                        | `gen_ai.response.finish_reasons = [value]`                                                                                           | single-element array                                                                                                    |
| sampling params                                                                                                                                                                                                                                                                                                                                                                                                       | `gen_ai.request.{temperature,top_p,top_k,max_tokens,stop_sequences,seed,frequency_penalty,presence_penalty}`                         | flat fields                                                                                                             |
| `timeToFirstChunkMs` / `time_to_first_chunk_ms`                                                                                                                                                                                                                                                                                                                                                                       | `gen_ai.response.time_to_first_chunk`                                                                                                | **seconds** on the wire (key has no `ms` suffix; ingest multiplies by 1000)                                             |
| `toolName`, `toolCallId`, `toolDescription`                                                                                                                                                                                                                                                                                                                                                                           | `gen_ai.tool.{name,call.id,description}`                                                                                             |                                                                                                                         |
| `agentName`, `agentId`                                                                                                                                                                                                                                                                                                                                                                                                | `gen_ai.agent.{name,id}`                                                                                                             |                                                                                                                         |
| `metadata.<k>` (per span)                                                                                                                                                                                                                                                                                                                                                                                             | `td.metadata.<k>`                                                                                                                    | reserved keys `userId`/`sessionId`/`user_id`/`session_id` dropped with a debug warning; never passed through the mask   |
| `propagateAttributes userId`                                                                                                                                                                                                                                                                                                                                                                                          | `user.id` on EVERY span/log in scope                                                                                                 | stamped by the span processor's onStart                                                                                 |
| `propagateAttributes sessionId`                                                                                                                                                                                                                                                                                                                                                                                       | `gen_ai.conversation.id` on EVERY span/log in scope                                                                                  | never `session.id`                                                                                                      |
| `propagateAttributes sessionId` on a span that starts with no active parent                                                                                                                                                                                                                                                                                                                                           | remote parent: trace id = SHA-256(apiKey ‖ 0x00 ‖ sessionId)[0:16], parent span id = digest[16:24]                                   | one trace per session per API key; the parent span is never emitted                                                     |
| `init({sessionMode: "process"})` / `init(session_mode="process")`                                                                                                                                                                                                                                                                                                                                                     | fresh UUID as `gen_ai.conversation.id` on spans/logs without an explicit or propagated session                                       | opt-in per enabled initialization; explicit/propagated sessions win; valid parent trace context is never replaced       |
| error                                                                                                                                                                                                                                                                                                                                                                                                                 | span status ERROR + `error.type` + `exception` event (`exception.type/message/stacktrace`, `log.severity_number: 17`)                |                                                                                                                         |
| `log()`                                                                                                                                                                                                                                                                                                                                                                                                               | OTLP log record to `/v1/logs` (always; never span events)                                                                            | severity debug=5/info=9/warn=13/error=17; trace-correlated via active context                                           |
| `withPrompt(prompt, fn)` / `with with_prompt(prompt):` → `prompt.name`                                                                                                                                                                                                                                                                                                                                                | `td.prompt.name`                                                                                                                     | on model-call spans that start in scope: operation `chat`, `text_completion`, or `generate_content`                     |
| outside a scope: a message object that `compile()` of a chat prompt returned, in the request messages of a provider integration                                                                                                                                                                                                                                                                                       | `td.prompt.name`, `td.prompt.version`, `td.prompt.label`                                                                             | on the `chat` spans of the provider integrations listed in [Prompts](#prompts); the scope wins                          |
| `prompt.version`                                                                                                                                                                                                                                                                                                                                                                                                      | `td.prompt.version` (int)                                                                                                            | omitted for a fallback prompt (`version` null / None)                                                                   |
| `prompt.label`                                                                                                                                                                                                                                                                                                                                                                                                        | `td.prompt.label`                                                                                                                    | only for a fetch by label; a fetch without label or version is the `production` label                                   |
| variables of the last `prompt.compile()` before `withPrompt` / `with_prompt` starts, or in its scope; outside a scope, of the newest `compile()` that made a message of the request                                                                                                                                                                                                                                   | `td.prompt.variables` (JSON object)                                                                                                  | mask → JSON stringify → truncate; gated by the effective captureInput of the span                                       |

`SpanHandle.serializeContent(value, key)` / `SpanHandle.serialize_content(value, key)` returns
what `input` / `output` would emit for `value` under `key`: mask → JSON stringify → truncate. It
returns `undefined` / `None` when the mask throws, the value cannot be serialized, or there is no
client, and never checks the capture flags. Integrations use it for content they write to other
attributes. A TypeScript mask that returns `undefined` omits the value; Python has no equivalent,
and a mask returning `None` serializes as `null`, like a TypeScript mask returning `null`.

Captured JSON uses `JSON.stringify` number spelling and property order: no separator whitespace,
non-finite numbers become `null`, negative zero becomes `0`, and array-index keys precede other
keys in numeric order. Other string keys retain insertion order. Python integers outside the safe
integer range are converted to binary64 numbers, as JavaScript does when reading JSON. JSON
strings escape lone surrogates and preserve valid non-ASCII characters. Provider JSON attributes
and Anthropic stream-size accounting use the same encoder. Native numeric OTLP attributes are
not JSON strings and are unchanged.

Raw string inputs still pass through without JSON quotes, and Python-only objects retain the
`repr` fallback (Bedrock unknown content blocks use `str`). Truncation counts UTF-16 units;
Python retains its existing exception at a cut inside a surrogate pair, dropping the partial
character rather than emitting the lone high surrogate retained by JavaScript's slice.

The mask never sees per-span or propagated metadata or raw span attributes. TypeScript `log()`
passes log attributes other than numbers and booleans through the mask; Python `log()` does not
mask log attributes.

Anthropic integrations record advisor tokens separately from executor `gen_ai.usage.*` under
`anthropic.advisor.usage.{input_tokens,output_tokens,cache_read_input_tokens,cache_creation_input_tokens}`.
Advisor input tokens exclude cache reads and writes, matching Anthropic's native counters.
`anthropic.advisor.model` identifies the model for pricing. Missing or conflicting model identities
omit that attribute, retain the counters, and set `telemetry.dev.capture.truncated = true`.
Counters sum advisor entries in the last 1,000 iterations; longer lists also set the truncation flag.
Streaming replaces cumulative usage snapshots rather than adding them. Absent counters stay absent;
zero counters remain zero. Advisor tokens do not change executor usage or its model.

Anthropic integrations record each request tool's type and name in `gen_ai.tool.definitions` as a
JSON array of `{type, name}`, with `type` `function` for custom tools and the tool's own type for
server tools. They record it regardless of capture flags and `mask`; descriptions and input
schemas stay in the captured input.

Shared constants: env vars `TELEMETRY_DEV_API_KEY` / `TELEMETRY_DEV_BASE_URL` (default
`https://api.telemetry.dev`, for telemetry export and `getPrompt` / `get_prompt`; earlier releases
used `https://ingest.telemetry.dev`, another hostname for the same service) /
`TELEMETRY_DEV_ENVIRONMENT` (default `production`) /
`OTEL_SERVICE_NAME` (default `unknown_service`); truncation cap 65536 chars **including** the
`...[truncated]` marker (custom caps shorter than the marker retain its prefix); batch defaults
64/1000ms/2048/30000ms; instrumentation scopes
`@telemetry-dev/sdk` and `telemetry_dev` (both classify as framework `otel`).

The SDK allowlists above drop unknown keys in SDK usage objects. Ingest reports
`unrecognized_usage_attribute` for unknown numeric `gen_ai.usage.*`, `ai.usage.*`, and
`llm.token_count.*` token attributes and for unknown cost aliases such as `td.cost`, received through
OTLP or another raw-attribute path, when the matching stored token or cost field stays empty.

Session mode defaults to `explicit`, preserving independent root traces. `process` mode creates a
new opaque UUID for each enabled SDK initialization and reuses it only for that client's lifetime.
Shutdown followed by init, or re-init replacement, creates a new UUID. It is not a machine-,
deployment-, or cross-process session. Invalid modes fail open through `onError` / `on_error`;
disabled and missing-key initialization remains a no-op and does not create a session.

Auto-metrics: `gen_ai.client.operation.duration` (unit `s`) for operations
{chat, invoke_agent, embeddings, execute_tool}; `gen_ai.client.token.usage` (unit `{token}`,
`gen_ai.token.type` input|output) for {chat, invoke_agent, embeddings}; nothing for `function`.
`gen_ai.client.operation.time_to_first_chunk` (unit `s`) records one sample for a `chat` span
with a finite, non-negative `gen_ai.response.time_to_first_chunk` value. Missing timings do not
produce a sample. `gen_ai.client.operation.time_per_output_chunk` (unit `s`) records the interval
between consecutive nonempty output chunks for `chat` spans. Usage-only and control events do not
count. Zero or one output chunk produces no interval. These are client-observed timings, not
server token timings; pull-based streams include time the consumer spends between reads.
Both metrics are recorded only after the span passes filtering at span end, including spans that
fail after receiving output. Chunk intervals use fixed-size histogram state, not retained samples;
completed spans merge into at most 2,000 attribute sets per export, including an
`otel.metric.overflow=true` series. No synthetic exemplars are generated for chunk intervals.
In Python, this exact aggregate merge is available only with the built-in OTLP metric exporter.
Passing a custom `metric_reader` preserves duration, token usage, and time-to-first-chunk metrics,
but disables time-per-output-chunk because OpenTelemetry Python 1.x has no public aggregate-input
or `MetricProducer` API. The first accepted span with intervals reports the limitation once through
`on_error` and the SDK logger; no undrainable aggregate is retained.
The metrics use DELTA temporality and the bucket boundaries in `packages/otel/src/metrics.ts`.

## Prompts

`getPrompt(name, options)` / `get_prompt(name, ...)` send `GET {baseUrl}/v1/prompts?name=<name>`
with `label=<label>` (default `production`) or `version=<n>`, the `Authorization: Bearer <key>`
header, and the SDK header. `{baseUrl}` is the `baseUrl` / `base_url` option, then the base URL of
an enabled `init()`, then `TELEMETRY_DEV_BASE_URL`, then `https://api.telemetry.dev`, so prompts
come from the host that telemetry goes to. For the prompt request, an empty value counts as unset.
The two SDKs keep one copy per base URL, API key fingerprint, name, and label or version.
Concurrent calls share one fetch.

- Cache: when the copy is older than `cacheTtlSeconds` / `cache_ttl_seconds` (default 60), a call
  returns the copy at once and starts one fetch in the background. Only a call with no copy waits
  for a fetch.
- Timeout: a call waits for a fetch for 10 seconds at most, and then gets `timeout`. The
  TypeScript SDK aborts the fetch at that time. The Python SDK fetches in a daemon thread and shuts
  its connection down at that time, because the urllib timeout applies to each socket operation
  and not to the full response. Only an address lookup can continue after the call stops waiting,
  until the system resolver gives up. Both SDKs measure the TTL and Retry-After in elapsed time, so
  a change of the system clock does not move them.
- Failure: when a fetch fails, the SDKs keep the last copy for another TTL and send the error to
  `onError` / `on_error`. With no copy, the call gets the fallback. With no fallback, the call gets
  a `PromptError`. With no copy, the calls of the next 10 seconds (or the TTL, if it is shorter)
  get the same result without a fetch.
- Retry-After: when a 429 or 503 response has a `Retry-After` header, in seconds or as an HTTP
  date (IMF-fixdate, `Sun, 06 Nov 1994 08:49:37 GMT`), no fetch of that prompt (the same base URL,
  key, name, and label or version) starts before that time, even when the copy or the error is
  older than its TTL. The SDKs cap the wait at one hour and ignore the header on other statuses
  and when it does not parse.
- Error codes: the `PromptError` code is the API error code, or `missing_api_key`, `timeout`,
  `network_error`, `invalid_response`, or `type_mismatch`. A response with a status that is not
  2xx and no API error code gives `http_<status>`.
- Redirects: the SDKs do not send the request to the location of a redirect, thus the key does not
  go to a different host. A 3xx response gives `http_<status>`, for example `http_302`. The
  TypeScript SDK uses `redirect: "manual"`, because Cloudflare Workers reject `redirect: "error"`.

A label with a rollout gives the rollout version when the bucket is less than the rollout
percent. The bucket is FNV-1a 32-bit (offset 2166136261, prime 16777619) over the UTF-8 bytes of
`"<name>:<label>:<session id>"`, mod 100. The session id is the propagated
`gen_ai.conversation.id`, then the process session id; without one, the bucket is random per call.
Test vector: `"support/reply:production:session-123"` → 777154315 → bucket 15.

`compile()` replaces each `{{name}}` with the value of the variable: a string as it is, a missing
value or null as empty text, and other values as `JSON.stringify` JSON. A chat message
`{role: "placeholder", name}` becomes the messages in its variable: a list, or in Python also a
tuple. A missing, null, or None variable gives no messages; any other value is a `TypeError`.

Concurrent tasks inside one `withPrompt` / `with_prompt` scope: in Python, a model-call span takes
the variables of the last `compile()` that ran in its own asyncio task, or before the task started
in the code that started it (a `ContextVar`), and the last `compile()` of the scope when there is
none. A `compile()` in a worker thread or a copied context, as with `asyncio.to_thread`, counts
only for code that has not compiled in its own task. JavaScript
cannot tell a concurrent task from an awaited helper, so in TypeScript all spans of the scope take
its last `compile()`; concurrent tasks that compile their own variables call `withPrompt` in each
task.

Outside a scope, the SDKs find the prompt from the message objects of the request, not from
their text:

- `compile()` of a chat prompt tags each message object that it makes with the prompt, a copy of
  the variables of that call and of each list in them (not of the objects in a list), and a serial
  number that grows with each chat compile in the process. With input capture off in the client
  at that time, the tag holds no variables. It does not tag the messages of a placeholder
  variable: the app passes those, and they keep the tag of their own compile. TypeScript: a
  non-enumerable symbol property; the keys and the JSON of a message do not change. Python: the
  messages are instances of a private `dict` subclass that holds the tag in a slot.
  `isinstance(message, dict)` is true and the keys and the JSON do not change, but a check of the
  exact type (`type(message) is dict`), `yaml.safe_dump`, and `marshal` reject it. Its
  constructor is the one of `dict`.
- A copy has no tag: a spread, `structuredClone`, or a JSON round trip (TypeScript); `dict()`,
  `.copy()`, `copy.copy`, `copy.deepcopy`, pickle, a JSON round trip, `type(message)(message)`, or
  `dataclasses.asdict` (Python). A new message with the same text has no tag either, thus a
  message from another call, task, or prompt version is never recorded.
- `promptAttributesFor(messages)` / `prompt_attributes_for(messages)` take a list of messages or
  one message (Python: a `list` or `tuple`, not a subclass, so the read runs no code of the app).
  They look at the first 1000 and the last 1000 items and give the name, version, label, and
  variables of the tag with the highest serial number: the newest compile. Thus a message that a
  placeholder brings from an older compile does not win over the compile that put it in the list.
  They do not look inside a message.
- In a `withPrompt` / `with_prompt` scope they give nothing. A model-call span whose parent context
  is in a scope gets the attributes of the scope and drops each `td.prompt.*` attribute that the
  app passes, at start and in each update (`update`, `end`, `updateActiveSpan` /
  `update_current_span`), also when the app calls these functions outside the scope and passes an
  explicit parent. They never throw: an error goes to `onError` / `on_error`.
- They give `td.prompt.variables` only when input capture is on in the client, at the compile and
  at the call. Then they serialize the variables, which calls the `mask` of the client. A span
  whose effective captureInput is off drops `td.prompt.variables` from its attributes, also when
  the app passes the result of these functions to its own span.
- A tag keeps the variables of its compile alive while a message of the compile lives. When a
  chat compile gets a tagged message as a variable, or as an item of a list (Python: a list or
  tuple) in a variable, and the tag is at least 100 serial numbers older, the older tag drops its
  variables. Thus a chat history that keeps a compiled message of each turn keeps the variables of
  about the last 100 compiles, not of each turn, and a request with only such messages records no
  `td.prompt.variables`. The SDKs do not look deeper in a variable. They keep no other reference to
  the variables, except those of the last compile of each prompt object and of each running scope,
  for `withPrompt` / `with_prompt`.
- `compile()` of a text prompt returns a string, which cannot carry a tag. Outside a scope, the
  SDKs do not record a text prompt: use `withPrompt` / `with_prompt`.
- The provider integrations call them with the request messages before the span starts, also
  when input capture is off: OpenAI chat completions (`messages`) and responses (`input`),
  Anthropic messages (`messages`; the `system` string has no tag), Amazon Bedrock Converse
  (`messages`), Google GenAI `generateContent` (`contents`), LiteLLM completion (`messages`) and
  responses (`input`) (Python), and OpenRouter chat (`messages`) and responses (`input`). Bedrock
  Converse and Google GenAI take another message shape: a compiled message that the app changes
  in place keeps its tag, and a new message does not. Bedrock InvokeModel takes a serialized body
  and is not recorded outside a scope. The other spans of the integrations (embeddings, images,
  audio, video, and OpenAI Realtime) do not look at the request.
- Python Anthropic `messages.stream()` starts its span at `__enter__`. It decides there, with the
  messages that `.stream()` received, thus a scope around `__enter__` wins.
- A provider integration reads the function from the core module when it runs, thus it also
  operates with an older core that does not have it. With such a core, the prompt is recorded
  only in a scope.

## The shared scenario

Both e2e runners emit exactly this (under
`propagate(userId="user_e2e", sessionId=<random>, metadata={plan:"pro"})`):

The SDK is initialized with a mask hook that replaces any string `gen_ai.input.messages` value
containing `SECRET` with `{masked: true}`.

1. agent span `support-agent` (`agentName: "support"`) containing:
   - generation `chat-completion` (gpt-4o/openai, input "What is the weather?", output
     "It is sunny.", usage 17/7 where input is 11 + 4 cache-read + 2 cache-write, with text 6/4/2,
     image 3/2/1, audio
     2/1/1 input/output/cache-read modality detail, finishReason stop) with `log("inside generation",
eventName: "e2e.inside")` inside it
   - tool `web-search` (toolCallId `call_1`, input `{q:"weather"}`, output `{hits:1}`)
   - observed function `format-output` (input `{text:"It is sunny."}`, output
     `{formatted:true}`)
   - plain span `broken-step` ended with an error whose message contains "boom"
     (TypeError in TS, ValueError in Python)
   - embedding span `embed-query` (model `text-embedding-4`, usage input 3)
   - plain span `private-step` (input "should-not-appear", captureInput false, error
     "PRIVATE_EXCEPTION" whose message and stack are omitted)
   - plain span `big-payload` (input = 10000 × "x" + 30000 × "🤖")
   - generation `priced-call` (custom-model-x/custom, usage 5/2, costUsd 0.5)
   - generation `openrouter-call` (provider openrouter, no usage)
   - generation `masked-step` (input "SECRET stuff", error "SECRET exception" whose message
     and stack are masked to "[redacted exception]")
   - plain span `json-compat` (nested non-finite values, numeric key ordering, binary64 numbers,
     Unicode and lone surrogates; serialized bytes also stored as a custom attribute)
   - generation `advisor-call` (claude-sonnet-4-5/anthropic, executor usage 8/3 with 2 cache-read
     and 1 cache-write; separate claude-opus-4-1 advisor counters 5/7 with 11 cache-read and
     13 cache-write, emitted through the core SDK)
   - generation `tool-offer` (claude-sonnet-4-5/anthropic, captureInput false, input with the
     tool description "PRIVATE_TOOL_DESCRIPTION", raw `gen_ai.tool.definitions` attribute
     naming `issue_refund` (function) and `web_search` (web_search_20250305))
   - generation `streamed-chat` (gpt-4o/openai, input "Stream please", bounded partial output
     "chunk...", `telemetry.dev.capture.truncated = true`, four output chunk calls at
     1000/1010/1050/1210 ms, timeToFirstChunkMs 250)
   - prompt `support/reply` (version 12, label production) fetched from a local stub server,
     compiled with `customer: "Ada"`, then in `withPrompt` / `with_prompt`: generation
     `prompted-call` (gpt-4o/openai, input the compiled messages) and generation
     `prompted-private` (the same with captureInput false)
2. plain span `background-job` started after the agent ended, with
   `parent = <agent traceparent captured via getTraceparent()>`
3. agent span `follow-up-turn` (`agentName: "support"`) started after the agent ended with no
   active parent: a second root call of the same session
4. standalone `log("outside spans", level: "warn", eventName: "e2e.outside")`
5. `flush()` + `shutdown()`
6. process-session phase: re-init with process mode, emit two independent roots, a standalone
   log, and an explicit-session root containing a child and log; generated correlation is shared
   while the explicit session wins for the whole nested context
7. fail-open phase: re-`init()` WITHOUT an api key, run an observed function + `log()`,
   `flush()` + `shutdown()` — must emit nothing and never throw

## Assertions (C1–C26)

Exception content follows the effective per-span capture policy in both SDKs. If either input
or output capture is off, C14 retains error status/type and the exception event but omits its
message and stack. With both enabled, C16 applies the content mask and truncation to
`exception.message` and `exception.stacktrace` before export. A truncated stack keeps the part where
the exception line normally is: TypeScript keeps the start of the stack, and Python keeps the end
of the traceback, with the marker first. Mask failures omit the affected field. The policy is per span, so an enclosing span with capture on records an error that propagates
to it. These rules do not sanitize raw OpenTelemetry events or arbitrary attributes.

| #   | Assertion                                                                                                                                                                                                                                 |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C1  | One trace per session: framework `otel`, userId/sessionId/metadata, token rollup 33/12 with the chat span's nine modality totals, server-computed cost >= 0.5, trace name `support-agent`                                                 |
| C2  | Agent root span: operation `invoke_agent`, `agent_name`, parented to the session parent (not stored); trace id = the session trace id                                                                                                     |
| C3  | Generation span: model/provider/aggregate, cache-read/cache-write, and nine modality token fields/finish reason, extracted inputJson/outputJson, per-span cost                                                                            |
| C4  | Tool span: `tool_name`, extracted arguments/result JSON                                                                                                                                                                                   |
| C5  | Observed function: operation `function`, inputJson `{text}` / outputJson `{formatted}` (via the ingest's `function` extraction branch)                                                                                                    |
| C6  | Every span carries `conversation_id` = sessionId and 16-hex span ids                                                                                                                                                                      |
| C7  | Error span: status `error`, `error_type` = TypeError (TS) / ValueError (Python)                                                                                                                                                           |
| C8  | In-span `log()`: source `otlp-log`, severity 9, eventName, span-correlated, propagated attrs on the record                                                                                                                                |
| C9  | Exception event → error log row (source `span-event`, level `error`, message contains the error)                                                                                                                                          |
| C10 | Standalone `log()`: stored without trace correlation, severity 13, source `otlp-log`                                                                                                                                                      |
| C11 | Metrics: duration histograms for chat/invoke_agent/execute_tool/embeddings; token totals chat 30/12, embeddings 3; first-chunk 1 × 0.25 s; output-chunk intervals count 3, sum 0.21 s, min 0.01 s, max 0.16 s; correct SDK scope          |
| C12 | Embedding span: operation `embeddings`, model extracted                                                                                                                                                                                   |
| C13 | Traceparent join: `background-job` lands in the same trace, parented to the agent, with the conversation id                                                                                                                               |
| C14 | captureInput:false strips content from inputJson and attributesJson                                                                                                                                                                       |
| C15 | `costUsd` client override stored verbatim (0.5)                                                                                                                                                                                           |
| C16 | Oversized content truncated to 65536 chars ending with `...[truncated]`, extracted into inputJson and stored once (source attribute stripped from the attribute columns); mask hook output replaces SECRET-bearing input                  |
| C17 | Exact span census (19) — the fail-open no-key phase contributes nothing                                                                                                                                                                   |
| C18 | Streaming generation: `timeToFirstChunkMs` stored as 250 ms; wire attribute `gen_ai.response.time_to_first_chunk` = 0.25 s                                                                                                                |
| C19 | OpenRouter generation: provider stored as `openrouter`                                                                                                                                                                                    |
| C20 | Session trace: `follow-up-turn` (a second root call of the same session) lands in the same trace, parented to the session parent, started at or after the agent ended                                                                     |
| C21 | Process mode: two otherwise independent roots and a standalone log share one generated UUID session; a nested explicit-session span/log subtree consistently uses the explicit ID instead                                                 |
| C22 | Streaming truncation: `telemetry.dev.capture.truncated = true` survives OTLP transport and ingest on the streamed generation                                                                                                              |
| C23 | Captured JSON matches an independently specified JSON.stringify byte sequence, including number spelling, key ordering, non-finite values and Unicode                                                                                     |
| C24 | Advisor model and native counters survive transport and storage without changing executor usage or rollup totals; provider extraction is covered in the integration packages' tests                                                       |
| C25 | Offered tool names in `gen_ai.tool.definitions` are stored byte-for-byte with input capture off, and the tool description is not; the Anthropic integration tests cover building them from request tools                                  |
| C26 | Prompt spans: `prompted-call` and `prompted-private` carry `td.prompt.name` `support/reply`, `td.prompt.version` 12 (integer) and `td.prompt.label` `production`; only `prompted-call` carries `td.prompt.variables` `{"customer":"Ada"}` |

## Modality publication gate

Do not publish any TypeScript or Python package version that emits the modality attributes in this
contract until the product rollout has completed all of these steps:

1. ClickHouse migration 015 is applied.
2. Every pre-feature ingest replica has drained.
3. The modality-aware trace and span projections are materialized and verified ready.
4. Modality extraction is enabled on every ingest replica and the matching web capability is ready.

Publishing the SDK packages is the final rollout step. Telemetry received before modality extraction
is enabled on ingest is not parsed into modality columns, and later rollout steps do not backfill it.
After extraction is enabled, modality values are stored but are not view-ready until both trace and
span projections and the matching web capability are ready.

## Running

Run the standalone SDK suites from this workspace:

```sh
vp install
vp test
pnpm run py:sync
pnpm run py:test
```

The ingestion assertions require the private product checkout and explicitly isolated PostgreSQL
and ClickHouse endpoints. Follow that repository's verification instructions; this workspace does
not include a database provisioning or migration command. The Python emission scenario remains at
`sdks/python/tests/e2e_scenarios.py` for use by that harness.
