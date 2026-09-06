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

| SDK field (TS / Py)                                                                                                | Wire attribute                                                                                                                       | Notes                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `type: "span"`                                                                                                     | `gen_ai.operation.name = "function"`                                                                                                 | Always explicit: a propagated `gen_ai.conversation.id` would otherwise trip the ingest's `hasGenAi` → `"llm"` heuristic |
| `type: "generation"`                                                                                               | `gen_ai.operation.name = "chat"`                                                                                                     |                                                                                                                         |
| `type: "tool"`                                                                                                     | `gen_ai.operation.name = "execute_tool"`                                                                                             |                                                                                                                         |
| `type: "agent"`                                                                                                    | `gen_ai.operation.name = "invoke_agent"`                                                                                             |                                                                                                                         |
| `type: "embedding"`                                                                                                | `gen_ai.operation.name = "embeddings"`                                                                                               |                                                                                                                         |
| `input` / `output`                                                                                                 | `gen_ai.input.messages` / `gen_ai.output.messages` (`gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` for tools)              | mask → JSON stringify → truncate; gated by capture flags                                                                |
| `model`, `provider`                                                                                                | `gen_ai.request.model`, `gen_ai.provider.name`                                                                                       |                                                                                                                         |
| `systemInstructions` / `system_instructions`                                                                       | `gen_ai.system_instructions`                                                                                                         | gated by captureInput                                                                                                   |
| `responseModel`, `responseId`, `outputType`                                                                        | `gen_ai.response.model`, `gen_ai.response.id`, `gen_ai.output.type`                                                                  |                                                                                                                         |
| `usage.{inputTokens,outputTokens,totalTokens,cacheReadInputTokens,cacheCreationInputTokens,reasoningOutputTokens}` | `gen_ai.usage.{input_tokens,output_tokens,total_tokens,cache_read.input_tokens,cache_creation.input_tokens,reasoning.output_tokens}` | exactly six fields, no passthrough (unknown keys generate `unrecognized_usage_attribute` ingest errors)                 |
| `costUsd` / `cost_usd`                                                                                             | `gen_ai.usage.cost` (number)                                                                                                         | optional override; server pricing (`apps/ingest/src/pricing.ts`) is the default                                         |
| `finishReason`                                                                                                     | `gen_ai.response.finish_reasons = [value]`                                                                                           | single-element array                                                                                                    |
| sampling params                                                                                                    | `gen_ai.request.{temperature,top_p,top_k,max_tokens,stop_sequences,seed,frequency_penalty,presence_penalty}`                         | flat fields                                                                                                             |
| `timeToFirstChunkMs` / `time_to_first_chunk_ms`                                                                    | `gen_ai.response.time_to_first_chunk`                                                                                                | **seconds** on the wire (key has no `ms` suffix; ingest multiplies by 1000)                                             |
| `toolName`, `toolCallId`, `toolDescription`                                                                        | `gen_ai.tool.{name,call.id,description}`                                                                                             |                                                                                                                         |
| `agentName`, `agentId`                                                                                             | `gen_ai.agent.{name,id}`                                                                                                             |                                                                                                                         |
| `metadata.<k>` (per span)                                                                                          | `td.metadata.<k>`                                                                                                                    | reserved keys `userId`/`sessionId`/`user_id`/`session_id` dropped with a debug warning                                  |
| `propagateAttributes userId`                                                                                       | `user.id` on EVERY span/log in scope                                                                                                 | stamped by the span processor's onStart                                                                                 |
| `propagateAttributes sessionId`                                                                                    | `gen_ai.conversation.id` on EVERY span/log in scope                                                                                  | never `session.id`                                                                                                      |
| `propagateAttributes sessionId` on a span that starts with no active parent                                        | remote parent: trace id = SHA-256(apiKey ‖ 0x00 ‖ sessionId)[0:16], parent span id = digest[16:24]                                   | one trace per session per API key; the parent span is never emitted                                                     |
| `init({sessionMode: "process"})` / `init(session_mode="process")`                                                  | fresh UUID as `gen_ai.conversation.id` on spans/logs without an explicit or propagated session                                       | opt-in per enabled initialization; explicit/propagated sessions win; valid parent trace context is never replaced       |
| error                                                                                                              | span status ERROR + `error.type` + `exception` event (`exception.type/message/stacktrace`, `log.severity_number: 17`)                |                                                                                                                         |
| `log()`                                                                                                            | OTLP log record to `/v1/logs` (always; never span events)                                                                            | severity debug=5/info=9/warn=13/error=17; trace-correlated via active context                                           |

Shared constants: env vars `TELEMETRY_DEV_API_KEY` / `TELEMETRY_DEV_BASE_URL` (default
`https://ingest.telemetry.dev`) / `TELEMETRY_DEV_ENVIRONMENT` (default `production`) /
`OTEL_SERVICE_NAME` (default `unknown_service`); truncation cap 65536 chars **including** the
`...[truncated]` marker; batch defaults 64/1000ms/2048/30000ms; instrumentation scopes
`@telemetry-dev/sdk` and `telemetry_dev` (both classify as framework `otel`).

Session mode defaults to `explicit`, preserving independent root traces. `process` mode creates a
new opaque UUID for each enabled SDK initialization and reuses it only for that client's lifetime.
Shutdown followed by init, or re-init replacement, creates a new UUID. It is not a machine-,
deployment-, or cross-process session. Invalid modes fail open through `onError` / `on_error`;
disabled and missing-key initialization remains a no-op and does not create a session.

Auto-metrics: `gen_ai.client.operation.duration` (unit `s`) for operations
{chat, invoke_agent, embeddings, execute_tool}; `gen_ai.client.token.usage` (unit `{token}`,
`gen_ai.token.type` input|output) for {chat, invoke_agent, embeddings}; nothing for `function`.
DELTA temporality; bucket boundaries copied from `packages/ai/src/otel.ts`.

## The shared scenario

Both e2e runners emit exactly this (under
`propagate(userId="user_e2e", sessionId=<random>, metadata={plan:"pro"})`):

The SDK is initialized with a mask hook that replaces any string `gen_ai.input.messages` value
containing `SECRET` with `{masked: true}`.

1. agent span `support-agent` (`agentName: "support"`) containing:
   - generation `chat-completion` (gpt-4o/openai, input "What is the weather?", output
     "It is sunny.", usage 11/7, finishReason stop) with `log("inside generation",
eventName: "e2e.inside")` inside it
   - tool `web-search` (toolCallId `call_1`, input `{q:"weather"}`, output `{hits:1}`)
   - observed function `format-output` (input `{text:"It is sunny."}`, output
     `{formatted:true}`)
   - plain span `broken-step` ended with an error whose message contains "boom"
     (TypeError in TS, ValueError in Python)
   - embedding span `embed-query` (model `text-embedding-4`, usage input 3)
   - plain span `private-step` (input "should-not-appear", captureInput false)
   - plain span `big-payload` (input = 70000 × "x")
   - generation `priced-call` (custom-model-x/custom, usage 5/2, costUsd 0.5)
   - generation `openrouter-call` (provider openrouter, no usage)
   - generation `masked-step` (input "SECRET stuff")
   - generation `streamed-chat` (gpt-4o/openai, input "Stream please", output
     "chunk...", timeToFirstChunkMs 250)
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

## Assertions (C1–C21)

| #   | Assertion                                                                                                                                                                                                                |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| C1  | One trace per session: framework `otel`, userId/sessionId/metadata, token rollup 19/9, server-computed cost >= 0.5, trace name `support-agent`                                                                           |
| C2  | Agent root span: operation `invoke_agent`, `agent_name`, parented to the session parent (not stored); trace id = the session trace id                                                                                    |
| C3  | Generation span: model/provider/tokens/finish reason, extracted inputJson/outputJson, per-span cost                                                                                                                      |
| C4  | Tool span: `tool_name`, extracted arguments/result JSON                                                                                                                                                                  |
| C5  | Observed function: operation `function`, inputJson `{text}` / outputJson `{formatted}` (via the ingest's `function` extraction branch)                                                                                   |
| C6  | Every span carries `conversation_id` = sessionId and 16-hex span ids                                                                                                                                                     |
| C7  | Error span: status `error`, `error_type` = TypeError (TS) / ValueError (Python)                                                                                                                                          |
| C8  | In-span `log()`: source `otlp-log`, severity 9, eventName, span-correlated, propagated attrs on the record                                                                                                               |
| C9  | Exception event → error log row (source `span-event`, level `error`, message contains the error)                                                                                                                         |
| C10 | Standalone `log()`: stored without trace correlation, severity 13, source `otlp-log`                                                                                                                                     |
| C11 | Metrics: duration histograms for chat/invoke_agent/execute_tool/embeddings; token totals chat 16/9, embeddings 3; correct SDK scope on every point                                                                       |
| C12 | Embedding span: operation `embeddings`, model extracted                                                                                                                                                                  |
| C13 | Traceparent join: `background-job` lands in the same trace, parented to the agent, with the conversation id                                                                                                              |
| C14 | captureInput:false strips content from inputJson and attributesJson                                                                                                                                                      |
| C15 | `costUsd` client override stored verbatim (0.5)                                                                                                                                                                          |
| C16 | Oversized content truncated to 65536 chars ending with `...[truncated]`, extracted into inputJson and stored once (source attribute stripped from the attribute columns); mask hook output replaces SECRET-bearing input |
| C17 | Exact span census (14) — the fail-open no-key phase contributes nothing                                                                                                                                                  |
| C18 | Streaming generation: `timeToFirstChunkMs` stored as 250 ms; wire attribute `gen_ai.response.time_to_first_chunk` = 0.25 s                                                                                               |
| C19 | OpenRouter generation: provider stored as `openrouter`                                                                                                                                                                   |
| C20 | Session trace: `follow-up-turn` (a second root call of the same session) lands in the same trace, parented to the session parent, started at or after the agent ended                                                    |
| C21 | Process mode: two otherwise independent roots and a standalone log share one generated UUID session; a nested explicit-session span/log subtree consistently uses the explicit ID instead                                |

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
