# @telemetry-dev/google-genai

Google GenAI (Gemini) SDK instrumentation for telemetry.dev. Wraps the official `@google/genai` client and emits generation and embedding spans through `@telemetry-dev/sdk`.

## Install

```sh
pnpm add @telemetry-dev/sdk @telemetry-dev/google-genai @google/genai
```

Initialize the core SDK first:

```ts
import { init, flush, shutdown } from "@telemetry-dev/sdk";

init({
  apiKey: process.env.TELEMETRY_DEV_API_KEY,
  baseUrl: process.env.TELEMETRY_DEV_BASE_URL,
  serviceName: "my-service",
});

// app code...
await flush();
await shutdown();
```

## Quickstart

```ts
import { GoogleGenAI } from "@google/genai";
import { wrapGoogleGenAI } from "@telemetry-dev/google-genai";

const ai = wrapGoogleGenAI(new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! }));

await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: "Tell me a joke about OpenTelemetry",
  config: {
    systemInstruction: "Be concise",
  },
});
```

Call `wrapGoogleGenAI` on each `GoogleGenAI` client you want instrumented. Wrapping is idempotent.

## What gets captured

Span names are `chat ${model}` for generation calls and `embeddings ${model}` for embeddings. Provider is `gcp.gemini` for the Gemini Developer API and `gcp.vertex_ai` when `client.vertexai === true`.

| Gemini request/response                                 | telemetry.dev field / attribute                                                                                                  |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `model`                                                 | `gen_ai.request.model`                                                                                                           |
| `contents`                                              | `gen_ai.input.messages`                                                                                                          |
| `config.systemInstruction`                              | `gen_ai.system_instructions`                                                                                                     |
| `config.temperature`, `topP`, `topK`, `seed`, penalties | same-named request attributes                                                                                                    |
| `config.maxOutputTokens`                                | `gen_ai.request.max_tokens`                                                                                                      |
| `config.stopSequences`                                  | `gen_ai.request.stop_sequences`                                                                                                  |
| JSON / schema output config                             | `gen_ai.output.type`                                                                                                             |
| `config.candidateCount`                                 | `gen_ai.request.choice.count`                                                                                                    |
| `config.tools`                                          | `gen_ai.tool.definitions`                                                                                                        |
| `config.toolConfig`                                     | `google_genai.request.tool_config`                                                                                               |
| `config.safetySettings`                                 | `google_genai.request.safety_settings`                                                                                           |
| `config.thinkingConfig`                                 | `google_genai.request.thinking_config`                                                                                           |
| `config.labels`                                         | `google_genai.request.labels`                                                                                                    |
| `config.cachedContent`                                  | `google_genai.request.cached_content`                                                                                            |
| `config.responseModalities`                             | `google_genai.request.response_modalities`                                                                                       |
| `modelVersion`, `responseId`                            | `gen_ai.response.model`, `gen_ai.response.id`                                                                                    |
| `candidates[].finishReason`                             | `gen_ai.response.finish_reasons`                                                                                                 |
| `candidates[].content`                                  | `gen_ai.output.messages`                                                                                                         |
| `usageMetadata.*` token fields                          | `gen_ai.usage.*` (+ `google_genai.usage.tool_use_prompt_tokens`)                                                                 |
| `promptFeedback.blockReason`                            | `google_genai.response.block_reason` (+ message attr)                                                                            |
| Safety / grounding metadata                             | `google_genai.response.safety_ratings`, `google_genai.response.grounding_metadata`, `google_genai.response.url_context_metadata` |
| AFC history present                                     | span `input` replaced with `automaticFunctionCallingHistory`, `google_genai.automatic_function_calling=true`                     |
| Embedding stats                                         | `google_genai.response.embedding_count`, `google_genai.response.embedding_dimensions`, `google_genai.usage.billable_characters`  |

`config.httpOptions` and `config.abortSignal` are never recorded. Embedding vectors are never captured as output.

## Streaming

`generateContentStream` emits one span per call. Chunks pass through unchanged. The integration aggregates text parts (merging consecutive text with the same `thought` flag), finish reasons, usage, and block/safety metadata across chunks, records `gen_ai.response.time_to_first_chunk` on the first chunk, and ends the span once when the stream completes, errors, or the consumer stops early. Streams that are never consumed do not finish their span.

## Chats

`client.chats.create(...).sendMessage` / `sendMessageStream` route through the wrapped `models.generateContent` methods, so chat sessions are traced automatically whether the chat is created before or after wrapping.

## Embeddings

`embedContent` emits `gen_ai.operation.name = "embeddings"` with request model/input, embedding count/dimensions, optional token usage, and billable character metadata. Vectors are not captured.

## Why no global instrumentation in TypeScript

In `@google/genai` v2, `Models.generateContent`, `generateContentStream`, and `embedContent` are arrow-function **instance fields**, not prototype methods. Prototype patching cannot intercept them, so this package exposes per-client `wrapGoogleGenAI` only.

## Fail-open guarantee

Mapping and span lifecycle code is defensive. Wrapped calls preserve the original return values and thrown errors. If telemetry is not initialized, wrapped calls behave exactly like the unwrapped SDK with zero spans emitted.

## Automatic function calling (AFC)

When the SDK performs an internal AFC loop for callable tools, one wrapped `generateContent` call produces one span covering the whole loop. Span input is replaced with `automaticFunctionCallingHistory` when present.

## Not instrumented

`countTokens`, `computeTokens`, `generateImages`, `generateVideos`, `live`, `caches`, `files`, `tunings`, and `batches` are not wrapped in this package.
