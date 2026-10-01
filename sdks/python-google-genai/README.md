# telemetry-dev-google-genai

Google GenAI (Gemini) SDK instrumentation for telemetry.dev. It wraps the official `google-genai` Python SDK and emits telemetry.dev generation and embedding spans through `telemetry-dev`.

## Install

```sh
pip install telemetry-dev-google-genai
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
from google import genai
from telemetry_dev_google_genai import wrap_google_genai

client = wrap_google_genai(genai.Client(api_key="..."))

client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[{"role": "user", "parts": [{"text": "Tell me a joke about OpenTelemetry"}]}],
)
```

Use this when you want explicit control over which clients are instrumented. `wrap_google_genai` patches both sync `client.models` and async `client.aio.models`.

## Global instrumentation

```py
from google import genai
from telemetry_dev_google_genai import instrument_google_genai, uninstrument_google_genai

instrument_google_genai()
client = genai.Client(api_key="...")

try:
    client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Tell me a joke about OpenTelemetry",
    )
finally:
    uninstrument_google_genai()
```

Use this as the app-wide one-liner at startup when all GenAI clients should be instrumented.

## Instrumented surfaces

Sync and async variants are covered:

- `client.models.generate_content(...)` / `client.aio.models.generate_content(...)`
- `client.models.generate_content_stream(...)` / `client.aio.models.generate_content_stream(...)`
- `client.models.embed_content(...)` / `client.aio.models.embed_content(...)`
- `generate_images(...)`, `edit_image(...)`, `upscale_image(...)`, and `generate_videos(...)` on sync and async models

`client.chats.create(...).send_message(...)` and `send_message_stream(...)` are covered automatically because they call the wrapped `models` methods.

When `generate_videos(...)` returns a nonterminal operation, its span completes only after
`operations.get(...)` on the same wrapped client observes that operation in a terminal state. An
operation polled by another client or process cannot complete the submission span. If the wrapped
client closes, instrumentation is removed, or the bounded operation tracker evicts an uncompleted
operation, the submission span ends with an error instead of remaining open.

## What gets captured

| Gemini signal | telemetry.dev field / attribute |
|---|---|
| `model` | `model` |
| `contents` | `input` |
| `config.system_instruction` | `system_instructions` |
| sampling params (`temperature`, `top_p`, `top_k`, `seed`, penalties, `max_output_tokens`, `stop_sequences`) | same-named span fields |
| JSON / schema output config | `output_type` (`json` or `text`) |
| `config.candidate_count` | `gen_ai.request.choice.count` |
| `config.tools` | `gen_ai.tool.definitions` |
| `config.tool_config`, `safety_settings`, `thinking_config`, `labels`, `cached_content`, `response_modalities` | `google_genai.request.*` attributes |
| response IDs, model version, finish reasons, output messages, usage tokens (including TEXT/IMAGE/AUDIO prompt, candidate, and cache details) | mapped span fields |
| block/safety/grounding/url-context metadata | `google_genai.response.*` attributes |
| AFC history on the final response | replaces span `input`; sets `google_genai.automatic_function_calling=true` |

The integration maps native Gemini request/response shapes directly. It never mutates caller requests.

Tool definitions (`gen_ai.tool.definitions`, including tool descriptions and parameter schemas) are request attributes, not captured content: `capture_input`, the mask, and `max_attribute_length=0` do not remove them. They are cut only by the OpenTelemetry attribute limit (65,536 characters when `max_attribute_length` is 0).

## Streaming

Gemini streams already include cumulative `usage_metadata` on chunks, so no request injection is needed. Stream spans record time-to-first-chunk, aggregate text parts (merging consecutive text with the same `thought` flag), last-seen usage/finish reasons, and end once when the stream completes, errors, or is closed.

## Automatic function calling (AFC)

When Python callables are passed in `tools`, the SDK may run an internal AFC loop across multiple transport calls. The integration emits one span for the public `generate_content` call and, when present, sets span input to `automatic_function_calling_history`.

## Embeddings

Embedding calls emit `gen_ai.operation.name = "embeddings"`, request model/input, embedding count/dimension attributes, optional token usage, and optional billable character counts. Embedding vectors are not captured as output.

## Provider values

- Gemini Developer API clients record provider `gcp.gemini`.
- Vertex AI clients (`vertexai=True`) record provider `gcp.vertex_ai`.

## Fail-open guarantee

Mapping code is defensive; wrapped calls return the SDK response unchanged and re-raise exceptions untouched. Telemetry bugs never break callers.

## Limitations

- Not instrumented: `count_tokens`, `compute_tokens`, `live`, `caches`, `files`, `tunings`, `batches`.
- Media operations capture text prompts, model/provider attribution, non-binary media references such as URIs and MIME types, and `gen_ai.operation.name=generate_content`. Image, video, and reference-image bytes are never captured. Image and video operations emit the corresponding `gen_ai.output.type`.
- Deliberately not captured: logprobs, citation metadata, `create_time`, `sdk_http_response`.
- Unconsumed streams end their spans only when the stream is exhausted, errors, or is closed.
