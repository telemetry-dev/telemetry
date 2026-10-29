# @telemetry-dev/otel

Bring-your-own OpenTelemetry base layer for [telemetry.dev](https://telemetry.dev). It provides the span processor and OTLP/protobuf trace exporter used by `@telemetry-dev/sdk`, without requiring the SDK runtime.

Docs: https://telemetry.dev/docs/sdk/opentelemetry

```sh
npm i @telemetry-dev/otel
```

Requires Node.js 20.19.0 or later. `@opentelemetry/api` 1.9 or later 1.x is a peer dependency and is installed automatically by current npm and pnpm versions.

## NodeSDK

```ts
import { NodeSDK } from "@opentelemetry/sdk-node";
import { TelemetrySpanProcessor } from "@telemetry-dev/otel";

const sdk = new NodeSDK({
  spanProcessors: [new TelemetrySpanProcessor()],
});

sdk.start();
```

## Vercel / Next.js `registerOTel`

```ts
import { registerOTel } from "@vercel/otel";
import { TelemetrySpanProcessor } from "@telemetry-dev/otel";

export function register() {
  registerOTel({
    serviceName: "my-app",
    spanProcessors: [new TelemetrySpanProcessor()],
  });
}
```

## BasicTracerProvider

```ts
import { trace } from "@opentelemetry/api";
import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import { TelemetrySpanProcessor } from "@telemetry-dev/otel";

const provider = new BasicTracerProvider({
  spanProcessors: [new TelemetrySpanProcessor()],
});

trace.setGlobalTracerProvider(provider);
```

## Native Claude SDK spans

`@anthropic-ai/sdk` 0.134.0 and later emits OpenTelemetry spans natively. The Claude SDK uses the global tracer provider of the application.
To send these spans to telemetry.dev, register a global tracer provider with `TelemetrySpanProcessor`:

```sh
npm i @telemetry-dev/otel @opentelemetry/sdk-trace-node @anthropic-ai/sdk
```

```ts
import Anthropic from "@anthropic-ai/sdk";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { TelemetrySpanProcessor } from "@telemetry-dev/otel";

const provider = new NodeTracerProvider({
  spanProcessors: [new TelemetrySpanProcessor()],
});
provider.register();

const anthropic = new Anthropic();
await anthropic.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 256,
  messages: [{ role: "user", content: "Say hello." }],
});

await provider.shutdown();
```

By default, the spans contain metadata only. To also send prompts and responses, set `ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE=content`. The client reads this variable one time, when you make the client.

Do not use the native spans and `@telemetry-dev/anthropic` together. If you use the two, each request has two spans.

You can also send the native spans without this package. Use an OTLP/HTTP protobuf exporter:

```sh
npm i @opentelemetry/sdk-trace-node @opentelemetry/exporter-trace-otlp-proto
```

```ts
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { BatchSpanProcessor, NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

const exporter = new OTLPTraceExporter({
  url: "https://api.telemetry.dev/v1/traces",
  headers: { Authorization: `Bearer ${process.env.TELEMETRY_DEV_API_KEY}` },
});
const provider = new NodeTracerProvider({
  spanProcessors: [new BatchSpanProcessor(exporter)],
});
provider.register();
```

The Claude API can also send server-side spans to telemetry.dev through a Console destination. Set the endpoint to `https://api.telemetry.dev`. Set the `Authorization` header to `Bearer <telemetry.dev API key>`. For more information, refer to [Claude Platform](https://telemetry.dev/docs/integrations/claude-platform).

When the SDK and the Claude API send the spans of a request to the same telemetry.dev project, telemetry.dev counts the tokens of the request one time, from the API span. The SDK span stays in the trace with no tokens. telemetry.dev matches the two spans by the request ID or the response ID, so the API span does not have to be a child of the SDK span. If the Console destination uses an API key that is scoped to one environment, its API spans take over only the SDK spans of that environment.

## Options

| Option         | Fallback                                                   | Default / behavior                                                                                                    |
| -------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `apiKey`       | `TELEMETRY_DEV_API_KEY`                                    | When absent and no `spanExporter` override is provided, the processor is a no-op.                                     |
| `baseUrl`      | `TELEMETRY_DEV_BASE_URL`, then `https://api.telemetry.dev` | Trailing slashes are removed before building OTLP URLs.                                                               |
| `exportMode`   | —                                                          | `"batched"`; use `"immediate"` for serverless paths that flush explicitly.                                            |
| `batch`        | —                                                          | Merged over `{ maxExportBatchSize: 64, scheduledDelayMillis: 1000, maxQueueSize: 2048, exportTimeoutMillis: 30000 }`. |
| `spanFilter`   | —                                                          | Exports every span by default because attaching the processor is an explicit BYO choice.                              |
| `metrics`      | —                                                          | `true`; records GenAI duration/token histograms when an API key is present.                                           |
| `serviceName`  | `OTEL_SERVICE_NAME`                                        | `unknown_service`; used on the metrics resource.                                                                      |
| `environment`  | `TELEMETRY_DEV_ENVIRONMENT`                                | `production`; used as `deployment.environment.name` on the metrics resource.                                          |
| `fetch`        | —                                                          | `globalThis.fetch`; override for tests or custom runtimes.                                                            |
| `onError`      | —                                                          | Receives export and processor errors, which diagnostics also get. A series of failed exports is one report.           |
| `spanExporter` | —                                                          | Advanced/test seam replacing the built-in OTLP trace exporter.                                                        |

Earlier versions sent data to `https://ingest.telemetry.dev` by default. That hostname reaches the
same service and stays supported: if a firewall allows only it, set `baseUrl` or
`TELEMETRY_DEV_BASE_URL` to it.

## Missing API key behavior

`new TelemetrySpanProcessor()` never throws. Without an API key, and without a `spanExporter` override, it becomes a no-op processor and emits only a debug-level diagnostic. This makes it safe to construct during local development or in environments where telemetry is disabled.

## Raw span exporter

Use `createTelemetrySpanExporter()` when you want to keep your own `BatchSpanProcessor` or `SimpleSpanProcessor` but send OTLP/HTTP protobuf payloads to telemetry.dev.

```ts
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { createTelemetrySpanExporter } from "@telemetry-dev/otel";

const processor = new BatchSpanProcessor(createTelemetrySpanExporter());
```

The exporter accepts `apiKey`, `baseUrl`, `fetch`, `onError`, and `exportTimeoutMillis`. The timeout caps each export request and defaults to 30,000 ms. The API key and base URL use the same environment variable fallbacks as `TelemetrySpanProcessor`.

## Correlation attributes

`propagateAttributes({ userId, sessionId, metadata }, fn)` works standalone with BYO providers. The function stores correlation attributes in the same OTel context key that `TelemetrySpanProcessor` reads on span start, so spans created inside `fn` receive `user.id`, `gen_ai.conversation.id`, and `td.metadata.*` attributes without installing `@telemetry-dev/sdk`.

`TelemetrySpanProcessor` on your provider adds `gen_ai.conversation.id`, but it cannot change trace IDs. Use `withSessionParent(context.active(), sessionId, apiKey)` for each root span. A nonempty API key and session ID put each session root in the same trace. A deterministic session parent (`SHA-256(apiKey ‖ 0x00 ‖ sessionId)`) is the parent, but the SDK does not send it. If one value is empty, the function keeps the input context.

**Install `sessionSampler(yourSampler)` on the provider before you use `withSessionParent` or `sessionRootTracerProvider`.** The OTel API does not expose a provider's sampler. A context helper alone cannot keep the root policy. Pass the existing programmatic sampler to the wrapper. Without an argument, `sessionSampler()` reads `OTEL_TRACES_SAMPLER` and `OTEL_TRACES_SAMPLER_ARG`. It does not inspect an existing provider.

The wrapper uses the original parent context and the deterministic session trace ID for the sampling decision. Real active and explicit parents keep their sampling decisions. The framework-root wrapper reparents recognized turns inside workflow spans, but keeps the workflow parent's sampling policy and trace state. Context values and baggage stay intact.

`sessionSpanContext` only calculates IDs. Its flags are not a sampling decision. Do not use it directly as a parent.

```ts
import { context } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  ParentBasedSampler,
  TraceIdRatioBasedSampler,
} from "@opentelemetry/sdk-trace-base";
import { sessionSampler, TelemetrySpanProcessor, withSessionParent } from "@telemetry-dev/otel";

const sampler = new ParentBasedSampler({ root: new TraceIdRatioBasedSampler(0.1) });
const provider = new BasicTracerProvider({
  sampler: sessionSampler(sampler),
  spanProcessors: [new TelemetrySpanProcessor()],
});
const span = provider
  .getTracer("my-app")
  .startSpan(
    "turn",
    {},
    withSessionParent(context.active(), sessionId, process.env.TELEMETRY_DEV_API_KEY),
  );
span.end();
```

The environment modes are `always_on`, `always_off`, `traceidratio`, `parentbased_always_on` (default), `parentbased_always_off`, and `parentbased_traceidratio`.
Ratio modes accept a finite argument in `[0, 1]`. Missing, invalid, or out-of-range arguments use `1` and cause an OTel diagnostic. Unknown modes use `parentbased_always_on`.
The TypeScript and Python SDKs share deterministic session IDs. Their built-in ratio samplers can select different sessions because the OTel algorithms differ.

## License

MIT. See [LICENSE](https://github.com/telemetry-dev/telemetry/blob/main/packages/otel/LICENSE).
