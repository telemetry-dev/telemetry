# @telemetry-dev/otel

Bring-your-own OpenTelemetry base layer for [telemetry.dev](https://telemetry.dev). It provides the span processor and OTLP/protobuf trace exporter used by `@telemetry-dev/sdk`, without requiring the SDK runtime.

```sh
npm i @telemetry-dev/otel
```

`@opentelemetry/api` is a peer dependency and is installed automatically by current npm and pnpm versions.

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

## Options

| Option         | Fallback                                                      | Default / behavior                                                                                                    |
| -------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `apiKey`       | `TELEMETRY_DEV_API_KEY`                                       | When absent and no `spanExporter` override is provided, the processor is a no-op.                                     |
| `baseUrl`      | `TELEMETRY_DEV_BASE_URL`, then `https://ingest.telemetry.dev` | Trailing slashes are removed before building OTLP URLs.                                                               |
| `exportMode`   | —                                                             | `"batched"`; use `"immediate"` for serverless paths that flush explicitly.                                            |
| `batch`        | —                                                             | Merged over `{ maxExportBatchSize: 64, scheduledDelayMillis: 1000, maxQueueSize: 2048, exportTimeoutMillis: 30000 }`. |
| `spanFilter`   | —                                                             | Exports every span by default because attaching the processor is an explicit BYO choice.                              |
| `metrics`      | —                                                             | `true`; records GenAI duration/token histograms when an API key is present.                                           |
| `serviceName`  | `OTEL_SERVICE_NAME`                                           | `unknown_service`; used on the metrics resource.                                                                      |
| `environment`  | `TELEMETRY_DEV_ENVIRONMENT`                                   | `production`; used as `deployment.environment.name` on the metrics resource.                                          |
| `fetch`        | —                                                             | `globalThis.fetch`; override for tests or custom runtimes.                                                            |
| `onError`      | —                                                             | Receives export and processor errors; errors are also reported through diagnostics.                                   |
| `spanExporter` | —                                                             | Advanced/test seam replacing the built-in OTLP trace exporter.                                                        |

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
