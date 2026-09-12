import { TelemetrySpanProcessor, type TelemetrySpanProcessorOptions } from "@telemetry-dev/sdk";
import { otelIntegration, type OtelIntegration } from "eve/instrumentation/otel";

import { isAiScope } from "./config.ts";

export interface TelemetryDevOtelIntegrationOptions extends Omit<
  TelemetrySpanProcessorOptions,
  "spanExporter"
> {
  recordInputs?: boolean;
  recordOutputs?: boolean;
}

/**
 * telemetry.dev as a destination in eve's `agent/instrumentation/` provider layout
 * (`experimental.instrumentationProviders`). Export it from one file in that directory.
 * eve owns the tracer provider there, so this attaches a processor instead of registering one.
 */
export function telemetryDevOtelIntegration(
  options: TelemetryDevOtelIntegrationOptions = {},
): OtelIntegration {
  const { recordInputs, recordOutputs, ...processorOptions } = options;

  return otelIntegration({
    recordInputs,
    recordOutputs,
    spanProcessors: [
      new TelemetrySpanProcessor({
        ...processorOptions,
        spanFilter:
          processorOptions.spanFilter ?? ((span) => isAiScope(span.instrumentationScope.name)),
      }),
    ],
  });
}
