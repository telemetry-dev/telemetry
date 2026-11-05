import { type ClientOverrides, init, shutdown, type TelemetryOptions } from "@telemetry-dev/sdk";

/** Options accepted by `telemetryDevProcessor`; see `TelemetryOptions` of `@telemetry-dev/sdk`. */
export type TelemetryDevOpenAIAgentsOptions = Omit<TelemetryOptions, "sdkName">;

export type { ClientOverrides };

let initialized = false;

/**
 * Initializes the telemetry.dev SDK once per process for this integration; whichever processor is
 * created first wins. The processor builds spans with the SDK's own tracer, so it does not need
 * the global OpenTelemetry provider.
 */
export function ensureInit(
  options: TelemetryDevOpenAIAgentsOptions = {},
  overrides?: ClientOverrides,
): void {
  if (initialized) return;
  initialized = true;
  init({ ...options, sdkName: "@telemetry-dev/openai-agents" }, overrides);
}

/** Test seam for unit tests that need a fresh SDK singleton. */
export async function resetForTesting(): Promise<void> {
  initialized = false;
  await shutdown();
}
