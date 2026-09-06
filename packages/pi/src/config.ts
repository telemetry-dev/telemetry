import { init, shutdown, type ClientOverrides, type TelemetryOptions } from "@telemetry-dev/sdk";

/**
 * SDK options accepted by @telemetry-dev/pi. `registerGlobal` is excluded on
 * purpose: pi runs its own OpenTelemetry pipeline in-process, so this
 * integration never touches the global provider — every span is created
 * through the SDK's own tracer with explicit parenting.
 */
export type TelemetryDevPiOptions = Omit<TelemetryOptions, "registerGlobal" | "sdkName">;

export type { ClientOverrides };

let initialized = false;

/** Initializes the telemetry.dev SDK exactly once per process for this integration. */
export function ensureInit(options: TelemetryDevPiOptions = {}, overrides?: ClientOverrides): void {
  if (initialized) return;
  initialized = true;
  init(
    {
      ...options,
      serviceName: options.serviceName ?? process.env.OTEL_SERVICE_NAME ?? "pi",
      sdkName: "@telemetry-dev/pi",
      // Enforce the guarantee above at runtime for untyped (plain JS) callers.
      registerGlobal: false,
    },
    overrides,
  );
}

/** Test seam for unit tests that need a fresh SDK singleton. */
export async function resetForTesting(): Promise<void> {
  initialized = false;
  await shutdown();
}
