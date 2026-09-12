import {
  init,
  SCOPE_NAME,
  shutdown,
  type ClientOverrides,
  type TelemetryOptions,
} from "@telemetry-dev/sdk";

/** Options accepted by every @telemetry-dev/eve entry point. */
export type TelemetryDevEveOptions = Omit<TelemetryOptions, "registerGlobal" | "sdkName">;

export type { ClientOverrides };

let initialized = false;

// eve's agent spans ("eve" in agent/instrumentation.ts, "eve.agent" in the provider layout), the
// AI SDK's model-call and tool spans ("gen_ai", emitted by @ai-sdk/otel), and this SDK's own
// client-wrapper spans. Everything else on the global provider (better-auth, Nitro, eve's
// "workflow" engine) is not AI telemetry.
const AI_SCOPES = {
  eve: true,
  "eve.agent": true,
  gen_ai: true,
  [SCOPE_NAME]: true,
} satisfies Record<string, true>;

export const isAiScope = (scope: string): boolean => Object.hasOwn(AI_SCOPES, scope);

/**
 * Initializes the telemetry.dev SDK exactly once per process for this integration.
 * registerGlobal:true lets eve's tracers resolve to the SDK provider. spanFilter defaults to
 * exporting only AI scopes (see AI_SCOPES); pass spanFilter: () => true to export every span.
 * eve starts each `ai.eve.turn` under a workflow-engine span of its own trace, so the turn is
 * re-rooted under the session parent and every turn of a session lands in one trace.
 */
export function ensureInit(
  options: TelemetryDevEveOptions = {},
  overrides?: ClientOverrides,
): void {
  if (initialized) return;
  initialized = true;
  init(
    {
      ...options,
      sdkName: "@telemetry-dev/eve",
      registerGlobal: true,
      spanFilter: options.spanFilter ?? ((span) => isAiScope(span.instrumentationScope.name)),
      sessionRootOf: (name, attributes) => {
        const sessionId = attributes["eve.session.id"];

        return name === "ai.eve.turn" && typeof sessionId === "string" ? sessionId : undefined;
      },
    },
    overrides,
  );
}

/** Test seam for unit tests that need a fresh SDK singleton. */
export async function resetForTesting(): Promise<void> {
  initialized = false;
  await shutdown();
}
