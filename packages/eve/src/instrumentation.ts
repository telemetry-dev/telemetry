import type {
  InstrumentationDefinition,
  InstrumentationEvents,
  InstrumentationRuntimeContext,
} from "eve/instrumentation";

import { ensureInit, type ClientOverrides, type TelemetryDevEveOptions } from "./config.ts";

export interface TelemetryDevInstrumentationOptions extends TelemetryDevEveOptions {
  /** ai.telemetry.functionId override; eve defaults it to the agent name. */
  functionId?: string;
  /** Record model inputs on spans. Default: captureInput ?? true. */
  recordInputs?: boolean;
  /** Record model outputs on spans. Default: captureOutput ?? true. */
  recordOutputs?: boolean;
  /** Static runtime context merged into every model-call span. */
  runtimeContext?: InstrumentationRuntimeContext;
  /** User step.started callback, composed with the integration's own. */
  stepStarted?: InstrumentationEvents["step.started"];
}

const envServiceName = (): string | undefined =>
  globalThis.process !== undefined ? process.env.OTEL_SERVICE_NAME : undefined;

function reportError(onError: ((error: Error) => void) | undefined, error: Error): void {
  try {
    onError?.(error);
  } catch {
    // instrumentation callbacks must never fail an Eve turn.
  }
}

export function telemetryDevInstrumentation(
  options: TelemetryDevInstrumentationOptions = {},
  overrides?: ClientOverrides,
): InstrumentationDefinition {
  const { functionId, recordInputs, recordOutputs, runtimeContext, stepStarted, ...sdkOptions } =
    options;

  const definition: InstrumentationDefinition = {
    recordInputs: recordInputs ?? sdkOptions.captureInput ?? true,
    recordOutputs: recordOutputs ?? sdkOptions.captureOutput ?? true,
    setup({ agentName }) {
      try {
        ensureInit(
          {
            ...sdkOptions,
            serviceName: sdkOptions.serviceName ?? envServiceName() ?? agentName,
          },
          overrides,
        );
      } catch (error) {
        reportError(sdkOptions.onError, error instanceof Error ? error : new Error(String(error)));
      }
    },
    events: {
      "step.started"(input) {
        const ctx = { ...runtimeContext } satisfies InstrumentationRuntimeContext;

        const userId =
          input.session.auth.initiator?.principalId ?? input.session.auth.current?.principalId;

        const runtimeContextWithUser = userId ? { ...ctx, "user.id": userId } : ctx;

        try {
          const userResult = stepStarted?.(input);

          if (userResult?.runtimeContext) {
            Object.assign(runtimeContextWithUser, userResult.runtimeContext);
          }
        } catch (error) {
          reportError(
            sdkOptions.onError,
            error instanceof Error ? error : new Error(String(error)),
          );
        }

        return Object.keys(runtimeContextWithUser).length > 0
          ? { runtimeContext: runtimeContextWithUser }
          : undefined;
      },
    },
  };

  if (functionId !== undefined) {
    return { ...definition, functionId };
  }

  return definition;
}
