import { context as otelContext, SpanKind, SpanStatusCode, type Span } from "@opentelemetry/api";
import type { TokenUsage } from "@tanstack/ai";
import {
  createGenerationEmitter,
  type GenerationEmitterOverrides,
  omitUndefined,
} from "@telemetry-dev/otel";

import { resolveConfig, type TelemetryDevOptions } from "./config.ts";
import { errorDetails } from "./errors.ts";

export interface GenerationMiddlewareContextCompat {
  requestId: string;
  activity: string;
  provider: string;
  model: string;
  context: unknown;
}

interface TerminalInfo {
  duration: number;
}

export interface GenerationMiddlewareCompat {
  name?: string;
  onStart?: (ctx: GenerationMiddlewareContextCompat) => void | Promise<void>;
  onUsage?: (ctx: GenerationMiddlewareContextCompat, usage: TokenUsage) => void | Promise<void>;
  onFinish?: (
    ctx: GenerationMiddlewareContextCompat,
    info: TerminalInfo & { usage?: TokenUsage },
  ) => void | Promise<void>;
  onError?: (
    ctx: GenerationMiddlewareContextCompat,
    info: TerminalInfo & { error: unknown },
  ) => void | Promise<void>;
  onAbort?: (
    ctx: GenerationMiddlewareContextCompat,
    info: TerminalInfo & { reason?: string },
  ) => void | Promise<void>;
}

interface EvaluationState {
  span: Span;
  usage?: TokenUsage;
}

export function generationTelemetryDev(
  options?: TelemetryDevOptions,
  overrides?: GenerationEmitterOverrides,
): GenerationMiddlewareCompat {
  const config = resolveConfig(options);

  if (!config.apiKey) return { name: "telemetry-dev" };

  const emitter = createGenerationEmitter(
    {
      ...config,
      sdkName: "@telemetry-dev/tanstack-ai",
      onError: config.onError
        ? (error) => config.onError?.(error instanceof Error ? error : new Error(error))
        : undefined,
    },
    overrides,
  );

  const states = new WeakMap<object, EvaluationState>();

  const attributes = (ctx: GenerationMiddlewareContextCompat, usage?: TokenUsage) =>
    omitUndefined({
      "gen_ai.operation.name": "evaluate",
      "gen_ai.provider.name": ctx.provider,
      "gen_ai.request.model": ctx.model,
      "gen_ai.request.id": ctx.requestId,
      "gen_ai.usage.input_tokens": usage?.promptTokens,
      "gen_ai.usage.output_tokens": usage?.completionTokens,
      "gen_ai.usage.cache_read.input_tokens": usage?.promptTokensDetails?.cachedTokens,
      "gen_ai.usage.cache_creation.input_tokens": usage?.promptTokensDetails?.cacheWriteTokens,
      "gen_ai.usage.reasoning.output_tokens": usage?.completionTokensDetails?.reasoningTokens,
      "gen_ai.usage.cost": usage?.cost,
    });

  const finish = async (
    ctx: GenerationMiddlewareContextCompat,
    info: TerminalInfo,
    terminal?: { type: string; message: string },
    finishUsage?: TokenUsage,
  ) => {
    const state = states.get(ctx as object);

    if (!state) return;
    states.delete(ctx as object);
    const usage = finishUsage ?? state.usage;
    state.span.setAttributes(attributes(ctx, usage));

    if (terminal) {
      state.span.setStatus({ code: SpanStatusCode.ERROR, message: terminal.message });
      state.span.setAttribute("error.type", terminal.type);
      state.span.addEvent("exception", {
        "exception.type": terminal.type,
        "exception.message": terminal.message,
      });
    }

    state.span.end();

    const metricAttributes = omitUndefined({
      "gen_ai.operation.name": "evaluate",
      "gen_ai.provider.name": ctx.provider,
      "gen_ai.request.model": ctx.model,
    });

    emitter.recordDuration(Math.max(info.duration, 0) / 1000, metricAttributes);

    if (usage?.promptTokens !== undefined) {
      emitter.recordTokens("input", usage.promptTokens, metricAttributes);
    }

    if (usage?.completionTokens !== undefined) {
      emitter.recordTokens("output", usage.completionTokens, metricAttributes);
    }

    await emitter.flush([state.span]);
  };

  return {
    name: "telemetry-dev",
    onStart(ctx) {
      try {
        if (ctx.activity !== "evaluate") return;

        const span = emitter.tracer.startSpan(
          "evaluate",
          { kind: SpanKind.CLIENT, attributes: attributes(ctx) },
          otelContext.active(),
        );

        states.set(ctx as object, { span });
      } catch (error) {
        config.onError?.(error);
      }
    },
    onUsage(ctx, usage) {
      const state = states.get(ctx as object);

      if (state) state.usage = usage;
    },
    onFinish(ctx, info) {
      return finish(ctx, info, undefined, info.usage).catch((error) => config.onError?.(error));
    },
    onError(ctx, info) {
      return finish(ctx, info, errorDetails(info.error)).catch((error) => config.onError?.(error));
    },
    onAbort(ctx, info) {
      return finish(ctx, info, {
        type: "cancelled",
        message: info.reason ?? "cancelled",
      }).catch((error) => config.onError?.(error));
    },
  };
}
