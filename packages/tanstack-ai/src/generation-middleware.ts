import {
  type Context,
  context as otelContext,
  SpanKind,
  SpanStatusCode,
  type Span,
} from "@opentelemetry/api";
import type { TokenUsage } from "@tanstack/ai";
import {
  createGenerationEmitter,
  type GenerationEmitterOverrides,
  omitUndefined,
  withSessionParent,
} from "@telemetry-dev/otel";

import { resolveConfig, type TelemetryDevOptions } from "./config.ts";
import { errorDetails } from "./errors.ts";
import { inputTokens } from "./usage.ts";

export interface GenerationMiddlewareContextCompat {
  requestId: string;
  activity: string;
  provider: string;
  model: string;
  context: unknown;
  runId?: string;
  /** Present on TanStack AI versions that let middleware observe the activity result. */
  resultTransforms?: Array<(result: never, ctx: never) => unknown>;
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

interface Operation {
  name: string;
  spanName: string;
  outputType?: string;
}

// Keyed by TanStack AI's open `activity` string; activities without an entry get no span.
interface OperationsByActivity {
  [activity: string]: Operation | undefined;
}

// Operation names match the other TypeScript integrations: `embeddings` for embedding calls,
// `generate_content` plus `gen_ai.output.type` for media, and `chat` for text generation.
const OPERATIONS: OperationsByActivity = {
  evaluate: { name: "evaluate", spanName: "evaluate" },
  embedding: { name: "embeddings", spanName: "embeddings" },
  summarize: { name: "chat", spanName: "summarize", outputType: "text" },
  image: { name: "generate_content", spanName: "generate_content image", outputType: "image" },
  tts: { name: "generate_content", spanName: "generate_content speech", outputType: "speech" },
  transcription: {
    name: "generate_content",
    spanName: "generate_content text",
    outputType: "text",
  },
  video: { name: "generate_content", spanName: "generate_content video", outputType: "video" },
};

interface GenerationState {
  operation: Operation;
  parent: Context;
  span?: Span;
  usage?: TokenUsage;
  responseId?: string;
  responseModel?: string;
}

function readString(value: unknown, key: string): string | undefined {
  if (value === null || typeof value !== "object") return undefined;

  try {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    const field = descriptor && "value" in descriptor ? descriptor.value : undefined;

    return typeof field === "string" && field.length > 0 ? field : undefined;
  } catch {
    return undefined;
  }
}

// TanStack AI keys a video job's later status calls with `video:<provider>:<jobId>` as the runId.
function videoJobId(ctx: GenerationMiddlewareContextCompat): string | undefined {
  const runId = ctx.runId;

  if (ctx.activity !== "video" || typeof runId !== "string") return undefined;
  const prefix = `video:${encodeURIComponent(ctx.provider)}:`;

  if (!runId.startsWith(prefix) || runId.length === prefix.length) return undefined;

  try {
    return decodeURIComponent(runId.slice(prefix.length));
  } catch {
    return undefined;
  }
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

  const states = new WeakMap<object, GenerationState>();
  // Contexts already finalized. TanStack AI can report onError for a context whose onFinish has
  // run (a later middleware's onFinish throws inside the streamed video try block).
  const finished = new WeakSet<object>();

  const attributes = (
    ctx: GenerationMiddlewareContextCompat,
    state: GenerationState,
    usage?: TokenUsage,
  ) =>
    omitUndefined({
      "gen_ai.operation.name": state.operation.name,
      "gen_ai.provider.name": ctx.provider,
      "gen_ai.request.model": ctx.model,
      "gen_ai.request.id": ctx.requestId,
      "gen_ai.output.type": state.operation.outputType,
      "gen_ai.response.id": state.responseId,
      "gen_ai.response.model": state.responseModel,
      "gen_ai.usage.input_tokens": inputTokens(ctx.provider, usage),
      "gen_ai.usage.output_tokens": usage?.completionTokens,
      "gen_ai.usage.cache_read.input_tokens": usage?.promptTokensDetails?.cachedTokens,
      "gen_ai.usage.cache_creation.input_tokens": usage?.promptTokensDetails?.cacheWriteTokens,
      "gen_ai.usage.reasoning.output_tokens": usage?.completionTokensDetails?.reasoningTokens,
      "gen_ai.usage.text.input_tokens": usage?.promptTokensDetails?.textTokens,
      "gen_ai.usage.text.output_tokens": usage?.completionTokensDetails?.textTokens,
      "gen_ai.usage.image.input_tokens": usage?.promptTokensDetails?.imageTokens,
      "gen_ai.usage.image.output_tokens": usage?.completionTokensDetails?.imageTokens,
      "gen_ai.usage.audio.input_tokens": usage?.promptTokensDetails?.audioTokens,
      "gen_ai.usage.audio.output_tokens": usage?.completionTokensDetails?.audioTokens,
      "gen_ai.usage.cost": usage?.cost,
    });

  const newState = (ctx: GenerationMiddlewareContextCompat): GenerationState | undefined => {
    const operation = OPERATIONS[ctx.activity];

    if (!operation) return undefined;
    const jobId = videoJobId(ctx);

    // Job creation and the status call that sees the job finish run in separate activity calls,
    // possibly in separate processes. Both derive their trace from the job's runId.
    const parent = jobId
      ? withSessionParent(otelContext.active(), ctx.runId, config.apiKey)
      : otelContext.active();

    return { operation, parent, responseId: jobId };
  };

  const startSpan = (
    ctx: GenerationMiddlewareContextCompat,
    state: GenerationState,
    startTime?: number,
  ) =>
    emitter.tracer.startSpan(
      state.operation.spanName,
      { kind: SpanKind.CLIENT, attributes: attributes(ctx, state), startTime },
      state.parent,
    );

  const finish = async (
    ctx: GenerationMiddlewareContextCompat,
    info: TerminalInfo,
    terminal?: { type: string; message: string },
    finishUsage?: TokenUsage,
    recordDuration = true,
  ) => {
    // A video status call that fails reports onError without onStart; a context that already
    // finished must not get a second span.
    const state =
      states.get(ctx as object) ??
      (ctx.activity === "video" && !finished.has(ctx as object) ? newState(ctx) : undefined);

    if (!state) return;
    finished.add(ctx as object);
    states.delete(ctx as object);
    const duration = Math.max(info.duration, 0);
    const span = state.span ?? startSpan(ctx, state, Date.now() - duration);
    const usage = finishUsage ?? state.usage;
    span.setAttributes(attributes(ctx, state, usage));

    if (terminal) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: terminal.message });
      span.setAttribute("error.type", terminal.type);
      span.addEvent("exception", {
        "exception.type": terminal.type,
        "exception.message": terminal.message,
      });
    }

    span.end();

    const metricAttributes = omitUndefined({
      "gen_ai.operation.name": state.operation.name,
      "gen_ai.provider.name": ctx.provider,
      "gen_ai.request.model": ctx.model,
    });

    if (recordDuration) emitter.recordDuration(duration / 1000, metricAttributes);

    const input = inputTokens(ctx.provider, usage);

    if (input !== undefined) emitter.recordTokens("input", input, metricAttributes);

    if (usage?.completionTokens !== undefined) {
      emitter.recordTokens("output", usage.completionTokens, metricAttributes);
    }

    await emitter.flush([span]);
  };

  // Reads the response model and video job id from the result. Never rewrites or throws.
  const observeResult = (ctx: GenerationMiddlewareContextCompat, state: GenerationState) => {
    if (!Array.isArray(ctx.resultTransforms)) return;

    ctx.resultTransforms.push(async (result: unknown) => {
      try {
        state.responseModel = readString(result, "model") ?? state.responseModel;

        if (ctx.activity !== "video") return undefined;
        state.responseId = readString(result, "jobId") ?? state.responseId;

        // generateVideo() without `stream` returns once the job is accepted and never calls a
        // terminal hook; only a status call that reaches `completed` carries a status.
        if (readString(result, "status") !== "completed" && states.get(ctx as object) === state) {
          await finish(ctx, { duration: 0 }, undefined, undefined, false);
        }
      } catch (error) {
        config.onError?.(error);
      }

      return undefined;
    });
  };

  return {
    name: "telemetry-dev",
    onStart(ctx) {
      try {
        const state = newState(ctx);

        if (!state) return;

        // Video spans start at the terminal hook: a status call fires onStart only after the
        // provider answered, so the span is backdated by the reported duration instead.
        if (ctx.activity !== "video") state.span = startSpan(ctx, state);

        states.set(ctx as object, state);

        if (ctx.activity !== "evaluate") observeResult(ctx, state);
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
