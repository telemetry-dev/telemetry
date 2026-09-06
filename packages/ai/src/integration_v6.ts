import {
  type Attributes,
  type Context,
  context as otelContext,
  ROOT_CONTEXT,
  type Span,
  SpanKind,
  SpanStatusCode,
  trace,
} from "@opentelemetry/api";
import { jsonAttr, omitUndefined, withSessionParent } from "@telemetry-dev/otel";

import { resolveConfig, type ResolvedConfig, type TelemetryDevOptions } from "./config.ts";
import { createEmitter, type Emitter, type EmitterOverrides } from "./otel.ts";
import {
  providerLabel,
  readId,
  SEVERITY_ERROR,
  SEVERITY_INFO,
  SEVERITY_WARN,
  type TelemetryDevEvent,
  type JsonValue,
} from "./shared.ts";

// Structural shapes of the ai@6 telemetry events the hooks read. Declared locally because the
// installed `ai` devDep is v7, which no longer exports the v6 `TelemetryIntegration` event types;
// `telemetryDev` narrows the loosely-typed public hook parameters to these.
interface V6ModelInfo {
  provider: string;
  modelId: string;
}

interface V6StartEvent {
  model: V6ModelInfo;
  system?: JsonValue;
  prompt?: JsonValue;
  messages?: JsonValue;
  metadata?: Record<string, JsonValue>;
  temperature?: number;
  topP?: number;
  topK?: number;
  maxOutputTokens?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  stopSequences?: string[];
  seed?: number;
  toolChoice?: JsonValue;
  functionId?: string;
}

interface V6StepStartEvent {
  stepNumber: number;
  messages?: JsonValue;
}

interface V6StepFinishEvent {
  stepNumber: number;
  model: V6ModelInfo;
  text?: string;
  finishReason?: string;
  response?: { id?: string; modelId?: string };
  usage: {
    inputTokens?: number;
    outputTokens?: number;
    cachedInputTokens?: number;
    reasoningTokens?: number;
    inputTokenDetails?: { cacheReadTokens?: number; cacheWriteTokens?: number };
    outputTokenDetails?: { reasoningTokens?: number };
  };
  warnings?: Array<Record<string, JsonValue>>;
}

interface V6ToolCallStartEvent {
  toolCall: { toolCallId: string };
}

interface V6ToolCallFinishEvent {
  toolCall: { toolCallId: string; toolName: string; input?: JsonValue };
  durationMs: number;
  stepNumber?: number;
  success: boolean;
  output?: JsonValue;
  error?: JsonValue;
}

interface V6FinishEvent {
  finishReason?: string;
  text?: string;
}

/**
 * The integration object returned by the `./v6` entry's `telemetryDev()`. Structurally satisfies
 * the ai@6 `TelemetryIntegration` interface without importing it (the type was removed in ai@7).
 */
export interface TelemetryDevIntegration {
  onStart?: (event: TelemetryDevEvent) => void | PromiseLike<void>;
  onStepStart?: (event: TelemetryDevEvent) => void | PromiseLike<void>;
  onStepFinish?: (event: TelemetryDevEvent) => void | PromiseLike<void>;
  onToolCallStart?: (event: TelemetryDevEvent) => void | PromiseLike<void>;
  onToolCallFinish?: (event: TelemetryDevEvent) => void | PromiseLike<void>;
  onFinish?: (event: TelemetryDevEvent) => void | PromiseLike<void>;
}

/**
 * Build a fresh Vercel AI SDK telemetry integration for ai@6 that streams generateText /
 * streamText / Agent runs to telemetry.dev as OpenTelemetry GenAI (`gen_ai.*`) spans + metrics.
 * Each call returns an independent object holding per-generation state, so passing a new
 * `telemetryDev()` per call (via `experimental_telemetry.integrations`) is concurrency-safe. A
 * single instance registered globally is not safe across overlapping concurrent generations — its
 * mutable state would interleave. On ai@7, use the package root entry instead.
 */
export function telemetryDev(
  options?: TelemetryDevOptions,
  overrides?: EmitterOverrides,
): TelemetryDevIntegration {
  const config = resolveConfig(options);

  // Gate on our own config: with no key, every hook is a no-op (the SDK fires hooks regardless of
  // experimental_telemetry.isEnabled and swallows any thrown errors).
  if (!config.apiKey) {
    return {};
  }

  const emitter = createEmitter(config, overrides);
  const v6 = createV6Hooks(config, emitter);

  // The public parameters are loose supertypes; the ai@6 dispatcher guarantees these shapes.
  return {
    onStart: (event) => v6.onStart(event as V6StartEvent),
    onStepStart: (event) => v6.onStepStart(event as V6StepStartEvent),
    onStepFinish: (event) => v6.onStepFinish(event as V6StepFinishEvent),
    onToolCallStart: (event) => v6.onToolCallStart(event as V6ToolCallStartEvent),
    onToolCallFinish: (event) => v6.onToolCallFinish(event as V6ToolCallFinishEvent),
    onFinish: (event) => v6.onFinish(event as V6FinishEvent),
  };
}

// The ai@6 hook set, exactly as shipped for `ai >=6.0.111 <7`: closure state per integration
// instance, one trace per generation, flushed at onFinish.
function createV6Hooks(config: ResolvedConfig, emitter: Emitter) {
  const onError = config.onError;

  // Per-generation state. Reset on every onStart.
  let rootSpan: Span | undefined;
  let rootCtx: Context = ROOT_CONTEXT;
  let rootStartedAt = new Date();
  let provider: string | null = null;
  let model: string | null = null;
  let responseModel: string | null = null;
  let rootInput: Record<string, JsonValue> | undefined;
  let rootOutput: string | undefined;
  let rootFinishReason: string | null = null;
  let userId: string | null = null;
  let sessionId: string | null = null;
  let restMetadata: Record<string, JsonValue> | undefined;
  let samplingAttributes: Attributes = {};
  const steps = new Map<
    number,
    { span?: Span; ctx: Context; startedAt: Date; messages?: JsonValue }
  >();
  const toolStarts = new Map<string, Date>();
  let childSpans: Span[] = [];
  let hasToolSpan = false;
  // Per-step metric inputs, collected at onStepFinish and emitted at onFinish. Model/provider are
  // captured per step so fallback or mixed-model runs attribute tokens and duration to the model
  // that actually ran the step, not the root request's model.
  const stepMetrics: Array<{
    durationSec: number;
    inputTokens: number | null;
    outputTokens: number | null;
    provider: string;
    requestModel: string;
    responseModel: string | null;
  }> = [];
  const toolMetrics: Array<{ durationSec: number }> = [];

  const conversationAttributes = (): Attributes =>
    sessionId ? { "gen_ai.conversation.id": sessionId } : {};

  const integration = {
    onStart(e: V6StartEvent) {
      try {
        rootStartedAt = new Date();
        steps.clear();
        toolStarts.clear();
        childSpans = [];
        hasToolSpan = false;
        stepMetrics.length = 0;
        toolMetrics.length = 0;
        rootOutput = undefined;
        rootFinishReason = null;
        responseModel = null;
        provider = providerLabel(e.model.provider);
        model = e.model.modelId;

        const input: Record<string, JsonValue> = {};
        if (e.system !== undefined) {
          input.system = e.system;
        }
        if (e.prompt !== undefined) {
          input.prompt = e.prompt;
        }
        if (e.messages !== undefined) {
          input.messages = e.messages;
        }
        rootInput = Object.keys(input).length > 0 ? input : undefined;

        const metadata = e.metadata;
        userId = readId(metadata?.userId);
        sessionId = readId(metadata?.sessionId);
        if (metadata) {
          const rest: Record<string, JsonValue> = {};
          for (const [key, value] of Object.entries(metadata)) {
            if (key !== "userId" && key !== "sessionId") {
              rest[key] = value;
            }
          }
          restMetadata = Object.keys(rest).length > 0 ? rest : undefined;
        } else {
          restMetadata = undefined;
        }

        samplingAttributes = omitUndefined({
          "gen_ai.request.temperature": e.temperature,
          "gen_ai.request.top_p": e.topP,
          "gen_ai.request.top_k": e.topK,
          "gen_ai.request.max_tokens": e.maxOutputTokens,
          "gen_ai.request.presence_penalty": e.presencePenalty,
          "gen_ai.request.frequency_penalty": e.frequencyPenalty,
          "gen_ai.request.stop_sequences": e.stopSequences,
          "gen_ai.request.seed": e.seed,
          "gen_ai.request.choice.tool_choice": jsonAttr(e.toolChoice),
        });

        // Session-parented: see withSessionParent.
        rootSpan = emitter.tracer.startSpan(
          e.functionId || "chat",
          { startTime: rootStartedAt, kind: SpanKind.INTERNAL },
          withSessionParent(otelContext.active(), sessionId ?? undefined, config.apiKey),
        );
        rootCtx = trace.setSpan(ROOT_CONTEXT, rootSpan);
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    onStepStart(e: V6StepStartEvent) {
      try {
        // Open the step (model `chat`) span now so tool calls that finish within this step parent
        // to it. Attributes/finish state land at onStepFinish; the span ends there too.
        const startedAt = new Date();
        const span = emitter.tracer.startSpan(
          "chat",
          { startTime: startedAt, kind: SpanKind.CLIENT, attributes: conversationAttributes() },
          rootCtx,
        );
        steps.set(e.stepNumber, {
          span,
          ctx: trace.setSpan(rootCtx, span),
          startedAt,
          messages: e.messages,
        });
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    onStepFinish(e: V6StepFinishEvent) {
      try {
        const endedAt = new Date();
        let step = steps.get(e.stepNumber);
        if (!step) {
          // No matching onStepStart: open the span anchored to the trace start so startTime never
          // exceeds endTime.
          const span = emitter.tracer.startSpan(
            "chat",
            {
              startTime: rootStartedAt,
              kind: SpanKind.CLIENT,
              attributes: conversationAttributes(),
            },
            rootCtx,
          );
          step = { span, ctx: trace.setSpan(rootCtx, span), startedAt: rootStartedAt };
          steps.set(e.stepNumber, step);
        }
        const span = step.span!;
        const startedAt =
          step.startedAt.getTime() <= endedAt.getTime() ? step.startedAt : rootStartedAt;
        const usage = e.usage;
        const inputTokens = usage.inputTokens ?? null;
        const outputTokens = usage.outputTokens ?? null;
        const cacheReadTokens =
          usage.inputTokenDetails?.cacheReadTokens ?? usage.cachedInputTokens ?? null;
        const cacheCreationTokens = usage.inputTokenDetails?.cacheWriteTokens ?? null;
        const reasoningTokens =
          usage.outputTokenDetails?.reasoningTokens ?? usage.reasoningTokens ?? null;
        const stepProvider = providerLabel(e.model.provider);
        responseModel = e.response?.modelId ?? responseModel;

        span.setAttributes(
          omitUndefined({
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": stepProvider,
            "gen_ai.request.model": e.model.modelId,
            "gen_ai.response.model": e.response?.modelId ?? undefined,
            "gen_ai.response.id": e.response?.id ?? undefined,
            "gen_ai.usage.input_tokens": inputTokens ?? undefined,
            "gen_ai.usage.output_tokens": outputTokens ?? undefined,
            "gen_ai.usage.cache_read.input_tokens": cacheReadTokens ?? undefined,
            "gen_ai.usage.cache_creation.input_tokens": cacheCreationTokens ?? undefined,
            "gen_ai.usage.reasoning.output_tokens": reasoningTokens ?? undefined,
            "gen_ai.response.finish_reasons": e.finishReason ? [e.finishReason] : undefined,
            "gen_ai.output.type": "text",
            "gen_ai.input.messages": jsonAttr(step.messages),
            "gen_ai.output.messages": jsonAttr(e.text),
          }),
        );

        if (e.finishReason === "error") {
          span.setStatus({ code: SpanStatusCode.ERROR });
        }

        const stepWarnings = e.warnings ?? [];
        for (const warning of stepWarnings) {
          const detail =
            warning.message?.constructor === String
              ? String(warning.message)
              : warning.type?.constructor === String
                ? String(warning.type)
                : "warning";
          span.addEvent(
            "model.warning",
            omitUndefined({
              "log.severity_number": SEVERITY_WARN,
              "log.message": `Model warning: ${detail}`,
              "warning.type":
                warning.type?.constructor === String ? String(warning.type) : undefined,
            }),
          );
        }

        span.end(endedAt);
        childSpans.push(span);
        stepMetrics.push({
          durationSec: Math.max(endedAt.getTime() - startedAt.getTime(), 0) / 1000,
          inputTokens,
          outputTokens,
          provider: stepProvider,
          requestModel: e.model.modelId,
          responseModel: e.response?.modelId ?? null,
        });
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    onToolCallStart(e: V6ToolCallStartEvent) {
      try {
        toolStarts.set(e.toolCall.toolCallId, new Date());
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    onToolCallFinish(e: V6ToolCallFinishEvent) {
      try {
        const endedAt = new Date();
        const started =
          toolStarts.get(e.toolCall.toolCallId) ?? new Date(endedAt.getTime() - e.durationMs);
        const startedAt = started.getTime() <= endedAt.getTime() ? started : rootStartedAt;
        const parentCtx =
          (e.stepNumber != null ? steps.get(e.stepNumber)?.ctx : undefined) ?? rootCtx;

        const span = emitter.tracer.startSpan(
          "execute_tool",
          {
            startTime: startedAt,
            kind: SpanKind.INTERNAL,
            attributes: omitUndefined({
              "gen_ai.operation.name": "execute_tool",
              "gen_ai.tool.name": e.toolCall.toolName,
              "gen_ai.tool.call.id": e.toolCall.toolCallId,
              ...conversationAttributes(),
              "gen_ai.tool.call.arguments": jsonAttr(e.toolCall.input),
              "gen_ai.tool.call.result": e.success ? jsonAttr(e.output) : undefined,
            }),
          },
          parentCtx,
        );

        if (!e.success) {
          const message = e.error instanceof Error ? e.error.message : String(e.error);
          const errorType = e.error instanceof Error ? e.error.name : "tool_error";
          span.setStatus({ code: SpanStatusCode.ERROR });
          span.setAttribute("error.type", errorType);
          span.addEvent("exception", {
            "exception.type": errorType,
            "exception.message": message,
            "log.severity_number": SEVERITY_ERROR,
          });
        }

        span.end(endedAt);
        childSpans.push(span);
        hasToolSpan = true;
        toolMetrics.push({
          durationSec: Math.max(endedAt.getTime() - startedAt.getTime(), 0) / 1000,
        });
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    async onFinish(e: V6FinishEvent) {
      try {
        if (!rootSpan) return;
        const rootEndedAt = new Date();
        rootFinishReason = e.finishReason ?? null;
        rootOutput = e.text;
        const operation = hasToolSpan ? "invoke_agent" : "chat";
        const finishReason = rootFinishReason ?? "unknown";
        const inputPresent = stepMetrics.some((s) => s.inputTokens !== null);
        const outputPresent = stepMetrics.some((s) => s.outputTokens !== null);
        const inputTokens = stepMetrics.reduce((sum, s) => sum + (s.inputTokens ?? 0), 0);
        const outputTokens = stepMetrics.reduce((sum, s) => sum + (s.outputTokens ?? 0), 0);
        const tokenParts: string[] = [];
        if (inputPresent) tokenParts.push(`${inputTokens} in`);
        if (outputPresent) tokenParts.push(`${outputTokens} out`);
        const tokenText = tokenParts.length > 0 ? `: ${tokenParts.join(" / ")} tokens` : "";
        const hasError = e.finishReason === "error";

        rootSpan.setAttributes(
          omitUndefined({
            "gen_ai.operation.name": operation,
            "gen_ai.provider.name": provider ?? undefined,
            "gen_ai.request.model": model ?? undefined,
            "gen_ai.response.model": responseModel ?? undefined,
            ...conversationAttributes(),
            "gen_ai.input.messages": jsonAttr(rootInput),
            "gen_ai.output.messages": jsonAttr(rootOutput),
            "gen_ai.response.finish_reasons": rootFinishReason ? [rootFinishReason] : undefined,
            "user.id": userId ?? undefined,
            ...samplingAttributes,
          }),
        );
        if (restMetadata) {
          for (const [key, value] of Object.entries(restMetadata)) {
            const attr = value?.constructor === String ? String(value) : jsonAttr(value);
            if (attr !== undefined) {
              rootSpan.setAttribute(`td.metadata.${key}`, attr);
            }
          }
        }

        rootSpan.addEvent(
          "generation.summary",
          omitUndefined({
            "log.severity_number": hasError ? SEVERITY_ERROR : SEVERITY_INFO,
            "log.message": hasError
              ? `Generation failed (${finishReason})`
              : `Generation completed (${finishReason})${tokenText}`,
            "gen_ai.usage.input_tokens": inputPresent ? inputTokens : undefined,
            "gen_ai.usage.output_tokens": outputPresent ? outputTokens : undefined,
          }),
        );

        if (hasError) {
          rootSpan.setStatus({ code: SpanStatusCode.ERROR });
          rootSpan.setAttribute("error.type", finishReason);
          rootSpan.addEvent("exception", {
            "exception.type": finishReason,
            "exception.message": `Generation failed (${finishReason})`,
            "log.severity_number": SEVERITY_ERROR,
          });
        }

        rootSpan.end(rootEndedAt);

        const metricBase: Attributes = omitUndefined({
          "gen_ai.provider.name": provider ?? undefined,
          "gen_ai.request.model": model ?? undefined,
          "gen_ai.response.model": responseModel ?? undefined,
        });
        for (const s of stepMetrics) {
          const stepAttrs: Attributes = omitUndefined({
            "gen_ai.provider.name": s.provider,
            "gen_ai.request.model": s.requestModel,
            "gen_ai.response.model": s.responseModel ?? undefined,
            "gen_ai.operation.name": "chat",
          });
          emitter.recordDuration(s.durationSec, stepAttrs);
          if (s.inputTokens !== null) emitter.recordTokens("input", s.inputTokens, stepAttrs);
          if (s.outputTokens !== null) emitter.recordTokens("output", s.outputTokens, stepAttrs);
        }
        for (const t of toolMetrics) {
          emitter.recordDuration(t.durationSec, {
            ...metricBase,
            "gen_ai.operation.name": "execute_tool",
          });
        }

        await emitter.flush([rootSpan, ...childSpans]);
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },
  };

  return integration;
}
