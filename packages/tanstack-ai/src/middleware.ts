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
import type {
  AbortInfo,
  AfterToolCallInfo,
  ChatMiddleware,
  ChatMiddlewareConfig,
  ChatMiddlewareContext,
  ErrorInfo,
  FinishInfo,
  TokenUsage,
  ToolCallHookContext,
  ToolPhaseCompleteInfo,
  UsageInfo,
} from "@tanstack/ai";
import {
  createGenerationEmitter,
  type GenerationEmitterOverrides,
  jsonAttr,
  omitUndefined,
  withSessionParent,
} from "@telemetry-dev/otel";

import { resolveConfig, type TelemetryDevOptions } from "./config.ts";

type JsonValue =
  | string
  | number
  | bigint
  | boolean
  | null
  | undefined
  | JsonValue[]
  | { [key: string]: JsonValue };

const isString = <T>(value: T): value is T & string => typeof value === "string";
const isNumber = <T>(value: T): value is T & number => typeof value === "number";
const isBigInt = <T>(value: T): value is T & bigint => typeof value === "bigint";

const isObject = <T>(value: T): value is T & { [key: string]: JsonValue } =>
  value !== null && typeof value === "object";

function readId<T>(value: T): string | null {
  if (isString(value)) {
    return value.length > 0 ? value : null;
  }

  if (isNumber(value) || isBigInt(value)) {
    return value.toString();
  }

  return null;
}

function firstNumber<T>(...candidates: T[]): number | undefined {
  for (const candidate of candidates) {
    if (isNumber(candidate) && Number.isFinite(candidate)) {
      return candidate;
    }
  }

  return undefined;
}

function errorTypeName<T>(err: T): string {
  if (err instanceof Error) return err.name || "Error";

  if (err && isObject(err) && "name" in err) {
    const n = (err as { name?: unknown }).name;

    if (isString(n) && n.length > 0) return n;
  }

  return "Error";
}

function errorMessage<T>(err: T): string {
  if (err instanceof Error) return err.message;

  if (isString(err)) return err;

  if (err && isObject(err) && "message" in err) {
    const m = (err as { message?: unknown }).message;

    if (isString(m)) return m;
  }

  return String(err);
}

const SEVERITY_INFO = 9;
const SEVERITY_ERROR = 17;

// Provider-native spellings of the output-token cap, mirrored from TanStack AI's
// `utilities/sampling-keys.ts` (not exported from the package root).
const MAX_TOKENS_KEYS = [
  "max_output_tokens",
  "max_tokens",
  "max_completion_tokens",
  "maxOutputTokens",
  "maxCompletionTokens",
  "maxTokens",
] as const;

// Sampling options live in opaque provider-native `modelOptions`; pick the first numeric value
// among the known spellings (including Ollama's nested `options`) for the gen_ai.request.* attrs.
function samplingAttributes(modelOptions: Record<string, JsonValue> | undefined): Attributes {
  const sampling = modelOptions ?? {};

  const nested =
    sampling["options"] && isObject(sampling["options"])
      ? (sampling["options"] as Record<string, JsonValue>)
      : undefined;

  return omitUndefined({
    "gen_ai.request.temperature": firstNumber(sampling["temperature"], nested?.["temperature"]),
    "gen_ai.request.top_p": firstNumber(sampling["top_p"], sampling["topP"], nested?.["top_p"]),
    "gen_ai.request.max_tokens": firstNumber(
      ...MAX_TOKENS_KEYS.map((key) => sampling[key]),
      nested?.["num_predict"],
    ),
  });
}

interface IterationState {
  span: Span;
  otelCtx: Context;
  startedAt: Date;
  structured: boolean;
  usage: TokenUsage | null;
  finishReason: string | null;
  responseModel: string | null;
  outputText: string | null;
}

interface RunState {
  rootSpan: Span;
  rootCtx: Context;
  provider: string;
  requestModel: string;
  responseModel: string | null;
  userId: string | null;
  sessionId: string;
  restMetadata: Record<string, JsonValue> | undefined;
  rootInput: string | undefined;
  rootSampling: Attributes;
  rootCaptured: boolean;
  iteration: IterationState | null;
  // Raw JSON from the legacy `chat({ outputSchema })` finalization iteration; preferred over
  // onFinish's `info.content` for the root output (finalization never updates accumulatedContent).
  structuredOutput: string | null;
  openTools: Map<string, { span: Span; startedAt: Date }>;
  childSpans: Span[];
  hasToolSpan: boolean;
  iterationMetrics: Array<{
    durationSec: number;
    inputTokens: number | null;
    outputTokens: number | null;
  }>;
  toolMetrics: Array<{ durationSec: number }>;
}

/**
 * Build a TanStack AI chat middleware that streams `chat()` runs to telemetry.dev as
 * OpenTelemetry GenAI (`gen_ai.*`) spans + metrics: one root span per `chat()` call (the roots
 * of one session share a trace, see withSessionParent), one CLIENT
 * span per agent-loop iteration, and one span per tool execution. Per-run state is keyed by the
 * middleware context in a WeakMap, so a single `telemetryDev()` instance is safe to share across
 * concurrent and overlapping `chat()` calls (e.g. registered once at module scope).
 */
export function telemetryDev(
  options?: TelemetryDevOptions,
  overrides?: GenerationEmitterOverrides,
): ChatMiddleware {
  const config = resolveConfig(options);

  // Gate on our own config: with no key, the middleware is inert and never throws.
  if (!config.apiKey) {
    return { name: "telemetry-dev" };
  }

  const onError = config.onError;

  const emitter = createGenerationEmitter(
    {
      ...config,
      sdkName: "@telemetry-dev/tanstack-ai",
      onError: onError
        ? (error) => onError(error instanceof Error ? error : new Error(error))
        : undefined,
    },
    overrides,
  );

  const states = new WeakMap<ChatMiddlewareContext, RunState>();

  const closeIteration = (state: RunState): void => {
    const iteration = state.iteration;

    if (!iteration) return;
    const endedAt = new Date();
    const usage = iteration.usage;
    iteration.span.setAttributes(
      omitUndefined({
        "gen_ai.response.model": iteration.responseModel ?? undefined,
        "gen_ai.response.finish_reasons": iteration.finishReason
          ? [iteration.finishReason]
          : undefined,
        "gen_ai.usage.input_tokens": usage?.promptTokens,
        "gen_ai.usage.output_tokens": usage?.completionTokens,
        "gen_ai.usage.cache_read.input_tokens": usage?.promptTokensDetails?.cachedTokens,
        "gen_ai.usage.cache_creation.input_tokens": usage?.promptTokensDetails?.cacheWriteTokens,
        "gen_ai.usage.reasoning.output_tokens": usage?.completionTokensDetails?.reasoningTokens,
        "gen_ai.usage.cost": usage?.cost,
        "gen_ai.output.type": iteration.structured ? "json" : "text",
        "gen_ai.output.messages": jsonAttr(iteration.outputText ?? undefined),
      }),
    );
    iteration.span.end(endedAt);
    state.childSpans.push(iteration.span);
    state.iterationMetrics.push({
      durationSec: Math.max(endedAt.getTime() - iteration.startedAt.getTime(), 0) / 1000,
      inputTokens: usage?.promptTokens ?? null,
      outputTokens: usage?.completionTokens ?? null,
    });

    if (iteration.structured && iteration.outputText !== null) {
      state.structuredOutput = iteration.outputText;
    }

    state.iteration = null;
  };

  // Root attributes shared by all three terminal hooks. Rolled-up usage is deliberately NOT set
  // as root gen_ai.usage.* attributes: the ingest sums usage across every span of a trace, so a
  // root rollup would double-count tokens/cost. The rollup lands on the generation.summary event.
  const setRootBaseAttributes = (state: RunState): void => {
    state.rootSpan.setAttributes(
      omitUndefined({
        "gen_ai.operation.name": state.hasToolSpan ? "invoke_agent" : "chat",
        "gen_ai.provider.name": state.provider,
        "gen_ai.request.model": state.requestModel,
        "gen_ai.response.model": state.responseModel ?? undefined,
        "gen_ai.conversation.id": state.sessionId,
        "gen_ai.input.messages": state.rootInput,
        "user.id": state.userId ?? undefined,
        ...state.rootSampling,
      }),
    );

    if (state.restMetadata) {
      for (const [key, value] of Object.entries(state.restMetadata)) {
        const attr = isString(value) ? value : jsonAttr(value);

        if (attr !== undefined) {
          state.rootSpan.setAttribute(`td.metadata.${key}`, attr);
        }
      }
    }
  };

  const addSummaryEvent = (state: RunState, hasError: boolean, message: string): void => {
    const inputPresent = state.iterationMetrics.some((m) => m.inputTokens !== null);
    const outputPresent = state.iterationMetrics.some((m) => m.outputTokens !== null);
    const inputTokens = state.iterationMetrics.reduce((sum, m) => sum + (m.inputTokens ?? 0), 0);
    const outputTokens = state.iterationMetrics.reduce((sum, m) => sum + (m.outputTokens ?? 0), 0);
    state.rootSpan.addEvent(
      "generation.summary",
      omitUndefined({
        "log.severity_number": hasError ? SEVERITY_ERROR : SEVERITY_INFO,
        "log.message": message,
        "gen_ai.usage.input_tokens": inputPresent ? inputTokens : undefined,
        "gen_ai.usage.output_tokens": outputPresent ? outputTokens : undefined,
      }),
    );
  };

  const recordRunMetrics = (state: RunState): void => {
    const metricBase: Attributes = omitUndefined({
      "gen_ai.provider.name": state.provider,
      "gen_ai.request.model": state.requestModel,
      "gen_ai.response.model": state.responseModel ?? undefined,
    });

    for (const m of state.iterationMetrics) {
      const attrs: Attributes = { ...metricBase, "gen_ai.operation.name": "chat" };
      emitter.recordDuration(m.durationSec, attrs);

      if (m.inputTokens !== null) emitter.recordTokens("input", m.inputTokens, attrs);

      if (m.outputTokens !== null) emitter.recordTokens("output", m.outputTokens, attrs);
    }

    for (const t of state.toolMetrics) {
      emitter.recordDuration(t.durationSec, {
        ...metricBase,
        "gen_ai.operation.name": "execute_tool",
      });
    }
  };

  // Mark every still-open iteration/tool span failed and end it, so the terminal error/abort
  // hooks never leave dangling spans out of the flushed batch.
  const failOpenSpans = (state: RunState, errType: string, message: string): void => {
    for (const [, entry] of state.openTools) {
      entry.span.setStatus({ code: SpanStatusCode.ERROR, message });
      entry.span.setAttribute("error.type", errType);
      entry.span.end();
      state.childSpans.push(entry.span);
    }

    state.openTools.clear();

    if (state.iteration) {
      state.iteration.span.setStatus({ code: SpanStatusCode.ERROR, message });
      state.iteration.span.setAttribute("error.type", errType);
      closeIteration(state);
    }
  };

  return {
    name: "telemetry-dev",

    onStart(ctx) {
      try {
        const rawMetadata = ctx.options?.["metadata"];

        const metadata =
          rawMetadata && isObject(rawMetadata)
            ? (rawMetadata as Record<string, JsonValue>)
            : undefined;

        const userId = readId(metadata?.["userId"]);
        const sessionId = readId(metadata?.["sessionId"]) ?? ctx.threadId;
        let restMetadata: Record<string, JsonValue> | undefined;

        if (metadata) {
          const rest: Record<string, JsonValue> = {};

          for (const [key, value] of Object.entries(metadata)) {
            if (key !== "userId" && key !== "sessionId") {
              rest[key] = value;
            }
          }

          restMetadata = Object.keys(rest).length > 0 ? rest : undefined;
        }

        // Session-parented: see withSessionParent.
        const rootSpan = emitter.tracer.startSpan(
          "chat",
          { startTime: new Date(), kind: SpanKind.INTERNAL },
          withSessionParent(otelContext.active(), sessionId ?? undefined, config.apiKey),
        );

        states.set(ctx, {
          rootSpan,
          rootCtx: trace.setSpan(ROOT_CONTEXT, rootSpan),
          provider: ctx.provider,
          requestModel: ctx.model,
          responseModel: null,
          userId,
          sessionId,
          restMetadata,
          rootInput: undefined,
          rootSampling: {},
          rootCaptured: false,
          iteration: null,
          structuredOutput: null,
          openTools: new Map(),
          childSpans: [],
          hasToolSpan: false,
          iterationMetrics: [],
          toolMetrics: [],
        });
      } catch (err) {
        onError?.(err);
      }
    },

    onConfig(ctx, chatConfig: ChatMiddlewareConfig) {
      // Both remaining phases are model-call boundaries: `beforeModel` per agent-loop iteration
      // and `structuredOutput` before the legacy finalization call the engine issues for
      // `chat({ outputSchema })` on adapters without native combined support. The latter needs
      // its own span — otherwise its onUsage would overwrite the last iteration's usage.
      if (ctx.phase !== "beforeModel" && ctx.phase !== "structuredOutput") return undefined;

      try {
        const state = states.get(ctx);

        if (!state) return undefined;

        // The previous iteration's span stays open through tool execution and onUsage so tool
        // spans nest under it and usage lands on it. Close it just before the next model call.
        closeIteration(state);

        const inputMessages: Array<{ role: string; content: unknown }> = [];

        for (const prompt of chatConfig.systemPrompts) {
          inputMessages.push({
            role: "system",
            content: isString(prompt) ? prompt : prompt.content,
          });
        }

        for (const message of chatConfig.messages) {
          inputMessages.push({ role: message.role, content: message.content });
        }

        const inputJson = jsonAttr(inputMessages);

        const sampling = samplingAttributes(
          (chatConfig.modelOptions ?? ctx.modelOptions) as Record<string, JsonValue> | undefined,
        );

        if (!state.rootCaptured) {
          state.rootCaptured = true;
          state.rootInput = inputJson;
          state.rootSampling = sampling;
        }

        const startedAt = new Date();

        const span = emitter.tracer.startSpan(
          "chat",
          {
            startTime: startedAt,
            kind: SpanKind.CLIENT,
            attributes: omitUndefined({
              "gen_ai.operation.name": "chat",
              "gen_ai.provider.name": ctx.provider,
              "gen_ai.request.model": ctx.model,
              "gen_ai.conversation.id": state.sessionId,
              "gen_ai.input.messages": inputJson,
              ...sampling,
            }),
          },
          state.rootCtx,
        );

        state.iteration = {
          span,
          otelCtx: trace.setSpan(state.rootCtx, span),
          startedAt,
          usage: null,
          finishReason: null,
          responseModel: null,
          outputText: null,
          structured: ctx.phase === "structuredOutput",
        };
      } catch (err) {
        onError?.(err);
      }

      return undefined;
    },

    onChunk(ctx, chunk) {
      if (chunk.type !== "RUN_FINISHED" && chunk.type !== "CUSTOM") return undefined;

      try {
        const state = states.get(ctx);
        const iteration = state?.iteration;

        if (!state || !iteration) return undefined;

        if (chunk.type === "CUSTOM") {
          // The finalization stream reports its JSON via this event; `ctx.accumulatedContent`
          // still holds the agent loop's text, so this is the structured span's only output.
          if (iteration.structured && chunk.name === "structured-output.complete") {
            const raw = (chunk.value as { raw?: unknown } | null | undefined)?.raw;

            if (isString(raw)) iteration.outputText = raw;
          }

          return undefined;
        }

        iteration.finishReason = chunk.finishReason ?? null;

        if (chunk.model) {
          iteration.responseModel = chunk.model;
          state.responseModel = chunk.model;
        }

        if (chunk.usage) iteration.usage = chunk.usage;

        if (!iteration.structured) {
          iteration.outputText = ctx.accumulatedContent.length > 0 ? ctx.accumulatedContent : null;
        }
      } catch (err) {
        onError?.(err);
      }

      return undefined;
    },

    onUsage(ctx, usage: UsageInfo) {
      try {
        const state = states.get(ctx);

        if (state?.iteration) {
          state.iteration.usage = usage;
        }
      } catch (err) {
        onError?.(err);
      }
    },

    onBeforeToolCall(ctx, hookCtx: ToolCallHookContext) {
      try {
        const state = states.get(ctx);

        if (!state) return undefined;
        const startedAt = new Date();

        const span = emitter.tracer.startSpan(
          "execute_tool",
          {
            startTime: startedAt,
            kind: SpanKind.INTERNAL,
            attributes: omitUndefined({
              "gen_ai.operation.name": "execute_tool",
              "gen_ai.tool.name": hookCtx.toolName,
              "gen_ai.tool.call.id": hookCtx.toolCallId,
              "gen_ai.conversation.id": state.sessionId,
              "gen_ai.tool.call.arguments": jsonAttr(hookCtx.args ?? null),
            }),
          },
          state.iteration?.otelCtx ?? state.rootCtx,
        );

        state.openTools.set(hookCtx.toolCallId, { span, startedAt });
        state.hasToolSpan = true;
      } catch (err) {
        onError?.(err);
      }

      return undefined;
    },

    onAfterToolCall(ctx, info: AfterToolCallInfo) {
      try {
        const state = states.get(ctx);
        const entry = state?.openTools.get(info.toolCallId);

        if (!state || !entry) return;
        state.openTools.delete(info.toolCallId);
        const { span } = entry;

        if (info.ok) {
          const result = jsonAttr(info.result ?? null);

          if (result !== undefined) {
            span.setAttribute("gen_ai.tool.call.result", result);
          }
        } else {
          const message = errorMessage(info.error);
          const errType = info.error instanceof Error ? info.error.name : "tool_error";
          span.setStatus({ code: SpanStatusCode.ERROR });
          span.setAttribute("error.type", errType);
          span.addEvent("exception", {
            "exception.type": errType,
            "exception.message": message,
            "log.severity_number": SEVERITY_ERROR,
          });
        }

        span.end();
        state.childSpans.push(span);
        state.toolMetrics.push({ durationSec: Math.max(info.duration, 0) / 1000 });
      } catch (err) {
        onError?.(err);
      }
    },

    async onToolPhaseComplete(ctx, info: ToolPhaseCompleteInfo) {
      // Ordinary tool phases finalize through the terminal hooks; only the wait path needs us.
      if (info.needsApproval.length === 0 && info.needsClientExecution.length === 0) return;

      try {
        const state = states.get(ctx);

        if (!state) return;
        states.delete(ctx);

        // The engine parks in toolPhase "wait" (user approval / client-side tool execution) and
        // never fires onFinish/onError/onAbort for this invocation — the resumed call is a new
        // chat() with a fresh ctx. Treat this as a terminal hook: close everything and flush,
        // otherwise the whole run's spans leak unexported.
        for (const [, entry] of state.openTools) {
          entry.span.end();
          state.childSpans.push(entry.span);
        }

        state.openTools.clear();
        const finishReason = state.iteration?.finishReason ?? "tool_calls";
        closeIteration(state);

        // The model requested tools even when none executed server-side (approval-gated and
        // client tools never reach onBeforeToolCall), so this run is an agent invocation.
        state.hasToolSpan = true;
        setRootBaseAttributes(state);
        state.rootSpan.setAttributes(
          omitUndefined({
            "gen_ai.output.messages": jsonAttr(
              ctx.accumulatedContent.length > 0 ? ctx.accumulatedContent : undefined,
            ),
            "gen_ai.response.finish_reasons": [finishReason],
          }),
        );

        const waitingTools = [...info.needsApproval, ...info.needsClientExecution]
          .map((t) => t.toolName)
          .join(", ");

        addSummaryEvent(state, false, `Generation paused awaiting tools (${waitingTools})`);

        state.rootSpan.end();
        recordRunMetrics(state);
        await emitter.flush([state.rootSpan, ...state.childSpans]);
      } catch (err) {
        onError?.(err);
      }
    },

    async onFinish(ctx, info: FinishInfo) {
      try {
        const state = states.get(ctx);

        if (!state) return;
        states.delete(ctx);

        // Close any tool spans that never received onAfterToolCall before the iteration span, so
        // the hierarchy ends depth-first and nothing dangles out of the flushed batch.
        for (const [, entry] of state.openTools) {
          entry.span.end();
          state.childSpans.push(entry.span);
        }

        state.openTools.clear();
        closeIteration(state);

        setRootBaseAttributes(state);
        state.rootSpan.setAttributes(
          omitUndefined({
            "gen_ai.output.messages": jsonAttr(state.structuredOutput ?? info.content),
            "gen_ai.response.finish_reasons": info.finishReason ? [info.finishReason] : undefined,
          }),
        );

        const finishReason = info.finishReason ?? "unknown";

        const inputTokens = state.iterationMetrics.reduce(
          (sum, m) => sum + (m.inputTokens ?? 0),
          0,
        );

        const outputTokens = state.iterationMetrics.reduce(
          (sum, m) => sum + (m.outputTokens ?? 0),
          0,
        );

        const tokenParts: string[] = [];

        if (state.iterationMetrics.some((m) => m.inputTokens !== null)) {
          tokenParts.push(`${inputTokens} in`);
        }

        if (state.iterationMetrics.some((m) => m.outputTokens !== null)) {
          tokenParts.push(`${outputTokens} out`);
        }

        const tokenText = tokenParts.length > 0 ? `: ${tokenParts.join(" / ")} tokens` : "";
        addSummaryEvent(state, false, `Generation completed (${finishReason})${tokenText}`);

        state.rootSpan.end();
        recordRunMetrics(state);
        await emitter.flush([state.rootSpan, ...state.childSpans]);
      } catch (err) {
        onError?.(err);
      }
    },

    async onError(ctx, info: ErrorInfo) {
      try {
        const state = states.get(ctx);

        if (!state) return;
        states.delete(ctx);

        const errType = errorTypeName(info.error);
        const message = errorMessage(info.error);
        failOpenSpans(state, errType, message);

        setRootBaseAttributes(state);
        state.rootSpan.setStatus({ code: SpanStatusCode.ERROR, message });
        state.rootSpan.setAttribute("error.type", errType);
        state.rootSpan.addEvent("exception", {
          "exception.type": errType,
          "exception.message": message,
          "log.severity_number": SEVERITY_ERROR,
        });
        addSummaryEvent(state, true, `Generation failed (${errType})`);

        state.rootSpan.end();
        recordRunMetrics(state);
        await emitter.flush([state.rootSpan, ...state.childSpans]);
      } catch (err) {
        onError?.(err);
      }
    },

    async onAbort(ctx, info: AbortInfo) {
      try {
        const state = states.get(ctx);

        if (!state) return;
        states.delete(ctx);

        const message = info.reason ?? "cancelled";
        failOpenSpans(state, "cancelled", message);

        setRootBaseAttributes(state);
        state.rootSpan.setStatus({ code: SpanStatusCode.ERROR, message });
        state.rootSpan.setAttribute("error.type", "cancelled");
        state.rootSpan.setAttribute("gen_ai.response.finish_reasons", ["cancelled"]);
        addSummaryEvent(state, true, `Generation cancelled (${message})`);

        state.rootSpan.end();
        recordRunMetrics(state);
        await emitter.flush([state.rootSpan, ...state.childSpans]);
      } catch (err) {
        onError?.(err);
      }
    },
  };
}
