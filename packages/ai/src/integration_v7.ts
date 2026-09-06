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

type TelemetryDevHook = (event: TelemetryDevEvent) => void | PromiseLike<void>;

/**
 * The integration object returned by the root entry's `telemetryDev()`. Structurally satisfies
 * the ai@7 `Telemetry` interface without importing it (the type does not exist in ai@6).
 * `onStepFinish` is deliberately not implemented: ai@7 fans step-end out to both `onStepEnd` and
 * the deprecated `onStepFinish`, so implementing both would double every step span.
 */
export interface TelemetryDevIntegration {
  onStart?: TelemetryDevHook;
  onStepStart?: TelemetryDevHook;
  onStepEnd?: TelemetryDevHook;
  onToolExecutionStart?: TelemetryDevHook;
  onToolExecutionEnd?: TelemetryDevHook;
  onObjectStepStart?: TelemetryDevHook;
  onObjectStepEnd?: TelemetryDevHook;
  onEmbedStart?: TelemetryDevHook;
  onEmbedEnd?: TelemetryDevHook;
  onRerankStart?: TelemetryDevHook;
  onRerankEnd?: TelemetryDevHook;
  // Context wrappers: run the provider model call / tool execute function inside this
  // integration's span context, so auto-instrumented provider requests parent to the step span
  // and nested AI SDK calls made inside tools parent to the tool span. Params are minimal
  // supertypes of ai@7's (contravariance keeps the object assignable to `Telemetry`).
  executeLanguageModelCall?: <T>(options: {
    callId: string;
    execute: () => PromiseLike<T>;
  }) => PromiseLike<T>;
  executeTool?: <T>(options: {
    callId: string;
    toolCallId: string;
    execute: () => PromiseLike<T>;
  }) => PromiseLike<T>;
  onEnd?: TelemetryDevHook;
  onAbort?: TelemetryDevHook;
  onError?: (cause: unknown) => void | PromiseLike<void>;
}

/**
 * Build the Vercel AI SDK telemetry integration for ai@7. Streams generateText / streamText /
 * Agent / generateObject / streamObject / embed / embedMany / rerank runs to telemetry.dev as
 * OpenTelemetry GenAI (`gen_ai.*`) spans + metrics. State is keyed by the SDK's per-call
 * `callId`, so a single instance — including one registered globally via `registerTelemetry` —
 * is safe across overlapping concurrent generations. On ai@6, use the `./v6` entry instead.
 */
export function telemetryDev(
  options?: TelemetryDevOptions,
  overrides?: EmitterOverrides,
): TelemetryDevIntegration {
  const config = resolveConfig(options);

  // Gate on our own config: with no key, every hook is a no-op (the SDK fires hooks regardless of
  // telemetry.isEnabled and swallows any thrown errors).
  if (!config.apiKey) {
    return {};
  }

  return createV7Hooks(config, createEmitter(config, overrides));
}

// Structural shapes of the ai@7 telemetry events the v7 hooks read, grounded in
// vercel/ai@7.0.13 (packages/ai/src/generate-text/*, generate-object/structured-output-events,
// embed/embed-events, rerank/rerank-events). Declared locally so the emitted types never import
// from `ai`.

interface V7Usage {
  inputTokens?: number;
  outputTokens?: number;
  inputTokenDetails?: { cacheReadTokens?: number; cacheWriteTokens?: number };
  outputTokenDetails?: { reasoningTokens?: number };
}

interface V7StartEvent {
  callId: string;
  operationId: string;
  provider: string;
  modelId: string;
  // Merged into every event by the ai@7 dispatcher from the call's `telemetry` options.
  functionId?: string;
  recordInputs?: boolean;
  recordOutputs?: boolean;
  // Text ops (StandardizedPrompt): the v6 `system`/`prompt` are folded into these.
  instructions?: JsonValue;
  // Object ops keep the split fields.
  system?: JsonValue;
  prompt?: JsonValue;
  messages?: JsonValue;
  // Already filtered by the SDK per `telemetry.includeRuntimeContext`; text ops only.
  runtimeContext?: Record<string, JsonValue>;
  temperature?: number;
  topP?: number;
  topK?: number;
  maxOutputTokens?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  stopSequences?: string[];
  seed?: number;
  toolChoice?: JsonValue;
  // Embed ops.
  value?: JsonValue;
  // Rerank ops.
  documents?: JsonValue;
  query?: string;
}

interface V7StepStartEvent {
  callId: string;
  provider: string;
  modelId: string;
  stepNumber: number;
  messages?: JsonValue;
  runtimeContext?: Record<string, JsonValue>;
}

interface V7Performance {
  stepTimeMs?: number;
  responseTimeMs?: number;
  timeToFirstOutputMs?: number;
  toolExecutionMs?: Record<string, number>;
}

// StepResult: the onStepEnd event IS the step result.
interface V7StepEndEvent {
  callId: string;
  stepNumber: number;
  model: { provider: string; modelId: string };
  text?: string;
  finishReason?: string;
  response?: { id?: string; modelId?: string };
  usage: V7Usage;
  performance?: V7Performance;
  warnings?: Array<Record<string, JsonValue>>;
  runtimeContext?: Record<string, JsonValue>;
}

interface V7ToolCall {
  toolCallId: string;
  toolName: string;
  input?: JsonValue;
}

interface V7ToolExecutionStartEvent {
  callId: string;
  toolCall: V7ToolCall;
}

interface V7ToolExecutionEndEvent {
  callId: string;
  toolCall: V7ToolCall;
  toolExecutionMs: number;
  toolOutput:
    | { type: "tool-result"; output?: JsonValue }
    | { type: "tool-error"; error?: JsonValue };
}

interface V7ObjectStepStartEvent {
  callId: string;
  provider?: string;
  modelId?: string;
  promptMessages?: JsonValue;
}

interface V7ObjectStepEndEvent {
  callId: string;
  finishReason?: string;
  response?: { id?: string; modelId?: string };
  usage: V7Usage;
  objectText?: string;
  msToFirstChunk?: number;
}

interface V7EmbedCallStartEvent {
  callId: string;
  embedCallId: string;
  provider?: string;
  modelId?: string;
}

interface V7EmbedCallEndEvent {
  callId: string;
  embedCallId: string;
  usage?: { tokens?: number };
}

// onEnd fires for every operation type; the fields populated depend on the stored op kind.
interface V7EndEvent {
  callId: string;
  // Text ops: usage is aggregated across steps.
  finishReason?: string;
  text?: string;
  usage?: V7Usage & { tokens?: number };
  // Object ops. `error` carries streamObject parse/schema-validation failures (generateObject
  // throws instead), possibly with a non-error finishReason like "stop".
  object?: JsonValue;
  error?: JsonValue;
  ranking?: JsonValue;
  response?: { modelId?: string };
  runtimeContext?: Record<string, JsonValue>;
}

// The ai@7 `callId` correlation key, when the event carries one (every v7 event does; no v6
// event has the field).
const callIdOf = (event: TelemetryDevEvent): string | undefined =>
  "callId" in event && event.callId?.constructor === String ? String(event.callId) : undefined;

type OpKind = "text" | "object" | "embed" | "rerank";

const opKindOf = (operationId: string): OpKind => {
  switch (operationId) {
    case "ai.generateObject":
    case "ai.streamObject":
      return "object";
    case "ai.embed":
    case "ai.embedMany":
      return "embed";
    case "ai.rerank":
      return "rerank";
    default:
      return "text";
  }
};

const ROOT_NAME_BY_KIND = {
  text: "chat",
  object: "chat",
  embed: "embeddings",
  rerank: "rerank",
} satisfies Record<OpKind, string>;

interface StepState {
  span: Span;
  ctx: Context;
  startedAt: Date;
  messages?: JsonValue;
  open: boolean;
}

interface StepMetric {
  operation: string;
  durationSec: number;
  inputTokens: number | null;
  outputTokens: number | null;
  provider: string;
  requestModel: string;
  responseModel: string | null;
}

interface CallState {
  opKind: OpKind;
  rootSpan: Span;
  rootCtx: Context;
  rootStartedAt: Date;
  provider: string;
  model: string;
  responseModel: string | null;
  userId: string | null;
  sessionId: string | null;
  restMetadata: Record<string, JsonValue> | undefined;
  recordInputs: boolean;
  recordOutputs: boolean;
  rootInput: JsonValue;
  samplingAttributes: Attributes;
  steps: Map<number, StepState>;
  currentStepNumber: number | null;
  toolStarts: Map<string, Date>;
  // Tool spans opened by the executeTool wrapper, keyed by toolCallId; consumed at
  // onToolExecutionEnd (or closed by closeOpenChildSpans on failure/abort).
  toolSpans: Map<string, { span: Span; startedAt: Date }>;
  childSpans: Span[];
  hasToolSpan: boolean;
  stepMetrics: StepMetric[];
  toolMetrics: Array<{ durationSec: number }>;
  objectStep: { span: Span; startedAt: Date } | undefined;
  embedSpans: Map<string, { span: Span; startedAt: Date }>;
  rerankSpan: { span: Span; startedAt: Date } | undefined;
}

/**
 * The ai@7 hook set: state is keyed by the per-call `callId`, so one instance safely observes
 * overlapping concurrent generations (each call gets its own trace, flushed independently at
 * onEnd/onError/onAbort). The emitted span/metric shape intentionally matches the v6 path so
 * ingest and the UI see a single format.
 */
export function createV7Hooks(config: ResolvedConfig, emitter: Emitter): TelemetryDevIntegration {
  const onError = config.onError;
  const calls = new Map<string, CallState>();

  const conversationAttributes = (state: CallState): Attributes =>
    state.sessionId ? { "gen_ai.conversation.id": state.sessionId } : {};

  const stateOf = (event: TelemetryDevEvent): CallState | undefined => {
    const callId = callIdOf(event);
    return callId === undefined ? undefined : calls.get(callId);
  };

  const readRuntimeContext = (runtimeContext: Record<string, JsonValue> | undefined) => {
    const userId = readId(runtimeContext?.userId);
    const sessionId = readId(runtimeContext?.sessionId);
    let restMetadata: Record<string, JsonValue> | undefined;
    if (runtimeContext) {
      const rest: Record<string, JsonValue> = {};
      for (const [key, value] of Object.entries(runtimeContext)) {
        if (key !== "userId" && key !== "sessionId") {
          rest[key] = value;
        }
      }
      restMetadata = Object.keys(rest).length > 0 ? rest : undefined;
    }
    return { userId, sessionId, restMetadata };
  };

  const applyRuntimeContext = (
    state: CallState,
    runtimeContext: Record<string, JsonValue> | undefined,
  ) => {
    if (runtimeContext === undefined) return;
    const next = readRuntimeContext(runtimeContext);
    state.userId = next.userId;
    state.restMetadata = next.restMetadata;
  };

  // Base identity attributes for the root span; shared by the onEnd/onError/onAbort exits so a
  // failed or aborted call is attributed (provider/model/user/session/metadata) like a completed
  // one.
  const rootBaseAttributes = (state: CallState): Attributes => {
    const attributes = omitUndefined({
      "gen_ai.provider.name": state.provider,
      "gen_ai.request.model": state.model,
      "gen_ai.response.model": state.responseModel ?? undefined,
      ...conversationAttributes(state),
      "gen_ai.input.messages": state.recordInputs ? jsonAttr(state.rootInput) : undefined,
      "user.id": state.userId ?? undefined,
      ...state.samplingAttributes,
    });
    if (state.restMetadata) {
      for (const [key, value] of Object.entries(state.restMetadata)) {
        const attr = value?.constructor === String ? String(value) : jsonAttr(value);
        if (attr !== undefined) {
          attributes[`td.metadata.${key}`] = attr;
        }
      }
    }
    return attributes;
  };

  const recordQueuedMetrics = (state: CallState) => {
    for (const s of state.stepMetrics) {
      const stepAttrs: Attributes = omitUndefined({
        "gen_ai.provider.name": s.provider,
        "gen_ai.request.model": s.requestModel,
        "gen_ai.response.model": s.responseModel ?? undefined,
        "gen_ai.operation.name": s.operation,
      });
      emitter.recordDuration(s.durationSec, stepAttrs);
      if (s.inputTokens !== null) emitter.recordTokens("input", s.inputTokens, stepAttrs);
      if (s.outputTokens !== null) emitter.recordTokens("output", s.outputTokens, stepAttrs);
    }
    const metricBase: Attributes = omitUndefined({
      "gen_ai.provider.name": state.provider,
      "gen_ai.request.model": state.model,
      "gen_ai.response.model": state.responseModel ?? undefined,
    });
    for (const t of state.toolMetrics) {
      emitter.recordDuration(t.durationSec, {
        ...metricBase,
        "gen_ai.operation.name": "execute_tool",
      });
    }
  };

  const markSpanFailed = (span: Span, errorType: string, message: string, endedAt: Date) => {
    span.setStatus({ code: SpanStatusCode.ERROR });
    span.setAttribute("error.type", errorType);
    span.addEvent("exception", {
      "exception.type": errorType,
      "exception.message": message,
      "log.severity_number": SEVERITY_ERROR,
    });
    span.end(endedAt);
  };

  // Ends every still-open child span of a call (steps, object step, embed/rerank model calls),
  // optionally marking them failed, and returns them for the final flush.
  const closeOpenChildSpans = (
    state: CallState,
    endedAt: Date,
    failure?: { errorType: string; message: string },
  ) => {
    const close = (span: Span) => {
      if (failure) {
        markSpanFailed(span, failure.errorType, failure.message, endedAt);
      } else {
        span.end(endedAt);
      }
      state.childSpans.push(span);
    };
    for (const step of state.steps.values()) {
      if (step.open) {
        close(step.span);
        step.open = false;
      }
    }
    if (state.objectStep) {
      close(state.objectStep.span);
      state.objectStep = undefined;
    }
    for (const embed of state.embedSpans.values()) {
      close(embed.span);
    }
    state.embedSpans.clear();
    if (state.rerankSpan) {
      close(state.rerankSpan.span);
      state.rerankSpan = undefined;
    }
    for (const tool of state.toolSpans.values()) {
      close(tool.span);
      state.hasToolSpan = true;
    }
    state.toolSpans.clear();
  };

  const finishCall = async (state: CallState, callId: string) => {
    recordQueuedMetrics(state);
    await emitter.flush([state.rootSpan, ...state.childSpans]);
    calls.delete(callId);
  };

  const integration: TelemetryDevIntegration = {
    onStart(event) {
      try {
        // Ignore events without the v7 correlation id (e.g. this instance mistakenly wired into
        // an ai@6 runtime, whose events never carry callId).
        if (callIdOf(event) === undefined) return;
        // The public parameter is a loose supertype; the ai@7 dispatcher guarantees this shape.
        const e = event as V7StartEvent;
        const opKind = opKindOf(e.operationId);
        const startedAt = new Date();

        let rootInput: JsonValue;
        if (opKind === "text") {
          const input: Record<string, JsonValue> = {};
          if (e.instructions !== undefined) input.instructions = e.instructions;
          if (e.messages !== undefined) input.messages = e.messages;
          rootInput = Object.keys(input).length > 0 ? input : undefined;
        } else if (opKind === "object") {
          const input: Record<string, JsonValue> = {};
          if (e.system !== undefined) input.system = e.system;
          if (e.prompt !== undefined) input.prompt = e.prompt;
          if (e.messages !== undefined) input.messages = e.messages;
          rootInput = Object.keys(input).length > 0 ? input : undefined;
        } else if (opKind === "embed") {
          rootInput = e.value;
        } else {
          rootInput = { query: e.query, documents: e.documents };
        }

        // User context arrives pre-filtered by `telemetry.includeRuntimeContext` (default: all
        // keys dropped), so an empty/absent runtimeContext means no user/session/metadata.
        const runtime = readRuntimeContext(e.runtimeContext);

        const samplingAttributes = omitUndefined({
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
        const rootSpan = emitter.tracer.startSpan(
          e.functionId || ROOT_NAME_BY_KIND[opKind],
          { startTime: startedAt, kind: SpanKind.INTERNAL },
          withSessionParent(otelContext.active(), runtime.sessionId ?? undefined, config.apiKey),
        );

        calls.set(e.callId, {
          opKind,
          rootSpan,
          rootCtx: trace.setSpan(ROOT_CONTEXT, rootSpan),
          rootStartedAt: startedAt,
          provider: providerLabel(e.provider),
          model: e.modelId,
          responseModel: null,
          userId: runtime.userId,
          sessionId: runtime.sessionId,
          restMetadata: runtime.restMetadata,
          recordInputs: e.recordInputs !== false,
          recordOutputs: e.recordOutputs !== false,
          rootInput,
          samplingAttributes,
          steps: new Map(),
          currentStepNumber: null,
          toolStarts: new Map(),
          toolSpans: new Map(),
          childSpans: [],
          hasToolSpan: false,
          stepMetrics: [],
          toolMetrics: [],
          objectStep: undefined,
          embedSpans: new Map(),
          rerankSpan: undefined,
        });
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    onStepStart(event) {
      try {
        const state = stateOf(event);
        if (!state) return;
        const e = event as V7StepStartEvent;
        applyRuntimeContext(state, e.runtimeContext);
        // Open the step (model `chat`) span now so tool calls that finish within this step parent
        // to it. Attributes/finish state land at onStepEnd; the span ends there too.
        const startedAt = new Date();
        const span = emitter.tracer.startSpan(
          "chat",
          {
            startTime: startedAt,
            kind: SpanKind.CLIENT,
            attributes: omitUndefined({
              ...conversationAttributes(state),
              "gen_ai.operation.name": "chat",
              "gen_ai.provider.name": providerLabel(e.provider),
              "gen_ai.request.model": e.modelId,
            }),
          },
          state.rootCtx,
        );
        state.steps.set(e.stepNumber, {
          span,
          ctx: trace.setSpan(state.rootCtx, span),
          startedAt,
          messages: e.messages,
          open: true,
        });
        state.currentStepNumber = e.stepNumber;
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    onStepEnd(event) {
      try {
        const state = stateOf(event);
        if (!state) return;
        const e = event as V7StepEndEvent;
        applyRuntimeContext(state, e.runtimeContext);
        const endedAt = new Date();
        let step = state.steps.get(e.stepNumber);
        if (!step) {
          // No matching onStepStart: open the span anchored to the trace start so startTime never
          // exceeds endTime.
          const span = emitter.tracer.startSpan(
            "chat",
            {
              startTime: state.rootStartedAt,
              kind: SpanKind.CLIENT,
              attributes: conversationAttributes(state),
            },
            state.rootCtx,
          );
          step = {
            span,
            ctx: trace.setSpan(state.rootCtx, span),
            startedAt: state.rootStartedAt,
            open: true,
          };
          state.steps.set(e.stepNumber, step);
        }
        const startedAt =
          step.startedAt.getTime() <= endedAt.getTime() ? step.startedAt : state.rootStartedAt;
        const usage = e.usage;
        const inputTokens = usage.inputTokens ?? null;
        const outputTokens = usage.outputTokens ?? null;
        // ai@7 has no flat cachedInputTokens/reasoningTokens fallbacks: details only.
        const cacheReadTokens = usage.inputTokenDetails?.cacheReadTokens ?? null;
        const cacheCreationTokens = usage.inputTokenDetails?.cacheWriteTokens ?? null;
        const reasoningTokens = usage.outputTokenDetails?.reasoningTokens ?? null;
        const stepProvider = providerLabel(e.model.provider);
        state.responseModel = e.response?.modelId ?? state.responseModel;
        const timeToFirstOutputMs = e.performance?.timeToFirstOutputMs;

        step.span.setAttributes(
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
            "gen_ai.input.messages": state.recordInputs ? jsonAttr(step.messages) : undefined,
            "gen_ai.output.messages": state.recordOutputs ? jsonAttr(e.text) : undefined,
            // Seconds, per semconv; ingest converts back to ms.
            "gen_ai.client.operation.time_to_first_chunk":
              timeToFirstOutputMs != null ? timeToFirstOutputMs / 1000 : undefined,
          }),
        );

        if (e.finishReason === "error") {
          step.span.setStatus({ code: SpanStatusCode.ERROR });
        }

        for (const warning of e.warnings ?? []) {
          const detail =
            warning.message?.constructor === String
              ? String(warning.message)
              : warning.type?.constructor === String
                ? String(warning.type)
                : "warning";
          step.span.addEvent(
            "model.warning",
            omitUndefined({
              "log.severity_number": SEVERITY_WARN,
              "log.message": `Model warning: ${detail}`,
              "warning.type":
                warning.type?.constructor === String ? String(warning.type) : undefined,
            }),
          );
        }

        step.span.end(endedAt);
        step.open = false;
        state.childSpans.push(step.span);
        const responseTimeMs = e.performance?.responseTimeMs;
        state.stepMetrics.push({
          operation: "chat",
          // Model-call time when reported (semconv gen_ai.client.operation.duration), else the
          // wall-clock step time.
          durationSec:
            responseTimeMs != null
              ? responseTimeMs / 1000
              : Math.max(endedAt.getTime() - startedAt.getTime(), 0) / 1000,
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

    // Runs the provider model call inside the current step (or root) span context, so
    // auto-instrumented provider requests parent to the model-call span instead of the ambient
    // OTel context.
    executeLanguageModelCall({ callId, execute }) {
      const state = calls.get(callId);
      if (!state) return execute();
      const ctx =
        (state.currentStepNumber != null
          ? state.steps.get(state.currentStepNumber)?.ctx
          : undefined) ?? state.rootCtx;
      return otelContext.with(ctx, execute);
    },

    onToolExecutionStart(event) {
      try {
        const state = stateOf(event);
        if (!state) return;
        const e = event as V7ToolExecutionStartEvent;
        state.toolStarts.set(e.toolCall.toolCallId, new Date());
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    // Runs the tool's execute function inside a live tool span context, so nested AI SDK calls
    // and auto-instrumented requests made by the tool parent to the tool span. The span is
    // finalized (attributes, status, end) at onToolExecutionEnd, or closed by
    // closeOpenChildSpans if the call fails/aborts first.
    executeTool({ callId, toolCallId, execute }) {
      let ctx: Context | undefined;
      const state = calls.get(callId);
      if (state) {
        try {
          const startedAt = state.toolStarts.get(toolCallId) ?? new Date();
          const parentCtx =
            (state.currentStepNumber != null
              ? state.steps.get(state.currentStepNumber)?.ctx
              : undefined) ?? state.rootCtx;
          const span = emitter.tracer.startSpan(
            "execute_tool",
            { startTime: startedAt, kind: SpanKind.INTERNAL },
            parentCtx,
          );
          state.toolSpans.set(toolCallId, { span, startedAt });
          ctx = trace.setSpan(parentCtx, span);
        } catch (err) {
          onError?.(err instanceof Error ? err : String(err));
        }
      }
      return ctx ? otelContext.with(ctx, execute) : execute();
    },

    onToolExecutionEnd(event) {
      try {
        const state = stateOf(event);
        if (!state) return;
        const e = event as V7ToolExecutionEndEvent;
        const endedAt = new Date();
        const success = e.toolOutput.type === "tool-result";
        const attributes = omitUndefined({
          "gen_ai.operation.name": "execute_tool",
          "gen_ai.tool.name": e.toolCall.toolName,
          "gen_ai.tool.call.id": e.toolCall.toolCallId,
          ...conversationAttributes(state),
          "gen_ai.tool.call.arguments": state.recordInputs ? jsonAttr(e.toolCall.input) : undefined,
          "gen_ai.tool.call.result":
            success && state.recordOutputs && "output" in e.toolOutput
              ? jsonAttr(e.toolOutput.output)
              : undefined,
        });

        // Prefer the live span opened by the executeTool wrapper; without one (wrapper not
        // invoked by this ai version), create the span after the fact.
        let span = state.toolSpans.get(e.toolCall.toolCallId)?.span;
        state.toolSpans.delete(e.toolCall.toolCallId);
        if (span) {
          span.setAttributes(attributes);
        } else {
          const started =
            state.toolStarts.get(e.toolCall.toolCallId) ??
            new Date(endedAt.getTime() - e.toolExecutionMs);
          const startedAt = started.getTime() <= endedAt.getTime() ? started : state.rootStartedAt;
          // The event carries no stepNumber: the tool belongs to the currently open step.
          const parentCtx =
            (state.currentStepNumber != null
              ? state.steps.get(state.currentStepNumber)?.ctx
              : undefined) ?? state.rootCtx;
          span = emitter.tracer.startSpan(
            "execute_tool",
            { startTime: startedAt, kind: SpanKind.INTERNAL, attributes },
            parentCtx,
          );
        }
        state.toolStarts.delete(e.toolCall.toolCallId);

        if (e.toolOutput.type === "tool-error") {
          const error = e.toolOutput.error;
          const message = error instanceof Error ? error.message : String(error);
          const errorType = error instanceof Error ? error.name : "tool_error";
          span.setStatus({ code: SpanStatusCode.ERROR });
          span.setAttribute("error.type", errorType);
          span.addEvent("exception", {
            "exception.type": errorType,
            "exception.message": message,
            "log.severity_number": SEVERITY_ERROR,
          });
        }

        span.end(endedAt);
        state.childSpans.push(span);
        state.hasToolSpan = true;
        state.toolMetrics.push({ durationSec: e.toolExecutionMs / 1000 });
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    // Deprecated in ai@7 but the only per-model-call signal for generateObject/streamObject:
    // exactly one step (step 0) per object operation.
    onObjectStepStart(event) {
      try {
        const state = stateOf(event);
        if (!state) return;
        const e = event as V7ObjectStepStartEvent;
        const startedAt = new Date();
        const span = emitter.tracer.startSpan(
          "chat",
          {
            startTime: startedAt,
            kind: SpanKind.CLIENT,
            attributes: omitUndefined({
              ...conversationAttributes(state),
              "gen_ai.operation.name": "chat",
              "gen_ai.provider.name": providerLabel(e.provider ?? state.provider),
              "gen_ai.request.model": e.modelId ?? state.model,
              "gen_ai.output.type": "json",
              "gen_ai.input.messages": state.recordInputs ? jsonAttr(e.promptMessages) : undefined,
            }),
          },
          state.rootCtx,
        );
        state.objectStep = { span, startedAt };
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    onObjectStepEnd(event) {
      try {
        const state = stateOf(event);
        if (!state?.objectStep) return;
        const e = event as V7ObjectStepEndEvent;
        const endedAt = new Date();
        const { span, startedAt } = state.objectStep;
        const usage = e.usage;
        const inputTokens = usage.inputTokens ?? null;
        const outputTokens = usage.outputTokens ?? null;
        state.responseModel = e.response?.modelId ?? state.responseModel;

        span.setAttributes(
          omitUndefined({
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": state.provider,
            "gen_ai.request.model": state.model,
            "gen_ai.response.model": e.response?.modelId ?? undefined,
            "gen_ai.response.id": e.response?.id ?? undefined,
            "gen_ai.usage.input_tokens": inputTokens ?? undefined,
            "gen_ai.usage.output_tokens": outputTokens ?? undefined,
            "gen_ai.usage.cache_read.input_tokens":
              usage.inputTokenDetails?.cacheReadTokens ?? undefined,
            "gen_ai.usage.cache_creation.input_tokens":
              usage.inputTokenDetails?.cacheWriteTokens ?? undefined,
            "gen_ai.usage.reasoning.output_tokens":
              usage.outputTokenDetails?.reasoningTokens ?? undefined,
            "gen_ai.response.finish_reasons": e.finishReason ? [e.finishReason] : undefined,
            "gen_ai.output.messages": state.recordOutputs ? jsonAttr(e.objectText) : undefined,
            "gen_ai.client.operation.time_to_first_chunk":
              e.msToFirstChunk != null ? e.msToFirstChunk / 1000 : undefined,
          }),
        );

        if (e.finishReason === "error") {
          span.setStatus({ code: SpanStatusCode.ERROR });
        }

        span.end(endedAt);
        state.childSpans.push(span);
        state.objectStep = undefined;
        state.stepMetrics.push({
          operation: "chat",
          durationSec: Math.max(endedAt.getTime() - startedAt.getTime(), 0) / 1000,
          inputTokens,
          outputTokens,
          provider: state.provider,
          requestModel: state.model,
          responseModel: e.response?.modelId ?? null,
        });
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    // Per doEmbed model call; embedMany may fire several (keyed by embedCallId).
    onEmbedStart(event) {
      try {
        const state = stateOf(event);
        if (!state) return;
        const e = event as V7EmbedCallStartEvent;
        const startedAt = new Date();
        const span = emitter.tracer.startSpan(
          "embeddings",
          {
            startTime: startedAt,
            kind: SpanKind.CLIENT,
            attributes: omitUndefined({
              ...conversationAttributes(state),
              "gen_ai.operation.name": "embeddings",
              "gen_ai.provider.name": providerLabel(e.provider ?? state.provider),
              "gen_ai.request.model": e.modelId ?? state.model,
            }),
          },
          state.rootCtx,
        );
        state.embedSpans.set(e.embedCallId, { span, startedAt });
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    onEmbedEnd(event) {
      try {
        const state = stateOf(event);
        if (!state) return;
        const e = event as V7EmbedCallEndEvent;
        const entry = state.embedSpans.get(e.embedCallId);
        if (!entry) return;
        const endedAt = new Date();
        const inputTokens = e.usage?.tokens ?? null;

        entry.span.setAttributes(
          omitUndefined({
            "gen_ai.operation.name": "embeddings",
            "gen_ai.provider.name": state.provider,
            "gen_ai.request.model": state.model,
            "gen_ai.usage.input_tokens": inputTokens ?? undefined,
          }),
        );
        entry.span.end(endedAt);
        state.childSpans.push(entry.span);
        state.embedSpans.delete(e.embedCallId);
        state.stepMetrics.push({
          operation: "embeddings",
          durationSec: Math.max(endedAt.getTime() - entry.startedAt.getTime(), 0) / 1000,
          inputTokens,
          outputTokens: null,
          provider: state.provider,
          requestModel: state.model,
          responseModel: null,
        });
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    onRerankStart(event) {
      try {
        const state = stateOf(event);
        if (!state) return;
        const startedAt = new Date();
        const e = event as { provider?: string; modelId?: string };
        const span = emitter.tracer.startSpan(
          "rerank",
          {
            startTime: startedAt,
            kind: SpanKind.CLIENT,
            attributes: omitUndefined({
              ...conversationAttributes(state),
              "gen_ai.operation.name": "rerank",
              "gen_ai.provider.name": providerLabel(e.provider ?? state.provider),
              "gen_ai.request.model": e.modelId ?? state.model,
            }),
          },
          state.rootCtx,
        );
        state.rerankSpan = { span, startedAt };
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    onRerankEnd(event) {
      try {
        const state = stateOf(event);
        if (!state?.rerankSpan) return;
        const endedAt = new Date();
        const { span, startedAt } = state.rerankSpan;
        span.setAttributes(
          omitUndefined({
            "gen_ai.operation.name": "rerank",
            "gen_ai.provider.name": state.provider,
            "gen_ai.request.model": state.model,
          }),
        );
        span.end(endedAt);
        state.childSpans.push(span);
        state.rerankSpan = undefined;
        state.stepMetrics.push({
          operation: "rerank",
          durationSec: Math.max(endedAt.getTime() - startedAt.getTime(), 0) / 1000,
          inputTokens: null,
          outputTokens: null,
          provider: state.provider,
          requestModel: state.model,
          responseModel: null,
        });
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    async onEnd(event) {
      try {
        const callId = callIdOf(event);
        const state = callId === undefined ? undefined : calls.get(callId);
        if (!state || callId === undefined) return;
        const e = event as V7EndEvent;
        applyRuntimeContext(state, e.runtimeContext);
        const endedAt = new Date();
        const root = state.rootSpan;

        if (state.opKind === "text") {
          const finishReason = e.finishReason ?? "unknown";
          const hasError = e.finishReason === "error";
          // Aggregated across steps by the SDK — use directly, do not re-sum.
          const inputTokens = e.usage?.inputTokens;
          const outputTokens = e.usage?.outputTokens;
          const tokenParts: string[] = [];
          if (inputTokens != null) tokenParts.push(`${inputTokens} in`);
          if (outputTokens != null) tokenParts.push(`${outputTokens} out`);
          const tokenText = tokenParts.length > 0 ? `: ${tokenParts.join(" / ")} tokens` : "";

          root.setAttributes(
            omitUndefined({
              "gen_ai.operation.name": state.hasToolSpan ? "invoke_agent" : "chat",
              ...rootBaseAttributes(state),
              "gen_ai.output.messages": state.recordOutputs ? jsonAttr(e.text) : undefined,
              "gen_ai.response.finish_reasons": e.finishReason ? [e.finishReason] : undefined,
            }),
          );

          root.addEvent(
            "generation.summary",
            omitUndefined({
              "log.severity_number": hasError ? SEVERITY_ERROR : SEVERITY_INFO,
              "log.message": hasError
                ? `Generation failed (${finishReason})`
                : `Generation completed (${finishReason})${tokenText}`,
              "gen_ai.usage.input_tokens": inputTokens ?? undefined,
              "gen_ai.usage.output_tokens": outputTokens ?? undefined,
            }),
          );

          if (hasError) {
            root.setStatus({ code: SpanStatusCode.ERROR });
            root.setAttribute("error.type", finishReason);
            root.addEvent("exception", {
              "exception.type": finishReason,
              "exception.message": `Generation failed (${finishReason})`,
              "log.severity_number": SEVERITY_ERROR,
            });
          }
        } else if (state.opKind === "object") {
          root.setAttributes(
            omitUndefined({
              "gen_ai.operation.name": "chat",
              "gen_ai.output.type": "json",
              ...rootBaseAttributes(state),
              "gen_ai.output.messages": state.recordOutputs ? jsonAttr(e.object) : undefined,
              "gen_ai.response.finish_reasons": e.finishReason ? [e.finishReason] : undefined,
            }),
          );
          // streamObject reports parse/schema-validation failures via `error` on the end event
          // (finishReason may still be "stop"); generateObject throws into onError instead.
          if (e.error !== undefined || e.finishReason === "error") {
            const errorType = e.error instanceof Error ? e.error.name || "error" : "error";
            const message =
              e.error !== undefined
                ? e.error instanceof Error
                  ? e.error.message
                  : (jsonAttr(e.error) ?? "unknown error")
                : `Generation failed (${e.finishReason})`;
            root.setStatus({ code: SpanStatusCode.ERROR });
            root.setAttribute("error.type", errorType);
            root.addEvent("exception", {
              "exception.type": errorType,
              "exception.message": message,
              "log.severity_number": SEVERITY_ERROR,
            });
          }
        } else if (state.opKind === "embed") {
          root.setAttributes(
            omitUndefined({
              "gen_ai.operation.name": "embeddings",
              ...rootBaseAttributes(state),
              "gen_ai.usage.input_tokens": e.usage?.tokens ?? undefined,
            }),
          );
        } else {
          state.responseModel = e.response?.modelId ?? state.responseModel;
          root.setAttributes(
            omitUndefined({
              "gen_ai.operation.name": "rerank",
              ...rootBaseAttributes(state),
              "gen_ai.output.messages": state.recordOutputs ? jsonAttr(e.ranking) : undefined,
            }),
          );
        }

        root.end(endedAt);
        await finishCall(state, callId);
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    // Thrown/unrecoverable failures: mark every open span and the root, flush what we have.
    async onError(event) {
      try {
        if (event === null || event === undefined || Object(event) !== event) return;
        const errorEvent = event as { callId?: string; error?: unknown };
        const callId = callIdOf(errorEvent);
        const state = callId === undefined ? undefined : calls.get(callId);
        if (!state || callId === undefined) return;
        const error = errorEvent.error;
        const endedAt = new Date();
        const errorType = error instanceof Error ? error.name || "error" : "error";
        const message = error instanceof Error ? error.message : String(error);

        closeOpenChildSpans(state, endedAt, { errorType, message });

        state.rootSpan.setAttributes(
          omitUndefined({
            // Preserve the operation kind: a failed embed/rerank must not be labeled chat.
            "gen_ai.operation.name": state.hasToolSpan
              ? "invoke_agent"
              : ROOT_NAME_BY_KIND[state.opKind],
            ...rootBaseAttributes(state),
          }),
        );
        markSpanFailed(state.rootSpan, errorType, message, endedAt);
        await finishCall(state, callId);
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },

    // User-initiated abort (streaming text only): not an error — root status stays UNSET.
    async onAbort(event) {
      try {
        const callId = callIdOf(event);
        const state = callId === undefined ? undefined : calls.get(callId);
        if (!state || callId === undefined) return;
        const endedAt = new Date();

        closeOpenChildSpans(state, endedAt);

        state.rootSpan.setAttributes(
          omitUndefined({
            "gen_ai.operation.name": state.hasToolSpan ? "invoke_agent" : "chat",
            ...rootBaseAttributes(state),
          }),
        );
        state.rootSpan.addEvent("generation.summary", {
          "log.severity_number": SEVERITY_INFO,
          "log.message": "Generation aborted",
        });
        state.rootSpan.end(endedAt);
        await finishCall(state, callId);
      } catch (err) {
        onError?.(err instanceof Error ? err : String(err));
      }
    },
  };

  return integration;
}
