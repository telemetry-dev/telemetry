import {
  flush,
  log,
  startSpan,
  type LogLevel,
  type LogOptions,
  type SpanHandle,
  type TokenUsage,
} from "@telemetry-dev/sdk";

import { ensureInit, type ClientOverrides, type TelemetryDevPiOptions } from "./config.ts";

/** Options for {@link telemetryDevExtension}. */
export interface TelemetryDevExtensionOptions extends TelemetryDevPiOptions {
  /** Value for `gen_ai.agent.name` on spans and logs. Defaults to `"pi"`. */
  agentName?: string;
}

/**
 * Structural slice of pi's `ExtensionContext` read by this integration.
 *
 * The host package (`@earendil-works/pi-coding-agent`) is an optional peer, so
 * its types must not appear in this package's emitted declarations;
 * assignability to the real host types is asserted in this package's tests.
 */
export interface TelemetryDevExtensionContext {
  /** Current working directory. */
  cwd: string;
  /** Current model descriptor, when one is selected. */
  model: unknown;
  /** Read-only session manager exposing the session id. */
  sessionManager: { getSessionId(): string | undefined };
  /** Effective system prompt after all before_agent_start handlers. */
  getSystemPrompt(): string;
}

/**
 * Structural stand-in for the host `ExtensionAPI` passed to extension
 * factories. The `never` parameters make any host event-subscription surface
 * assignable; this integration is not meant to be called through this type.
 */
export interface TelemetryDevExtensionHost {
  on(event: never, handler: never): void;
}

/** Structural stand-in for the host `ExtensionFactory` type. */
export type TelemetryDevExtension = (pi: TelemetryDevExtensionHost) => void;

type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

type JsonRecord = { [key: string]: JsonValue };

type Attrs = NonNullable<LogOptions["attributes"]>;

function asRecord(cause: unknown): JsonRecord | undefined {
  if (cause === null || cause instanceof Function || Object(cause) !== cause) return undefined;

  return cause as JsonRecord;
}

function stringField(record: JsonRecord | undefined, key: string): string | undefined {
  const value = record?.[key];

  return value?.constructor === String && value.length > 0 ? value : undefined;
}

function numberField(record: JsonRecord | undefined, key: string): number | undefined {
  const value = record?.[key];

  return value?.constructor === Number && Number.isFinite(value) ? value : undefined;
}

function booleanField(record: JsonRecord | undefined, key: string): boolean | undefined {
  const value = record?.[key];

  return value?.constructor === Boolean ? value : undefined;
}

function reportError(onError: Function | undefined, cause: unknown): void {
  try {
    onError?.call(undefined, cause);
  } catch {
    // telemetry must never fail a pi session.
  }
}

function sessionId(ctx: TelemetryDevExtensionContext): string | undefined {
  const id = ctx.sessionManager.getSessionId();

  return id?.constructor === String && id.length > 0 ? id : undefined;
}

/** Joins the text blocks of a pi message content array. */
function textContent(content: JsonValue | undefined): string | undefined {
  if (!Array.isArray(content)) return undefined;
  const parts: string[] = [];

  for (const block of content) {
    const record = asRecord(block);

    if (record?.type === "text" && record.text?.constructor === String) parts.push(record.text);
  }

  return parts.length > 0 ? parts.join("\n") : undefined;
}

function usageFields(message: JsonRecord): TokenUsage | undefined {
  const usage = asRecord(message.usage);

  if (!usage) return undefined;

  return {
    inputTokens: numberField(usage, "input"),
    outputTokens: numberField(usage, "output"),
    totalTokens: numberField(usage, "totalTokens"),
    cacheReadInputTokens: numberField(usage, "cacheRead"),
    cacheCreationInputTokens: numberField(usage, "cacheWrite"),
    reasoningOutputTokens: numberField(usage, "reasoning"),
  };
}

/** Named error so `error.type` reflects the failure class instead of "Error". */
function failureError(name: string, message: string | undefined): Error {
  const error = new Error(message ?? name);
  error.name = name;

  return error;
}

/**
 * Creates a pi extension factory that exports agent loops, model calls, tool
 * executions, and lifecycle logs to telemetry.dev.
 *
 * Trace shape: one trace per pi session with a `session` root span. Each user
 * prompt is an `invoke_agent` child, with one `chat {model}` child span per
 * assistant message and one `execute_tool {tool}` child span per tool
 * execution. Payloads include the prompt and its images, the provider-formatted
 * request, and every returned content block (thinking, text, tool calls).
 * Logs join the same session through `gen_ai.conversation.id`.
 */
export function telemetryDevExtension(
  options: TelemetryDevExtensionOptions = {},
  overrides?: ClientOverrides,
): TelemetryDevExtension {
  const { agentName = "pi", ...sdkOptions } = options;
  const onError = sdkOptions.onError;

  return (pi) => {
    try {
      ensureInit(sdkOptions, overrides);
    } catch (error) {
      reportError(onError, error);
    }

    /** Calls through `pi` so the host's `on` keeps its `this` binding. */
    function register<Event>(
      event: string,
      handler: (event: Event, ctx: TelemetryDevExtensionContext) => void | Promise<void>,
    ): void {
      pi.on(event as never, handler as never);
    }

    /** Root of the one trace per pi session; every prompt span nests under it. */
    let sessionSpan: SpanHandle | undefined;
    let agentSpan: SpanHandle | undefined;
    let assistantStartedAt: number | undefined;
    let pendingPrompt: JsonValue | undefined;
    let systemPrompt: string | undefined;
    let request: unknown;
    /** Latest agent_end outcome, applied to the prompt span when the run settles. */
    let lastRunResult: Parameters<SpanHandle["end"]>[0];
    /** Chat span that issued each pending tool call, so tool spans nest under it. */
    const chatSpanByToolCall = new Map<string, SpanHandle>();
    const toolSpans = new Map<string, SpanHandle>();

    function baseAttributes(ctx: TelemetryDevExtensionContext) {
      return {
        "gen_ai.conversation.id": sessionId(ctx),
        "gen_ai.agent.name": agentName,
      };
    }

    function emit(
      eventName: string,
      ctx: TelemetryDevExtensionContext,
      level: LogLevel,
      message: string,
      attributes: Attrs = {},
    ): void {
      log(message, { level, eventName, attributes: { ...baseAttributes(ctx), ...attributes } });
    }

    function spanAttributes(ctx: TelemetryDevExtensionContext): Record<string, string> {
      const id = sessionId(ctx);

      return id ? { "gen_ai.conversation.id": id } : {};
    }

    function endDanglingToolSpans(): void {
      for (const span of toolSpans.values()) {
        span.end({ error: failureError("incomplete", "tool execution did not complete") });
      }

      toolSpans.clear();
      chatSpanByToolCall.clear();
    }

    function endAgentSpan(fields: Parameters<SpanHandle["end"]>[0]): void {
      endDanglingToolSpans();
      agentSpan?.end(fields);
      agentSpan = undefined;
    }

    function endSessionSpan(): void {
      sessionSpan?.end();
      sessionSpan = undefined;
    }

    function on<Event>(
      event: string,
      handler: (event: Event, ctx: TelemetryDevExtensionContext) => void,
    ): void {
      register(event, (payload: Event, ctx) => {
        try {
          handler(payload, ctx);
        } catch (error) {
          reportError(onError, error);
        }
      });
    }

    register("before_agent_start", (event: { prompt?: JsonValue; images?: JsonValue[] }) => {
      try {
        const prompt = event.prompt?.constructor === String ? event.prompt : undefined;
        pendingPrompt =
          Array.isArray(event.images) && event.images.length > 0
            ? [{ type: "text", text: prompt ?? "" }, ...event.images]
            : prompt;
      } catch (error) {
        reportError(onError, error);
      }
    });

    // All context handlers and host conversion run before this hook.
    on("before_provider_request", (event: { payload: unknown }, ctx) => {
      request = event.payload;
      systemPrompt = ctx.getSystemPrompt();
    });

    on("agent_start", (_event, ctx) => {
      // Automatic retries, compaction, and queued continuations re-enter the
      // agent loop before the run settles; keep the original prompt span open.
      if (agentSpan) return;
      sessionSpan ??= startSpan("session", {
        type: "agent",
        agentName,
        attributes: spanAttributes(ctx),
      });
      agentSpan = startSpan("invoke_agent", {
        type: "agent",
        parent: sessionSpan,
        agentName,
        input: pendingPrompt,
        attributes: spanAttributes(ctx),
      });
      pendingPrompt = undefined;
    });

    on("agent_end", (event: { messages: JsonValue[] }, _ctx) => {
      const messages = Array.isArray(event.messages) ? event.messages : [];
      let lastAssistant: JsonRecord | undefined;

      for (const message of messages) {
        const record = asRecord(message);

        if (record?.role === "assistant") lastAssistant = record;
      }

      const stopReason = stringField(lastAssistant, "stopReason");
      const errorMessage = stringField(lastAssistant, "errorMessage");
      // agent_end only closes one low-level run; pi may remove a retryable
      // failure and start another run before the prompt settles. Record the
      // outcome and close the span at agent_settled.
      lastRunResult = {
        output: lastAssistant?.content,
        finishReason: stopReason,
        error: stopReason === "error" ? failureError(stopReason, errorMessage) : undefined,
      };
      void flush();
    });

    on("agent_settled", (_event, _ctx) => {
      endAgentSpan(lastRunResult ?? { finishReason: "incomplete" });
      lastRunResult = undefined;
      void flush();
    });

    on("message_start", (event: { message: JsonValue }, _ctx) => {
      const message = asRecord(event.message);

      if (message?.role === "assistant") assistantStartedAt = Date.now();
    });

    on("message_end", (event: { message: JsonValue }, ctx) => {
      const message = asRecord(event.message);

      if (message?.role !== "assistant") return;
      const startTime = assistantStartedAt;
      assistantStartedAt = undefined;
      const endTime = Date.now();
      const model = stringField(message, "model");
      const responseModel = stringField(message, "responseModel") ?? model;
      const errorMessage = stringField(message, "errorMessage");
      const stopReason = stringField(message, "stopReason");

      const span = startSpan(model ? `chat ${model}` : "chat", {
        type: "generation",
        parent: agentSpan,
        startTime,
        model,
        provider: stringField(message, "provider"),
        responseModel,
        responseId: stringField(message, "responseId"),
        usage: usageFields(message),
        costUsd: numberField(asRecord(asRecord(message.usage)?.cost), "total"),
        finishReason: stopReason,
        systemInstructions: systemPrompt,
        input: request,
        output: message.content,
        error: stopReason === "error" ? failureError(stopReason, errorMessage) : undefined,
        attributes: spanAttributes(ctx),
      });

      request = undefined;
      systemPrompt = undefined;

      for (const block of Array.isArray(message.content) ? message.content : []) {
        const record = asRecord(block);

        if (record?.type === "toolCall" && record.id?.constructor === String) {
          chatSpanByToolCall.set(record.id, span);
        }
      }

      span.end({ endTime });
    });

    on(
      "tool_execution_start",
      (event: { toolCallId: string; toolName: string; args: JsonValue }, ctx) => {
        const span = startSpan(`execute_tool ${event.toolName}`, {
          type: "tool",
          parent: chatSpanByToolCall.get(event.toolCallId) ?? agentSpan,
          toolName: event.toolName,
          toolCallId: event.toolCallId,
          input: event.args,
          attributes: spanAttributes(ctx),
        });

        chatSpanByToolCall.delete(event.toolCallId);
        toolSpans.set(event.toolCallId, span);
      },
    );

    on(
      "tool_execution_end",
      (
        event: { toolCallId: string; toolName: string; result: JsonValue; isError: boolean },
        _ctx,
      ) => {
        const span = toolSpans.get(event.toolCallId);

        if (!span) return;
        toolSpans.delete(event.toolCallId);
        const resultText = textContent(asRecord(event.result)?.content);
        span.end({
          output: event.result,
          error: event.isError ? failureError("ToolExecutionError", resultText) : undefined,
        });
      },
    );

    on("turn_start", (event: { turnIndex: number }, ctx) => {
      emit("turn_start", ctx, "debug", "Turn started", { "pi.turn.index": event.turnIndex });
    });

    on("turn_end", (event: { turnIndex: number }, ctx) => {
      emit("turn_end", ctx, "debug", "Turn completed", { "pi.turn.index": event.turnIndex });
    });

    on("session_start", (event: { reason: string; previousSessionFile?: string }, ctx) => {
      // A new, resumed, forked, or switched session is a new trace.
      endSessionSpan();
      const model = asRecord(ctx.model);
      emit("session_start", ctx, "info", "Session started", {
        "gen_ai.request.model": stringField(model, "id"),
        "gen_ai.provider.name": stringField(model, "provider"),
        "pi.cwd": ctx.cwd,
        "pi.session.reason": event.reason,
      });
    });

    on("session_compact", (event: { reason: string; fromExtension: boolean }, ctx) => {
      emit("session_compact", ctx, "info", "Session compacted", {
        "pi.compaction.from_extension": booleanField(asRecord(event), "fromExtension"),
        "pi.compaction.reason": event.reason,
      });
    });

    on(
      "model_select",
      (event: { model: JsonValue; previousModel: JsonValue; source: string }, ctx) => {
        emit("model_select", ctx, "info", "Model changed", {
          "pi.model.from": stringField(asRecord(event.previousModel), "id"),
          "pi.model.source": event.source,
          "pi.model.to": stringField(asRecord(event.model), "id"),
        });
      },
    );

    register(
      "session_shutdown",
      async (event: { reason?: string }, ctx: TelemetryDevExtensionContext) => {
        try {
          if (agentSpan || toolSpans.size > 0) {
            endAgentSpan(lastRunResult ?? { finishReason: "incomplete" });
            lastRunResult = undefined;
          }

          endSessionSpan();
          emit("session_shutdown", ctx, "info", "Session shutdown", {
            "pi.session.reason": event.reason,
          });
          await flush();
        } catch (error) {
          reportError(onError, error);
        }
      },
    );
  };
}
