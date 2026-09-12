import {
  flush,
  log,
  startSpan,
  type LogLevel,
  type SpanHandle,
  type TokenUsage,
} from "@telemetry-dev/sdk";

import { ensureInit, type ClientOverrides, type TelemetryDevOmpOptions } from "./config.ts";

/** Options for {@link telemetryDevExtension}. */
export interface TelemetryDevExtensionOptions extends TelemetryDevOmpOptions {
  /** Value for `gen_ai.agent.name` on spans and logs. Defaults to `"omp"`. */
  agentName?: string;
}

/**
 * Structural slice of omp's `ExtensionContext` read by this integration.
 *
 * The host package (`@oh-my-pi/pi-coding-agent`) is an optional peer, so its
 * types must not appear in this package's emitted declarations; assignability
 * to the real host types is asserted in this package's tests.
 */
export interface TelemetryDevExtensionContext {
  /** Current working directory. */
  cwd: string;
  /** Current model descriptor, when one is selected. */
  model: { id?: string; provider?: string } | undefined;
  /** Read-only session manager exposing the session id. */
  sessionManager: { getSessionId(): string | undefined };
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

type JsonValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | JsonValue[]
  | { [key: string]: JsonValue };

type Attrs = { [key: string]: JsonValue };

type JsonRecord = { [key: string]: JsonValue };

function asRecord(value: JsonValue): JsonRecord | undefined {
  if (value === null || Array.isArray(value) || !(value instanceof Object)) return undefined;

  return value as JsonRecord;
}

function stringField(record: JsonRecord | undefined, key: string): string | undefined {
  const value = record?.[key];

  if (value === undefined || value === null || value.constructor !== String || value === "")
    return undefined;

  return value as string;
}

function numberField(record: JsonRecord | undefined, key: string): number | undefined {
  const value = record?.[key];

  if (value === undefined || value === null || value.constructor !== Number) return undefined;
  const number = value as number;

  return Number.isFinite(number) ? number : undefined;
}

function booleanField(record: JsonRecord | undefined, key: string): boolean | undefined {
  const value = record?.[key];

  if (value === undefined || value === null || value.constructor !== Boolean) return undefined;

  return value as boolean;
}

function reportError(onError: ((error: Error) => void) | undefined, cause: unknown): void {
  try {
    onError?.(cause instanceof Error ? cause : new Error(String(cause)));
  } catch {
    // telemetry must never fail an omp session.
  }
}

function sessionId(ctx: TelemetryDevExtensionContext): string | undefined {
  const id = ctx.sessionManager.getSessionId();

  return id && id.length > 0 ? id : undefined;
}

/** Joins the text blocks of an omp message content array. */
function textContent(content: JsonValue): string | undefined {
  if (!Array.isArray(content)) return undefined;
  const parts: string[] = [];

  for (const block of content) {
    const record = asRecord(block);
    const text = record?.text;

    if (
      record?.type === "text" &&
      text !== undefined &&
      text !== null &&
      text.constructor === String
    ) {
      parts.push(text as string);
    }
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
    reasoningOutputTokens: numberField(usage, "reasoningTokens"),
  };
}

function assistantMessageKey(message: JsonRecord): string | undefined {
  const timestamp = numberField(message, "timestamp");

  if (timestamp === undefined) return undefined;

  return JSON.stringify([
    timestamp,
    stringField(message, "provider") ?? "",
    stringField(message, "model") ?? "",
    stringField(message, "responseId") ?? "",
    stringField(message, "stopReason") ?? "",
  ]);
}

/** Named error so `error.type` reflects the failure class instead of "Error". */
function failureError(name: string, message: string | undefined): Error {
  const error = new Error(message ?? name);
  error.name = name;

  return error;
}

/**
 * Creates an omp extension factory that exports agent loops, model calls, and
 * tool executions to telemetry.dev.
 *
 * Trace shape per user prompt: one `invoke_agent` span, with one `chat {model}`
 * child span per assistant message and one `execute_tool {tool}` child span per
 * tool execution. Session lifecycle, compaction, and retry events are emitted
 * as logs joined via `gen_ai.conversation.id`.
 */
export function telemetryDevExtension(
  options: TelemetryDevExtensionOptions = {},
  overrides?: ClientOverrides,
): TelemetryDevExtension {
  const { agentName = "omp", ...sdkOptions } = options;
  const onError = sdkOptions.onError;

  return (pi) => {
    try {
      ensureInit(sdkOptions, overrides);
    } catch (error) {
      reportError(onError, error);
    }

    /** Calls through `pi` so the host's `on` keeps its `this` binding. */
    function register<E extends JsonValue>(
      event: string,
      handler: (event: E, ctx: TelemetryDevExtensionContext) => void | Promise<void>,
    ): void {
      const registerEvent = pi.on.bind(pi) as (
        event: string,
        handler: (event: E, ctx: TelemetryDevExtensionContext) => void | Promise<void>,
      ) => void;

      registerEvent(event, handler);
    }

    let agentSpan: SpanHandle | undefined;
    /** Stable key of the latest assistant message emitted for the active agent span. */
    let agentMessageKey: string | undefined;
    let pendingPrompt: string | undefined;
    /** Chat span that issued each pending tool call, so tool spans nest under it. */
    const chatSpanByToolCall = new Map<string, SpanHandle>();
    const toolSpans = new Map<string, SpanHandle>();

    function baseAttributes(ctx: TelemetryDevExtensionContext) {
      return {
        "gen_ai.conversation.id": sessionId(ctx),
        "gen_ai.agent.name": agentName,
      } satisfies Attrs;
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
      agentMessageKey = undefined;
    }

    function on<E extends JsonValue>(
      event: string,
      handler: (event: E, ctx: TelemetryDevExtensionContext) => void | Promise<void>,
    ): void {
      register<E>(event, (payload, ctx) => {
        try {
          return handler(payload, ctx);
        } catch (error) {
          reportError(onError, error);

          return undefined;
        }
      });
    }

    register("before_agent_start", (event: { prompt?: JsonValue }) => {
      try {
        pendingPrompt = stringField(event, "prompt");
      } catch (error) {
        reportError(onError, error);
      }
    });

    on("agent_start", (_event, ctx) => {
      if (agentSpan) {
        if (pendingPrompt === undefined) {
          // Continuations (auto-retry, compaction, queued continuations)
          // re-enter the loop without a fresh before_agent_start prompt, and
          // the host launches the willContinue agent_end notification without
          // awaiting it, so this agent_start can arrive before that event.
          // Keep the original prompt span open until the terminal agent_end.
          return;
        }

        // A fresh prompt while a previous loop never emitted agent_end: close
        // the dangling span instead of silently merging two prompts into it.
        endAgentSpan({ finishReason: "incomplete" });
      }

      agentSpan = startSpan("invoke_agent", {
        type: "agent",
        agentName,
        input: pendingPrompt,
        attributes: spanAttributes(ctx),
      });
      pendingPrompt = undefined;
    });

    on("agent_end", (event: { messages: JsonValue[]; willContinue?: boolean }, _ctx) => {
      if (!agentSpan) return;

      if (event.willContinue === true) {
        // The host scheduled another loop for this prompt (auto-retry,
        // compaction, or a queued continuation); this agent_end is not
        // terminal, so keep the prompt span open.
        void flush();

        return;
      }

      const messages = Array.isArray(event.messages) ? event.messages : [];
      let lastAssistant: JsonRecord | undefined;

      for (const message of messages) {
        const record = asRecord(message);

        if (record?.role === "assistant") lastAssistant = record;
      }

      if (lastAssistant) {
        const eventMessageKey = assistantMessageKey(lastAssistant);

        if (eventMessageKey === undefined || eventMessageKey !== agentMessageKey) {
          void flush();

          return;
        }
      }

      const stopReason = stringField(lastAssistant, "stopReason");
      const errorMessage = stringField(lastAssistant, "errorMessage");
      endAgentSpan({
        output: textContent(lastAssistant?.content),
        finishReason: stopReason,
        error: stopReason === "error" ? failureError(stopReason, errorMessage) : undefined,
      });
      void flush();
    });

    on("message_end", (event: { message: JsonValue }, ctx) => {
      const message = asRecord(event.message);

      if (message?.role !== "assistant") return;

      if (agentSpan) agentMessageKey = assistantMessageKey(message);
      const model = stringField(message, "model");
      const startTime = numberField(message, "timestamp");
      const duration = numberField(message, "duration");
      const errorMessage = stringField(message, "errorMessage");
      const stopReason = stringField(message, "stopReason");

      const span = startSpan(model ? `chat ${model}` : "chat", {
        type: "generation",
        parent: agentSpan,
        startTime,
        model,
        provider: stringField(message, "provider"),
        responseId: stringField(message, "responseId"),
        usage: usageFields(message),
        finishReason: stopReason,
        timeToFirstChunkMs: numberField(message, "ttft"),
        output: textContent(message.content),
        error: stopReason === "error" ? failureError(stopReason, errorMessage) : undefined,
        attributes: spanAttributes(ctx),
      });

      for (const block of Array.isArray(message.content) ? message.content : []) {
        const record = asRecord(block);
        const id = stringField(record, "id");

        if (record?.type === "toolCall" && id !== undefined) {
          chatSpanByToolCall.set(id, span);
        }
      }

      span.end({
        endTime:
          startTime !== undefined && duration !== undefined ? startTime + duration : undefined,
      });
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
      emit("turn_start", ctx, "debug", "Turn started", { "omp.turn.index": event.turnIndex });
    });

    on("turn_end", (event: { turnIndex: number }, ctx) => {
      emit("turn_end", ctx, "debug", "Turn completed", { "omp.turn.index": event.turnIndex });
    });

    on("session_start", (_event, ctx) => {
      const model = asRecord(ctx.model);
      emit("session_start", ctx, "info", "Session started", {
        "gen_ai.request.model": stringField(model, "id"),
        "gen_ai.provider.name": stringField(model, "provider"),
        "omp.cwd": ctx.cwd,
      });
    });

    on("session_switch", (_event, ctx) => {
      emit("session_switch", ctx, "info", "Session switched");
    });

    on("session_branch", (_event, ctx) => {
      emit("session_branch", ctx, "info", "Session branched");
    });

    on("session_compact", (_event, ctx) => {
      emit("session_compact", ctx, "info", "Session compacted");
    });

    on("auto_compaction_start", (event: { reason: string; action: string }, ctx) => {
      emit("auto_compaction_start", ctx, "info", `Auto-compaction started (${event.reason})`, {
        "omp.compaction.reason": event.reason,
        "omp.compaction.action": event.action,
      });
    });

    on(
      "auto_compaction_end",
      (
        event: {
          action: string;
          aborted: boolean;
          willRetry: boolean;
          errorMessage?: string;
          skipped?: boolean;
        },
        ctx,
      ) => {
        const record = asRecord(event);
        const errorMessage = stringField(record, "errorMessage");
        emit(
          "auto_compaction_end",
          ctx,
          errorMessage ? "error" : "info",
          errorMessage ? `Auto-compaction failed: ${errorMessage}` : "Auto-compaction completed",
          {
            "omp.compaction.action": event.action,
            "omp.compaction.aborted": booleanField(record, "aborted"),
            "omp.compaction.skipped": booleanField(record, "skipped"),
          },
        );
      },
    );

    on(
      "auto_retry_start",
      (
        event: { attempt: number; maxAttempts: number; delayMs: number; errorMessage: string },
        ctx,
      ) => {
        emit(
          "auto_retry_start",
          ctx,
          "warn",
          `Auto-retry attempt ${event.attempt}/${event.maxAttempts}: ${event.errorMessage}`,
          {
            "omp.retry.attempt": event.attempt,
            "omp.retry.max_attempts": event.maxAttempts,
            "omp.retry.delay_ms": event.delayMs,
          },
        );
      },
    );

    on(
      "auto_retry_end",
      (event: { success: boolean; attempt: number; finalError?: string }, ctx) => {
        const finalError = stringField(asRecord(event), "finalError");
        emit(
          "auto_retry_end",
          ctx,
          event.success ? "info" : "error",
          event.success
            ? `Auto-retry succeeded after ${event.attempt} attempt(s)`
            : `Auto-retry failed: ${finalError ?? "unknown error"}`,
          { "omp.retry.attempt": event.attempt },
        );
      },
    );

    register("session_shutdown", async (_event: JsonValue, ctx: TelemetryDevExtensionContext) => {
      try {
        if (agentSpan) endAgentSpan({ finishReason: "incomplete" });
        emit("session_shutdown", ctx, "info", "Session shutdown");
        await flush();
      } catch (error) {
        reportError(onError, error);
      }
    });
  };
}
