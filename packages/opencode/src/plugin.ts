import {
  flush,
  log,
  startSpan,
  type LogLevel,
  type SpanHandle,
  type TokenUsage,
} from "@telemetry-dev/sdk";

import { ensureInit, type ClientOverrides, type TelemetryDevOpencodeOptions } from "./config.ts";
import { asJsonObject, type JsonObject, type JsonValue } from "./type-guards.ts";

/** Options for {@link telemetryDevPlugin}. */
export interface TelemetryDevPluginOptions extends TelemetryDevOpencodeOptions {
  /** Value for `gen_ai.agent.name` on spans and logs. Defaults to `"opencode"`. */
  agentName?: string;
}

/**
 * Structural stand-ins for opencode's plugin types. The host package
 * (`@opencode-ai/plugin`) is an optional peer, so its types must not appear in
 * this package's emitted declarations; assignability to the real host types is
 * asserted in this package's tests.
 */
export interface TelemetryDevHooks {
  "chat.message": (
    input: { sessionID: string; agent?: string },
    output: { message: object; parts: object[] },
  ) => Promise<void>;
  event: (input: { event: object }) => Promise<void>;
  "tool.execute.before": (
    input: { tool: string; sessionID: string; callID: string },
    output: { args: JsonValue },
  ) => Promise<void>;
  /** opencode's synthetic Task path invokes this hook with no output on failure. */
  "tool.execute.after": (
    input: { tool: string; sessionID: string; callID: string; args: JsonValue },
    output: { title: string; output: string; metadata: JsonValue } | undefined,
  ) => Promise<void>;
  dispose: () => Promise<void>;
}

/** Structural stand-in for the host `Plugin` type. */
export type TelemetryDevPlugin = <Input, Options extends object>(
  input: Input,
  options?: Options,
) => Promise<TelemetryDevHooks>;

type Attrs = { [key: string]: JsonValue };

type PendingTool = {
  sessionID: string;
  callID: string;
  tool: string;
  args: JsonValue;
  startedAt: number;
};

function stringField(record: JsonObject | undefined, key: string): string | undefined {
  const value = record?.[key];

  if (value instanceof Object) return undefined;
  const parsed = String(value);

  return parsed === value && parsed.length > 0 ? parsed : undefined;
}

function numberField(record: JsonObject | undefined, key: string): number | undefined {
  const value = record?.[key];
  const parsed = Number(value);

  return parsed === value && Number.isFinite(parsed) ? parsed : undefined;
}

function reportError(onError: ((cause: Error) => void) | undefined, cause: unknown): void {
  try {
    onError?.(cause instanceof Error ? cause : new Error(String(cause)));
  } catch {
    // telemetry must never fail an opencode tool call or turn.
  }
}

function textParts<T>(parts: T): string | undefined {
  if (!Array.isArray(parts)) return undefined;
  const text: string[] = [];

  for (const part of parts) {
    const record = asJsonObject(part);

    if (stringField(record, "type") === "text") {
      const value = stringField(record, "text");

      if (value !== undefined) text.push(value);
    }
  }

  return text.length > 0 ? text.join("\n") : undefined;
}

/** Named error so `error.type` reflects the host failure class instead of `Error`. */
function failureError(name: string, message: string | undefined): Error {
  const error = new Error(message ?? name);
  error.name = name;

  return error;
}

function usageFields(message: JsonObject): TokenUsage {
  const tokens = asJsonObject(message.tokens);
  const cache = asJsonObject(tokens?.cache);
  const input = numberField(tokens, "input") ?? 0;
  const output = numberField(tokens, "output") ?? 0;
  const reasoning = numberField(tokens, "reasoning") ?? 0;
  const cacheRead = numberField(cache, "read") ?? 0;
  const cacheWrite = numberField(cache, "write") ?? 0;

  return {
    inputTokens: input,
    outputTokens: output + reasoning,
    totalTokens:
      numberField(tokens, "total") ?? input + output + reasoning + cacheRead + cacheWrite,
    cacheReadInputTokens: cacheRead,
    cacheCreationInputTokens: cacheWrite,
    reasoningOutputTokens: reasoning,
  };
}

function messageError(message: JsonObject): Error | undefined {
  if (message.error === undefined || message.error === null) return undefined;
  const error = asJsonObject(message.error);
  const data = asJsonObject(error?.data);

  return failureError(
    stringField(error, "name") ?? "AssistantMessageError",
    stringField(data, "message"),
  );
}

/**
 * Creates an opencode plugin that exports agent turns, model generations, tool
 * executions, and session lifecycle logs to telemetry.dev.
 */
export function telemetryDevPlugin(
  options: TelemetryDevPluginOptions = {},
  overrides?: ClientOverrides,
): TelemetryDevPlugin {
  return async <Input, Options extends object>(_input: Input, configuredOptions?: Options) => {
    const mergedOptions = {
      ...options,
      ...configuredOptions,
    };

    const { agentName: configuredAgentName, ...sdkOptions } = mergedOptions;

    const agentName =
      String(configuredAgentName) === configuredAgentName && configuredAgentName.length > 0
        ? configuredAgentName
        : "opencode";

    const onError = sdkOptions.onError;

    try {
      ensureInit(sdkOptions, overrides);
    } catch (error) {
      reportError(onError, error);
    }

    const agentSpans = new Map<string, SpanHandle>();
    /** Open assistant-message chat spans, keyed by messageID. */
    const openChats = new Map<string, { span: SpanHandle; sessionID: string }>();
    /** The session's currently open chat span, so tool spans nest under it. */
    const chatBySession = new Map<string, SpanHandle>();
    /** Child (task/subagent) session -> parent session, from session.created. */
    const parentSession = new Map<string, string>();
    const pendingTools = new Map<`${string}:${string}`, PendingTool>();
    /** Synthetic Task part id -> the tool part's real LLM call id, keyed by session. */
    const toolCallByPart = new Map<string, Map<string, string>>();
    const completedMessages = new Set<string>();
    /** Latest session.error per session, applied when the session settles. */
    const pendingErrors = new Map<string, Error>();

    function conversationAttributes(sessionID: string | undefined): Record<string, string> {
      return sessionID ? { "gen_ai.conversation.id": sessionID } : {};
    }

    function emit(
      eventName: string,
      sessionID: string | undefined,
      level: LogLevel,
      message: string,
      attributes: Attrs = {},
    ): void {
      log(message, {
        level,
        eventName,
        attributes: {
          ...conversationAttributes(sessionID),
          "gen_ai.agent.name": agentName,
          ...attributes,
        },
      });
    }

    /** Closes a session's open chat spans when the session settles without completing them. */
    function endOpenChats(sessionID: string, error?: Error): void {
      for (const [messageID, open] of openChats) {
        if (open.sessionID !== sessionID) continue;
        open.span.end({ error, finishReason: error ? undefined : "incomplete" });
        openChats.delete(messageID);
      }

      chatBySession.delete(sessionID);
    }

    /**
     * Converts a terminated session's still-pending tools into error spans.
     * The missing-agent synthetic Task path throws after tool.execute.before,
     * so neither tool.execute.after nor an error-state tool part ever arrives.
     */
    function endPendingTools(sessionID: string, error: Error): void {
      for (const [key, pending] of pendingTools) {
        if (pending.sessionID !== sessionID) continue;
        pendingTools.delete(key);

        const span = startSpan(`execute_tool ${pending.tool}`, {
          type: "tool",
          parent: chatBySession.get(sessionID) ?? agentSpans.get(sessionID),
          startTime: pending.startedAt,
          toolName: pending.tool,
          toolCallId: pending.callID,
          input: pending.args,
          attributes: conversationAttributes(sessionID),
        });

        span.end({ error });
      }
    }

    /** Ends a session's spans when it settles (idle or terminal status). */
    function settleSession(sessionID: string | undefined): void {
      if (sessionID) {
        toolCallByPart.delete(sessionID);
        const error = pendingErrors.get(sessionID);
        pendingErrors.delete(sessionID);

        if (error) endPendingTools(sessionID, error);
        endOpenChats(sessionID, error);
        const span = agentSpans.get(sessionID);

        if (span) {
          span.end({ error });
          agentSpans.delete(sessionID);
        }
      }

      void flush().catch((error) => reportError(onError, error));
    }

    const hooks: TelemetryDevHooks = {
      "chat.message": async (input, output) => {
        try {
          if (agentSpans.has(input.sessionID)) return;
          const parentID = parentSession.get(input.sessionID);

          const parent = parentID
            ? (chatBySession.get(parentID) ?? agentSpans.get(parentID))
            : undefined;

          const attributes: Record<string, string> = conversationAttributes(input.sessionID);

          if (input.agent) attributes["opencode.agent"] = input.agent;
          agentSpans.set(
            input.sessionID,
            startSpan("invoke_agent", {
              type: "agent",
              parent,
              agentName,
              input: textParts(output.parts),
              attributes,
            }),
          );
        } catch (error) {
          reportError(onError, error);
        }
      },

      event: async ({ event }) => {
        try {
          const eventRecord = asJsonObject(event);
          const eventType = stringField(eventRecord, "type");
          const properties = asJsonObject(eventRecord?.properties);

          switch (eventType) {
            case "message.updated": {
              const info = asJsonObject(properties?.info);
              const time = asJsonObject(info?.time);
              const completed = numberField(time, "completed");
              const messageID = stringField(info, "id");
              const sessionID = stringField(info, "sessionID");
              const model = stringField(info, "modelID");

              if (
                stringField(info, "role") !== "assistant" ||
                messageID === undefined ||
                sessionID === undefined ||
                model === undefined ||
                completedMessages.has(messageID)
              ) {
                return;
              }

              // Open the chat span on first sight so tool spans can nest under it.
              let open = openChats.get(messageID);

              if (!open) {
                open = {
                  span: startSpan(`chat ${model}`, {
                    type: "generation",
                    parent: agentSpans.get(sessionID),
                    startTime: numberField(time, "created"),
                    model,
                    provider: stringField(info, "providerID"),
                    attributes: conversationAttributes(sessionID),
                  }),
                  sessionID,
                };
                openChats.set(messageID, open);
                chatBySession.set(sessionID, open.span);
              }

              if (completed === undefined) return;

              completedMessages.add(messageID);
              openChats.delete(messageID);

              if (chatBySession.get(sessionID) === open.span) chatBySession.delete(sessionID);
              open.span.end({
                endTime: completed,
                finishReason: stringField(info, "finish"),
                // Zero is a valid cost (free model/provider paths); never drop it.
                costUsd: numberField(info, "cost"),
                usage: usageFields(info ?? {}),
                error: messageError(info ?? {}),
              });

              return;
            }

            case "message.part.updated": {
              const part = asJsonObject(properties?.part);

              if (stringField(part, "type") !== "tool") return;
              const state = asJsonObject(part?.state);
              const status = stringField(state, "status");
              const sessionID = stringField(part, "sessionID");
              const callID = stringField(part, "callID");
              const partID = stringField(part, "id");
              const tool = stringField(part, "tool");

              if (!sessionID || !callID || !tool) return;

              if (status === "running") {
                if (!partID) return;
                const calls = toolCallByPart.get(sessionID) ?? new Map<string, string>();
                calls.set(partID, callID);
                toolCallByPart.set(sessionID, calls);

                return;
              }

              if (status !== "error") return;

              // The synthetic Task path keys its hook calls by part id, not by
              // the LLM callID carried on the part; check both.
              const callKey = `${sessionID}:${callID}` as const;
              const partKey = `${sessionID}:${partID}` as const;
              const pending = pendingTools.get(callKey) ?? pendingTools.get(partKey);

              if (!pending) return;
              pendingTools.delete(callKey);
              pendingTools.delete(partKey);

              if (partID) toolCallByPart.get(sessionID)?.delete(partID);
              const timing = asJsonObject(state?.time);

              const span = startSpan(`execute_tool ${tool}`, {
                type: "tool",
                parent: chatBySession.get(sessionID) ?? agentSpans.get(sessionID),
                startTime: pending.startedAt ?? numberField(timing, "start"),
                toolName: tool,
                toolCallId: callID,
                input: state?.input,
                attributes: conversationAttributes(sessionID),
              });

              span.end({
                endTime: numberField(timing, "end"),
                error: failureError(
                  "ToolExecutionError",
                  stringField(state, "error") ?? "tool execution failed",
                ),
              });

              return;
            }

            case "session.idle": {
              settleSession(stringField(properties, "sessionID"));

              return;
            }

            case "session.status": {
              // Modern settle signal published before the deprecated
              // session.idle event; settleSession is idempotent, so handling
              // both is safe.
              if (stringField(asJsonObject(properties?.status), "type") === "idle") {
                settleSession(stringField(properties, "sessionID"));
              }

              return;
            }

            case "session.error": {
              const sessionID = stringField(properties, "sessionID");
              const hostError = asJsonObject(properties?.error);
              const errorData = asJsonObject(hostError?.data);
              const errorName = stringField(hostError, "name") ?? "SessionError";
              const errorMessage = stringField(errorData, "message");
              const attributes = { "opencode.error.name": errorName };

              if (errorMessage)
                Object.assign(attributes, { "opencode.error.message": errorMessage });
              emit("session.error", sessionID, "error", "Session error", attributes);

              // Not always terminal: a context overflow publishes this event,
              // then compacts and continues the same prompt. Record the error
              // and apply it when the session settles.
              if (sessionID) pendingErrors.set(sessionID, failureError(errorName, errorMessage));

              return;
            }

            case "session.created": {
              const info = asJsonObject(properties?.info);
              const sessionID = stringField(info, "id");
              const parentID = stringField(info, "parentID");
              const attributes: Attrs = {};

              if (parentID) {
                attributes["opencode.session.parent_id"] = parentID;

                if (sessionID) parentSession.set(sessionID, parentID);
              }

              emit("session.created", sessionID, "info", "Session created", attributes);

              return;
            }

            case "session.compacted": {
              const sessionID = stringField(properties, "sessionID");

              // Compaction after a context-overflow session.error means the
              // host recovered and the prompt continues; drop the stale error.
              if (sessionID) pendingErrors.delete(sessionID);
              emit("session.compacted", sessionID, "info", "Session compacted");

              return;
            }
          }
        } catch (error) {
          reportError(onError, error);
        }
      },

      "tool.execute.before": async (input, output) => {
        try {
          pendingTools.set(`${input.sessionID}:${input.callID}`, {
            sessionID: input.sessionID,
            callID: toolCallByPart.get(input.sessionID)?.get(input.callID) ?? input.callID,
            tool: input.tool,
            args: output.args,
            startedAt: Date.now(),
          });
        } catch (error) {
          reportError(onError, error);
        }
      },

      "tool.execute.after": async (input, output) => {
        try {
          // opencode's synthetic Task path invokes this hook with no output
          // when execution fails; keep the pending state so the error-state
          // tool part event can still reconstruct the span.
          if (!output) return;
          const key = `${input.sessionID}:${input.callID}` as const;
          const pending = pendingTools.get(key);
          pendingTools.delete(key);
          const mappedCallID = toolCallByPart.get(input.sessionID)?.get(input.callID);
          toolCallByPart.get(input.sessionID)?.delete(input.callID);

          const span = startSpan(`execute_tool ${input.tool}`, {
            type: "tool",
            parent: chatBySession.get(input.sessionID) ?? agentSpans.get(input.sessionID),
            startTime: pending?.startedAt ?? Date.now(),
            toolName: input.tool,
            toolCallId: pending?.callID ?? mappedCallID ?? input.callID,
            input: input.args ?? pending?.args,
            output: {
              title: output.title,
              output: output.output,
              metadata: output.metadata,
            },
            attributes: conversationAttributes(input.sessionID),
          });

          span.end();
        } catch (error) {
          reportError(onError, error);
        }
      },

      dispose: async () => {
        try {
          // Disposal is not gated on session idle; apply each session's
          // deferred session.error instead of ending spans as merely incomplete.
          for (const [sessionID, error] of pendingErrors) endPendingTools(sessionID, error);

          for (const [messageID, open] of openChats) {
            const error = pendingErrors.get(open.sessionID);
            open.span.end({ error, finishReason: error ? undefined : "incomplete" });
            openChats.delete(messageID);
          }

          chatBySession.clear();

          for (const [sessionID, span] of agentSpans) {
            const error = pendingErrors.get(sessionID);
            span.end({ error, finishReason: error ? undefined : "incomplete" });
          }

          agentSpans.clear();
          parentSession.clear();
          pendingTools.clear();
          toolCallByPart.clear();
          pendingErrors.clear();
          completedMessages.clear();
          await flush();
        } catch (error) {
          reportError(onError, error);
        }
      },
    };

    return hooks;
  };
}
