import { flush, log, startSpan, type LogLevel, type SpanHandle } from "@telemetry-dev/sdk";

import { ensureInit, type ClientOverrides, type TelemetryDevCursorOptions } from "./config.ts";
import { createTitleLookup } from "./titles.ts";

type JsonValue = string | number | boolean | null | undefined | Date | JsonValue[] | JsonRecord;

interface JsonRecord {
  [key: string]: JsonValue;
}

/** Hook events forwarded to the daemon; see `install.ts` for the hooks.json wiring. */
export const HOOK_EVENTS = [
  "sessionStart",
  "sessionEnd",
  "beforeSubmitPrompt",
  "preToolUse",
  "postToolUse",
  "postToolUseFailure",
  "subagentStart",
  "subagentStop",
  "afterFileEdit",
  "afterAgentResponse",
  "afterAgentThought",
  "preCompact",
  "stop",
] as const;

export interface CursorTelemetry {
  /** Maps one Cursor hook payload to spans/logs. Never throws. */
  handle(event: Record<string, unknown>): void;
  /** True while a turn span is open. */
  open(): boolean;
  /** Ends open turns as incomplete and flushes exporters. */
  settle(): Promise<void>;
}

interface Turn {
  span: SpanHandle;
  /** End of the previous assistant message; start of the next chat span. */
  lastBoundary: number;
  /** Time of the conversation's newest hook event; end time for retroactive closes. */
  lastEvent: number;
  /** Turn name when the chat has no title: the prompt or the subagent task. */
  fallbackName?: string;
  /** Session this turn belongs to, so sessionEnd can close it without a conversation_id. */
  sessionId?: string;
  /** Generation ids seen during this turn; a stop for a closed turn's generation is stale. */
  generations: Set<string>;
  /** These generations belong to the turns that beforeSubmitPrompt replaced. */
  priorGenerations?: Set<string>;
}

interface PendingTool {
  startedAt: number;
  input: JsonValue;
}

/**
 * Keys a subagent's own conversation into its parent turn. Registered when a
 * new conversation claims a pending Task tool call (the CLI emits no
 * subagentStart), or by subagentStart on hosts that send it.
 */
interface SubagentLink {
  parent: string;
  type?: string;
  model?: string;
  task?: string;
  /** True once the subagent's turn span has ended, so subagentStop must not re-emit it. */
  closed: boolean;
}

/**
 * A parent's in-flight Task tool call, awaiting the subagent conversation it
 * spawned. Hook payloads carry no field linking a subagent conversation to its
 * parent, so the next unknown conversation to send events claims the oldest
 * unclaimed wait.
 */
interface TaskWait {
  parent: string;
  callId?: string;
  type?: string;
  task?: string;
  claimedBy?: string;
}

function asRecord<T>(value: T): JsonRecord | undefined {
  return value !== null && !(value instanceof Function) && Object(value) === value
    ? (value as JsonRecord)
    : undefined;
}

function readString<T>(value: T): string | undefined {
  const raw: unknown = value;

  return String(raw) === raw ? raw : undefined;
}

function str(record: JsonRecord, key: string): string | undefined {
  return readString(record[key]);
}

function num(record: JsonRecord, key: string): number | undefined {
  const raw = record[key];
  const value = Number(raw);

  return value === raw && Number.isFinite(value) ? value : undefined;
}

function asError<T>(cause: T): Error {
  return cause instanceof Error ? cause : new Error(String(cause));
}

/** Named error so `error.type` reflects the host failure class instead of `Error`. */
function failureError(name: string, message: string | undefined): Error {
  const error = new Error(message ?? name);
  error.name = name;

  return error;
}

/** First line of the prompt or task, shortened to a span-name-sized label. */
function shortName(text: string | undefined): string | undefined {
  const line = text?.trim().split("\n", 1)[0];

  if (!line) return undefined;

  return line.length > 60 ? `${line.slice(0, 59)}\u2026` : line;
}

/** `tool_output` arrives JSON-stringified; keep the string when it is not JSON. */
function parseMaybeJson(value: string | undefined): JsonValue {
  if (value === undefined) return undefined;

  try {
    return JSON.parse(value) as JsonValue;
  } catch {
    return value;
  }
}

/**
 * Creates the Cursor hook event mapper. Each user turn becomes an
 * `invoke_agent` trace with `chat`, `thought`, and `execute_tool` child spans;
 * subagent conversations nest as `invoke_agent` child spans inside the parent
 * turn's trace; lifecycle events become logs. Every record carries
 * `gen_ai.conversation.id`, and turn spans take the Cursor chat title as their
 * name when one exists.
 */
export function createCursorTelemetry(
  options: TelemetryDevCursorOptions = {},
  overrides?: ClientOverrides,
): CursorTelemetry {
  const onError = options.onError;

  try {
    ensureInit(options, overrides);
  } catch (error) {
    onError?.(asError(error));
  }

  const turns = new Map<string, Turn>();
  const pendingTools = new Map<string, PendingTool>();
  const subagents = new Map<string, SubagentLink>();
  const taskWaits: TaskWait[] = [];
  const sessions = new Set<string>();
  const titleFor = createTitleLookup(options.chatsDir);

  function baseAttributes(event: JsonRecord) {
    const attributes: Record<string, string> = {};
    const conversationId = str(event, "conversation_id");

    if (conversationId) attributes["gen_ai.conversation.id"] = conversationId;
    const generationId = str(event, "generation_id");

    if (generationId) attributes["cursor.generation_id"] = generationId;
    const email = str(event, "user_email");

    if (email) attributes["cursor.user_email"] = email;
    const roots = event.workspace_roots;
    const workspace = Array.isArray(roots) ? readString(roots[0]) : undefined;

    if (workspace !== undefined) attributes["cursor.workspace"] = workspace;

    return attributes;
  }

  function emit(
    event: JsonRecord,
    level: LogLevel,
    message: string,
    attributes: JsonRecord = {},
  ): void {
    log(message, {
      level,
      eventName: str(event, "hook_event_name"),
      attributes: { "gen_ai.agent.name": "cursor", ...baseAttributes(event), ...attributes },
    });
  }

  function endTurn(conversationId: string, fields: Parameters<SpanHandle["end"]>[0]): void {
    const turn = turns.get(conversationId);

    if (!turn) return;
    turns.delete(conversationId);

    // Purge per-turn state so a later event or conversation cannot pick up
    // stale tool timings or claim a Task wait from this ended turn.
    for (const key of pendingTools.keys()) {
      if (key.startsWith(`${conversationId}:`)) pendingTools.delete(key);
    }

    for (let i = taskWaits.length - 1; i >= 0; i--) {
      if (taskWaits[i]!.parent === conversationId) taskWaits.splice(i, 1);
    }

    const link = subagents.get(conversationId);

    if (link) link.closed = true;

    // A parent that ends leaves no close signal for its open subagent turns;
    // end them at their own last activity.
    for (const [childId, childLink] of subagents) {
      if (childLink.parent !== conversationId || !turns.has(childId)) continue;
      const child = turns.get(childId);
      endTurn(childId, { finishReason: "incomplete", endTime: child?.lastEvent });
    }

    turn.span.end({ name: titleFor(conversationId) ?? turn.fallbackName, ...fields });
  }

  /**
   * Returns the conversation's open turn, starting one when needed. The CLI
   * does not always fire beforeSubmitPrompt, so any event can open a turn.
   * Only beforeSubmitPrompt, stop, and sessionEnd close it: generation_id
   * advances within one turn (per model call in the CLI), so it must not
   * rotate turns.
   *
   * Subagent conversations open as child spans of the parent turn instead of
   * new root traces. A conversation counts as a subagent when subagentStart
   * registered it, or when its conversation and generation ids match while a
   * Task tool call is pending. The latter is the identity Cursor uses for
   * headless subagent conversations, which send no subagentStart event.
   */
  function ensureTurn(
    event: JsonRecord,
    input?: JsonValue,
    userPrompt = false,
    priorGenerations?: Set<string>,
  ): Turn | undefined {
    const conversationId = str(event, "conversation_id");

    if (!conversationId) return undefined;
    const existing = turns.get(conversationId);

    if (existing) return existing;
    let link = subagents.get(conversationId);

    if (
      !link &&
      !userPrompt &&
      !sessions.has(conversationId) &&
      str(event, "generation_id") === conversationId
    ) {
      const wait = taskWaits.find((w) => !w.claimedBy && w.parent !== conversationId);

      if (wait) {
        wait.claimedBy = conversationId;
        link = { parent: wait.parent, type: wait.type, task: wait.task, closed: false };
        subagents.set(conversationId, link);
      }
    }

    const agentName = link ? (link.type ?? "subagent") : "cursor";
    const promptOrTask = readString(input) ?? link?.task;

    const turn: Turn = {
      span: startSpan(`invoke_agent ${agentName}`, {
        type: "agent",
        parent: link ? turns.get(link.parent)?.span : undefined,
        agentName,
        model: str(event, "model_id") ?? str(event, "model") ?? link?.model,
        input: input ?? link?.task,
        attributes: baseAttributes(event),
      }),
      lastBoundary: Date.now(),
      lastEvent: Date.now(),
      fallbackName: shortName(promptOrTask),
      sessionId: str(event, "session_id"),
      generations: new Set(str(event, "generation_id") ? [str(event, "generation_id")!] : []),
      priorGenerations,
    };

    turns.set(conversationId, turn);

    return turn;
  }

  /** Matches a subagentStop payload to its registered conversation. */
  function resolveSubagent(event: JsonRecord): string | undefined {
    // The transcript path embeds the subagent's own conversation id and is
    // checked first: a nested child's stop can be delivered in its subagent
    // parent's context, where conversation_id names the parent, not the child.
    const transcript = str(event, "agent_transcript_path");

    if (transcript) {
      const parts = transcript.split(/[\\/]+/);

      for (const id of subagents.keys()) {
        if (parts.includes(id)) return id;
      }
    }

    const conversationId = str(event, "conversation_id");

    if (conversationId && subagents.has(conversationId)) return conversationId;

    return undefined;
  }

  function handleEvent(event: JsonRecord): void {
    const name = str(event, "hook_event_name");
    // A terminal event without a conversation_id still closes the sole open
    // turn, so a harness that omits the field cannot leak an open span.
    const soleTurn = turns.size === 1 ? [...turns.keys()][0] : undefined;

    const conversationId =
      str(event, "conversation_id") ??
      (name === "stop" || name === "sessionEnd" ? soleTurn : undefined);

    const now = Date.now();
    const generationId = str(event, "generation_id");
    // A stop can arrive late, after beforeSubmitPrompt already closed its turn
    // and opened the next one; matching it to the closed turn's generations
    // keeps it from prematurely closing the fresh turn.
    let staleStop = false;

    if (conversationId) {
      const active = turns.get(conversationId);

      if (active) {
        staleStop =
          name === "stop" &&
          generationId !== undefined &&
          !active.generations.has(generationId) &&
          active.priorGenerations?.has(generationId) === true;

        if (!staleStop) {
          active.lastEvent = now;

          if (generationId) active.generations.add(generationId);
          const sessionId = str(event, "session_id");

          if (sessionId) active.sessionId = sessionId;
        }
      }
    }

    switch (name) {
      case "sessionStart": {
        const sessionId = str(event, "session_id");

        if (sessionId) sessions.add(sessionId);
        emit(event, "info", "Session started", {
          "cursor.composer_mode": str(event, "composer_mode"),
          "cursor.is_background_agent": event.is_background_agent === true,
        });

        return;
      }

      case "sessionEnd": {
        const reason = str(event, "reason");
        emit(event, reason === "error" ? "error" : "info", "Session ended", {
          "cursor.session_end_reason": reason,
          "cursor.error.message": str(event, "error_message"),
        });
        const sessionId = str(event, "session_id");

        if (sessionId) sessions.delete(sessionId);

        if (conversationId) sessions.delete(conversationId);
        const targets = new Set<string>();

        if (conversationId) targets.add(conversationId);

        if (sessionId) {
          for (const [id, turn] of turns) {
            if (turn.sessionId === sessionId) targets.add(id);
          }
        }

        for (const id of targets) {
          endTurn(id, {
            error:
              reason === "error"
                ? failureError("SessionError", str(event, "error_message"))
                : undefined,
            finishReason: reason === "error" ? undefined : (reason ?? "incomplete"),
          });
        }

        void flush().catch((error) => onError?.(asError(error)));

        return;
      }

      case "beforeSubmitPrompt": {
        let priorGenerations: Set<string> | undefined;

        if (conversationId) {
          const prior = turns.get(conversationId);

          if (prior) {
            priorGenerations = new Set(prior.priorGenerations);

            for (const generation of prior.generations) {
              priorGenerations.add(generation);
            }
          }

          endTurn(conversationId, { finishReason: "incomplete" });
        }

        ensureTurn(event, str(event, "prompt"), true, priorGenerations);

        return;
      }

      case "preToolUse": {
        ensureTurn(event);
        const callId = str(event, "tool_use_id");

        if (callId && conversationId) {
          pendingTools.set(`${conversationId}:${callId}`, {
            startedAt: now,
            input: event.tool_input,
          });
        }

        // A Task call spawns a subagent conversation; the next unknown
        // conversation claims this wait (see ensureTurn).
        if (conversationId && str(event, "tool_name") === "Task") {
          const record = asRecord(event.tool_input) ?? {};
          taskWaits.push({
            parent: conversationId,
            callId,
            type: str(record, "subagent_type"),
            task: str(record, "prompt") ?? str(record, "description"),
          });
        }

        return;
      }

      case "postToolUse":
      case "postToolUseFailure": {
        const turn = ensureTurn(event);
        const tool = str(event, "tool_name") ?? "unknown";
        const callId = str(event, "tool_use_id");
        const pending = callId ? pendingTools.get(`${conversationId}:${callId}`) : undefined;

        if (callId) pendingTools.delete(`${conversationId}:${callId}`);
        const duration = num(event, "duration");

        const span = startSpan(`execute_tool ${tool}`, {
          type: "tool",
          parent: turn?.span,
          startTime: pending?.startedAt ?? (duration === undefined ? now : now - duration),
          agentName: "cursor",
          toolName: tool,
          toolCallId: callId,
          input: event.tool_input ?? pending?.input,
          output: parseMaybeJson(str(event, "tool_output")),
          attributes: baseAttributes(event),
        });

        span.end({
          error:
            name === "postToolUseFailure"
              ? failureError(
                  str(event, "failure_type") ?? "ToolFailure",
                  str(event, "error_message"),
                )
              : undefined,
        });

        // The Task tool result closes the subagent conversations it spawned.
        // Parallel workers share one tool_use_id, so one result can close
        // several claimed waits.
        if (str(event, "tool_name") === "Task") {
          const matches = taskWaits.filter(
            (w) => w.parent === conversationId && (callId ? w.callId === callId : true),
          );

          for (const wait of matches) {
            const index = taskWaits.indexOf(wait);

            if (index >= 0) taskWaits.splice(index, 1);
            const child = wait.claimedBy;

            if (!child || !turns.has(child)) continue;
            const failed = name === "postToolUseFailure";
            const childTurn = turns.get(child);
            childTurn?.span.update({ output: parseMaybeJson(str(event, "tool_output")) });
            endTurn(child, {
              error: failed
                ? failureError(
                    str(event, "failure_type") ?? "SubagentError",
                    str(event, "error_message"),
                  )
                : undefined,
              finishReason: failed ? undefined : "completed",
              endTime: childTurn?.lastEvent,
            });
          }
        }

        return;
      }

      case "subagentStart": {
        const parent = str(event, "parent_conversation_id") ?? conversationId;
        const subagentId = str(event, "subagent_id");

        if (parent) {
          const link: SubagentLink = {
            parent,
            type: str(event, "subagent_type"),
            model: str(event, "subagent_model"),
            task: str(event, "task"),
            closed: false,
          };

          if (subagentId) subagents.set(subagentId, link);
          // When the Task preToolUse already registered a wait, enrich it so a
          // conversation claiming it inherits the subagent type and model.
          const toolCallId = str(event, "tool_call_id");

          const wait = taskWaits.find(
            (w) => w.parent === parent && (toolCallId ? w.callId === toolCallId : !w.claimedBy),
          );

          if (wait) {
            wait.type ??= link.type;
            wait.task ??= link.task;
          }

          // Some payloads deliver the subagent's own conversation_id here.
          if (conversationId && conversationId !== parent) subagents.set(conversationId, link);
          // Open the parent turn so the subagent's spans have a trace to nest in.
          ensureTurn({ ...event, conversation_id: parent });
        }

        emit(event, "info", "Subagent started", {
          "cursor.subagent_id": subagentId,
          "cursor.subagent_type": str(event, "subagent_type"),
          "cursor.subagent_model": str(event, "subagent_model"),
        });

        return;
      }

      case "subagentStop": {
        const status = str(event, "status");

        const error =
          status === "error" ? failureError("SubagentError", str(event, "task")) : undefined;

        const subagentId = resolveSubagent(event);
        const link = subagentId ? subagents.get(subagentId) : undefined;

        if (subagentId && link) {
          const child = turns.get(subagentId);

          if (child) {
            child.span.update({
              output: str(event, "summary"),
              attributes: { "cursor.subagent_status": status ?? "unknown" },
            });
            endTurn(subagentId, { error, finishReason: status === "error" ? undefined : status });
          }

          for (const [id, other] of subagents) {
            if (other === link) subagents.delete(id);
          }

          // The subagent's own events already produced its turn span; only a
          // link that never opened (and never closed) a turn needs the
          // synthetic span below.
          if (child || link.closed) return;
        }

        const turn = link ? turns.get(link.parent) : ensureTurn(event);
        const duration = num(event, "duration_ms") ?? 0;

        const span = startSpan(`invoke_agent ${str(event, "subagent_type") ?? "subagent"}`, {
          type: "agent",
          parent: turn?.span,
          startTime: now - duration,
          agentName: str(event, "subagent_type") ?? "subagent",
          input: str(event, "task"),
          output: str(event, "summary"),
          attributes: {
            ...baseAttributes(event),
            "cursor.subagent_status": status ?? "unknown",
          },
        });

        span.end({ error, finishReason: status === "error" ? undefined : status });

        return;
      }

      case "afterFileEdit": {
        ensureTurn(event);
        const edits = event.edits;
        emit(event, "info", "File edited", {
          "cursor.file_path": str(event, "file_path"),
          "cursor.edit_count": Array.isArray(edits) ? edits.length : 0,
        });

        return;
      }

      case "afterAgentThought": {
        const turn = ensureTurn(event);
        const duration = num(event, "duration_ms") ?? 0;
        startSpan("thought", {
          type: "span",
          parent: turn?.span,
          startTime: now - duration,
          output: str(event, "text"),
          attributes: baseAttributes(event),
        }).end();

        return;
      }

      case "afterAgentResponse": {
        const turn = ensureTurn(event);
        const model = str(event, "model_id") ?? str(event, "model");

        const span = startSpan(`chat ${model ?? "unknown"}`, {
          type: "generation",
          parent: turn?.span,
          startTime: turn?.lastBoundary,
          agentName: "cursor",
          model,
          output: str(event, "text"),
          attributes: baseAttributes(event),
        });

        span.end();

        if (turn) {
          turn.lastBoundary = now;
          turn.span.update({ output: str(event, "text") });
        }

        return;
      }

      case "preCompact": {
        emit(event, "info", "Context compacted", {
          "cursor.compaction_trigger": str(event, "trigger"),
          "cursor.context_usage_percent": num(event, "context_usage_percent"),
          "cursor.context_tokens": num(event, "context_tokens"),
          "cursor.messages_to_compact": num(event, "messages_to_compact"),
        });

        return;
      }

      case "stop": {
        if (staleStop) return;
        const status = str(event, "status");

        if (conversationId) {
          endTurn(conversationId, {
            error: status === "error" ? failureError("AgentError", "agent loop failed") : undefined,
            finishReason: status === "error" ? undefined : status,
          });
        }

        void flush().catch((error) => onError?.(asError(error)));

        return;
      }
    }
  }

  return {
    handle(event) {
      try {
        handleEvent(asRecord(event) ?? {});
      } catch (error) {
        onError?.(asError(error));
      }
    },
    open() {
      return turns.size > 0;
    },
    async settle() {
      try {
        for (const [conversationId, turn] of turns) {
          endTurn(conversationId, { finishReason: "incomplete", endTime: turn.lastEvent });
        }

        pendingTools.clear();
        subagents.clear();
        taskWaits.length = 0;
        sessions.clear();
        await flush();
      } catch (error) {
        onError?.(asError(error));
      }
    },
  };
}
