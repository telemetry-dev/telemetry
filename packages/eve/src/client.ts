import type { Attributes } from "@opentelemetry/api";
import {
  startSpan,
  type ClientOverrides,
  type SpanHandle,
  type TokenUsage,
} from "@telemetry-dev/sdk";
import type {
  Client,
  ClientSession,
  ClientSessions,
  InputResponse,
  MessageStreamEvent,
  SendTurnInput,
  SendTurnOptions,
} from "eve/client";
import { isCurrentTurnBoundaryEvent, MessageResponse } from "eve/client";

import { ensureInit, type TelemetryDevEveOptions } from "./config.ts";

export interface WrapEveClientOptions extends TelemetryDevEveOptions {
  /** gen_ai.agent.name on turn spans. */
  agentName?: string;
  /** Span name for each send() turn. Default "invoke_agent". */
  spanName?: string;
}

type EventRecord = { [key: string]: EventValue };
type EventValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | readonly EventValue[]
  | EventRecord;
function asRecord(value: EventValue): EventRecord | undefined {
  if (value === null || Array.isArray(value) || !(value instanceof Object)) return undefined;
  return value as EventRecord;
}

function stringField(record: EventRecord | undefined, key: string): string | undefined {
  const value = record?.[key];
  return value?.constructor === String && value.length > 0 ? value : undefined;
}

function numberField(record: EventRecord | undefined, key: string): number | undefined {
  const value = record?.[key];
  return value?.constructor === Number ? value : undefined;
}

function eventData(event: MessageStreamEvent): EventRecord {
  if (!("data" in event)) return {};
  return asRecord(event.data as EventValue) ?? {};
}

function eventAttributes(
  attributes: Record<string, string | number | boolean | undefined>,
): Attributes {
  const out: Attributes = {};
  for (const [key, value] of Object.entries(attributes)) {
    if (value !== undefined) out[key] = value;
  }
  return out;
}

function addUsage(usage: TokenUsage, key: keyof TokenUsage, value: number | undefined): boolean {
  if (value === undefined) return false;
  usage[key] = (usage[key] ?? 0) + value;
  return true;
}

function failureError(code: string | undefined, message: string | undefined): Error {
  const error = new Error(message ?? code ?? "error");
  if (code) error.name = code;
  return error;
}

function bindOrReturn<T extends object>(target: T, prop: string | symbol) {
  const value = target[prop as keyof T];
  return value instanceof Function ? value.bind(target) : value;
}

function addLifecycleEvent(span: SpanHandle, event: MessageStreamEvent): void {
  const data = eventData(event);
  switch (event.type) {
    case "subagent.called": {
      const remote = asRecord(data.remote);
      span.span.addEvent(
        event.type,
        eventAttributes({
          "eve.subagent.name": stringField(data, "name"),
          "gen_ai.tool.name": stringField(data, "toolName"),
          "gen_ai.tool.call.id": stringField(data, "callId"),
          "eve.child.session_id": stringField(data, "childSessionId"),
          "eve.workflow.id": stringField(data, "workflowId"),
          "eve.remote.url": stringField(remote, "url"),
        }),
      );
      return;
    }
    case "input.requested": {
      const requests = Array.isArray(data.requests) ? data.requests : undefined;
      span.span.addEvent(
        event.type,
        eventAttributes({ "eve.input.request_count": requests?.length }),
      );
      return;
    }
    case "authorization.required":
      span.span.addEvent(
        event.type,
        eventAttributes({ "eve.authorization.name": stringField(data, "name") }),
      );
      return;
    default:
      return;
  }
}

async function* instrumentedStream<TOutput>(
  response: MessageResponse<TOutput>,
  span: SpanHandle,
  start: number,
  signal: AbortSignal | undefined,
  endCancelled: (() => void) | undefined,
): AsyncGenerator<MessageStreamEvent> {
  const usage: TokenUsage = {};
  let sawUsage = false;
  let output: unknown;
  let finishReason: string | undefined;
  let costUsd: number | undefined;
  let allCostsKnown = true;
  let timeToFirstChunkMs: number | undefined;
  let error: unknown;
  let done = false;
  let sawTerminalEvent = false;

  const finish = (): void => {
    if (done) return;
    done = true;
    if (!sawTerminalEvent && error === undefined) {
      if (signal?.aborted) {
        error = failureError("cancelled", "cancelled");
        finishReason ??= "cancelled";
      } else {
        error = failureError("stream_incomplete", "stream ended before a terminal event");
        finishReason ??= "error";
      }
    }
    span.end({
      usage: sawUsage ? usage : undefined,
      finishReason,
      costUsd:
        sawTerminalEvent && error === undefined && allCostsKnown && Number.isFinite(costUsd)
          ? costUsd
          : undefined,
      output,
      timeToFirstChunkMs,
      error: error instanceof Error ? error : error === undefined ? undefined : new Error("error"),
    });
  };

  try {
    for await (const event of response) {
      const data = eventData(event);
      if (event.type === "turn.started") {
        const turnTrace = asRecord(data.trace);
        const traceId = stringField(turnTrace, "traceId");
        const spanId = stringField(turnTrace, "spanId");
        const turnId = stringField(data, "turnId");
        if (traceId && spanId && turnId) {
          span.span.setAttribute("td.eve.turn_root", `${traceId}/${spanId}`);
          span.span.setAttribute("eve.turn.id", turnId);
        }
      }
      if (isCurrentTurnBoundaryEvent(event)) sawTerminalEvent = true;
      if (
        timeToFirstChunkMs === undefined &&
        (event.type === "message.appended" || event.type === "reasoning.appended")
      ) {
        timeToFirstChunkMs = performance.now() - start;
      }

      if (event.type === "step.completed") {
        const eventUsage = asRecord(data.usage);
        sawUsage =
          addUsage(usage, "inputTokens", numberField(eventUsage, "inputTokens")) || sawUsage;
        sawUsage =
          addUsage(usage, "outputTokens", numberField(eventUsage, "outputTokens")) || sawUsage;
        sawUsage =
          addUsage(usage, "cacheReadInputTokens", numberField(eventUsage, "cacheReadTokens")) ||
          sawUsage;
        sawUsage =
          addUsage(
            usage,
            "cacheCreationInputTokens",
            numberField(eventUsage, "cacheWriteTokens"),
          ) || sawUsage;
        const stepCost = numberField(eventUsage, "costUsd");
        if (stepCost === undefined || !Number.isFinite(stepCost) || stepCost < 0) {
          allCostsKnown = false;
        } else {
          costUsd = (costUsd ?? 0) + stepCost;
        }
        finishReason = stringField(data, "finishReason") ?? finishReason;
      }

      if (
        event.type === "message.completed" &&
        stringField(data, "finishReason") !== "tool-calls"
      ) {
        output = data.message;
      }
      if (event.type === "result.completed") {
        output = data.result;
      }
      if (
        event.type === "turn.failed" ||
        event.type === "session.failed" ||
        event.type === "step.failed"
      ) {
        if (error === undefined) {
          const code = stringField(data, "code");
          error = failureError(code, stringField(data, "message"));
          if (code) span.span.setAttribute("error.code", code);
        }
        finishReason = "error";
      }
      if (event.type === "turn.cancelled") {
        error = failureError("cancelled", "cancelled");
        finishReason = "cancelled";
      }

      addLifecycleEvent(span, event);
      yield event;
    }
  } catch (caught) {
    if (signal?.aborted) {
      error = failureError("cancelled", "cancelled");
      finishReason ??= "cancelled";
    } else {
      error = caught instanceof Error ? caught : new Error(String(caught));
    }
    throw caught;
  } finally {
    finish();
    if (endCancelled) signal?.removeEventListener("abort", endCancelled);
  }
}

async function tracedTurn<TOutput>(
  options: WrapEveClientOptions,
  input: SendTurnInput["message"] | readonly InputResponse[],
  turnOptions: SendTurnOptions<TOutput> | undefined,
  knownSessionId: string | undefined,
  send: () => Promise<MessageResponse<TOutput>>,
): Promise<MessageResponse<TOutput>> {
  // The session id is only known after the POST (first turn), so the span starts once the
  // request resolves with the original start time; that lets it join the session trace.
  const startTime = new Date();
  const start = performance.now();
  const signal = turnOptions?.signal;
  const spanFor = (sessionId: string | undefined) =>
    startSpan(options.spanName ?? "invoke_agent", {
      type: "agent",
      agentName: options.agentName,
      input,
      startTime,
      attributes: sessionId === undefined ? undefined : { "gen_ai.conversation.id": sessionId },
    });
  let response: MessageResponse<TOutput>;
  try {
    response = await send();
  } catch (error) {
    const span = spanFor(knownSessionId);
    if (signal?.aborted) {
      span.end({ error: failureError("cancelled", "cancelled"), finishReason: "cancelled" });
    } else {
      span.end({ error: error instanceof Error ? error : new Error(String(error)) });
    }
    throw error;
  }
  const span = spanFor(response.sessionId);
  let streamStarted = false;
  let endedBeforeStream = false;
  const endCancelled = () => {
    if (streamStarted) return;
    endedBeforeStream = true;
    span.end({ error: failureError("cancelled", "cancelled"), finishReason: "cancelled" });
  };
  if (signal?.aborted) {
    endCancelled();
  } else {
    signal?.addEventListener("abort", endCancelled, { once: true });
  }
  // MessageResponse's constructor is @internal in eve; re-wrapping the stream has no
  // public alternative. cancel() forwards to the original response, whose turn id
  // resolves because instrumentedStream consumes it.
  return new MessageResponse<TOutput>({
    cancelTurn: () => response.cancel(),
    createStream: () => {
      streamStarted = true;
      // The span already ended on abort; hand back the raw stream instead of ending it twice.
      if (endedBeforeStream)
        return (async function* () {
          yield* response;
        })();
      return instrumentedStream(response, span, start, signal, endCancelled);
    },
    sessionId: response.sessionId,
  });
}

function wrapSession(session: ClientSession, options: WrapEveClientOptions): ClientSession {
  return new Proxy(session, {
    get(target, prop) {
      if (prop === "send") {
        const send: ClientSession["send"] = (message, turnOptions) =>
          tracedTurn(options, message, turnOptions, target.state.sessionId, () =>
            target.send(message, turnOptions),
          );
        return send;
      }
      if (prop === "respond") {
        const respond: ClientSession["respond"] = (inputResponses, turnOptions) =>
          tracedTurn(options, inputResponses, turnOptions, target.state.sessionId, () =>
            target.respond(inputResponses, turnOptions),
          );
        return respond;
      }
      return bindOrReturn(target, prop);
    },
  });
}

function wrapSessions(sessions: ClientSessions, options: WrapEveClientOptions): ClientSessions {
  return new Proxy(sessions, {
    get(target, prop) {
      if (prop === "create") {
        const create: ClientSessions["create"] = async <TOutput = unknown>(
          input: SendTurnInput<TOutput>,
        ) => {
          let session: ClientSession | undefined;
          const response = await tracedTurn(options, input.message, input, undefined, async () => {
            const created = await target.create(input);
            session = created.session;
            return created.response;
          });
          return { response, session: wrapSession(session!, options) };
        };
        return create;
      }
      if (prop === "attach") {
        const attach: ClientSessions["attach"] = (sessionId, attachOptions) =>
          wrapSession(target.attach(sessionId, attachOptions), options);
        return attach;
      }
      return bindOrReturn(target, prop);
    },
  });
}

export function wrapEveClient<C extends Client>(
  client: C,
  options: WrapEveClientOptions = {},
  overrides?: ClientOverrides,
): C {
  const { agentName: _agentName, spanName: _spanName, ...sdkOptions } = options;
  ensureInit(sdkOptions, overrides);
  const sessions = wrapSessions(client.sessions, options);
  return new Proxy(client, {
    get(target, prop) {
      if (prop === "sessions") return sessions;
      return bindOrReturn(target, prop);
    },
  }) as C;
}
