import { log, type LogLevel } from "@telemetry-dev/sdk";
import type { MessageStreamEvent } from "eve/client";
import type { HookContext, HookDefinition } from "eve/hooks";

import { ensureInit, type ClientOverrides, type TelemetryDevEveOptions } from "./config.ts";

type Attrs = { [key: string]: Value };

type Value = string | number | boolean | null | undefined | readonly Value[] | Attrs;

function asRecord(value: Value): Attrs | undefined {
  if (value === null || Array.isArray(value) || !(value instanceof Object)) return undefined;

  return value as Attrs;
}

function eventData(event: MessageStreamEvent): Attrs {
  if (!("data" in event)) return {};

  return asRecord(event.data as Value) ?? {};
}

function stringField(record: Attrs | undefined, key: string): string | undefined {
  const value = record?.[key];

  return value?.constructor === String && value.length > 0 ? value : undefined;
}

function numberField(record: Attrs | undefined, key: string): number | undefined {
  const value = record?.[key];

  return value?.constructor === Number ? value : undefined;
}

function jsonField(record: Attrs | undefined, key: string): string | undefined {
  const value = record?.[key];

  if (value === undefined) return undefined;

  try {
    return JSON.stringify(value);
  } catch {
    return undefined;
  }
}

function baseAttributes(event: MessageStreamEvent, ctx: HookContext) {
  const data = eventData(event);

  return {
    "gen_ai.conversation.id": ctx.session.id,
    "gen_ai.agent.name": ctx.agent.name,
    "eve.channel.kind": ctx.channel.kind,
    "eve.turn.id": stringField(data, "turnId"),
    "eve.step.index": numberField(data, "stepIndex"),
    "eve.turn.sequence": numberField(data, "sequence"),
  };
}

function toolAttrs(result: Value) {
  const record = asRecord(result);
  const subagentName = stringField(record, "subagentName");

  return {
    "gen_ai.tool.name":
      stringField(record, "toolName") ??
      (subagentName ? `eve:subagent:${subagentName}` : undefined),
    "gen_ai.tool.call.id": stringField(record, "callId"),
    "eve.subagent.name": subagentName,
  };
}

function reportError(onError: ((error: Error) => void) | undefined, error: Error): void {
  try {
    onError?.(error);
  } catch {
    // telemetry hooks must never fail an Eve turn.
  }
}

function emit(
  event: MessageStreamEvent,
  ctx: HookContext,
  level: LogLevel,
  message: string,
  attributes: Attrs = {},
  baseEvent: MessageStreamEvent = event,
): void {
  const at = event.meta.at;
  log(message, {
    level,
    eventName: event.type,
    timestamp: at ? new Date(at) : undefined,
    attributes: { ...baseAttributes(baseEvent, ctx), ...attributes },
  });
}

function stepCompletedMessage(data: Attrs): string {
  const finishReason = stringField(data, "finishReason") ?? "unknown";
  const usage = asRecord(data.usage);
  const inputTokens = numberField(usage, "inputTokens");
  const outputTokens = numberField(usage, "outputTokens");

  const suffix =
    inputTokens !== undefined && outputTokens !== undefined
      ? `: ${inputTokens} in / ${outputTokens} out tokens`
      : "";

  return `Step completed (${finishReason})${suffix}`;
}

function logEvent(event: MessageStreamEvent, ctx: HookContext): void {
  const data = eventData(event);

  switch (event.type) {
    case "session.started": {
      const runtime = asRecord(data.runtime);
      const invocation = asRecord(data.invocation);
      const trace = asRecord(data.trace);
      emit(event, ctx, "info", "Session started", {
        "eve.version": stringField(runtime, "eveVersion"),
        "eve.agent.id": stringField(runtime, "agentId"),
        "eve.trace.id": stringField(trace, "traceId"),
        "eve.parent.session_id": stringField(invocation, "parentSessionId"),
        "eve.parent.call_id": stringField(invocation, "parentCallId"),
        "eve.parent.turn_id": stringField(invocation, "parentTurnId"),
        "eve.subagent.name": stringField(invocation, "name"),
      });

      return;
    }

    case "turn.started":
      emit(event, ctx, "debug", "Turn started");

      return;
    case "message.received":
      emit(event, ctx, "debug", "User message received");

      return;
    case "step.started":
      emit(event, ctx, "debug", "Step started", {
        "gen_ai.request.model": stringField(data, "modelId"),
      });

      return;
    case "step.completed": {
      const usage = asRecord(data.usage);
      emit(event, ctx, "info", stepCompletedMessage(data), {
        "gen_ai.usage.input_tokens": numberField(usage, "inputTokens"),
        "gen_ai.usage.output_tokens": numberField(usage, "outputTokens"),
        "eve.usage.cache_read_tokens": numberField(usage, "cacheReadTokens"),
        "eve.usage.cache_write_tokens": numberField(usage, "cacheWriteTokens"),
      });

      return;
    }

    case "step.failed":
      emit(event, ctx, "error", `Step failed: ${stringField(data, "message") ?? ""}`, {
        "error.code": stringField(data, "code"),
        "eve.error.details": jsonField(data, "details"),
      });

      return;
    case "action.result": {
      const status = stringField(data, "status");

      if (status === "completed") return;
      const error = asRecord(data.error);
      emit(
        event,
        ctx,
        status === "failed" ? "error" : "warn",
        `Tool ${status ?? "failed"}: ${stringField(error, "message") ?? ""}`,
        {
          "error.code": stringField(error, "code"),
          ...toolAttrs(data.result),
        },
      );

      return;
    }

    case "input.requested": {
      const requests = Array.isArray(data.requests) ? data.requests : undefined;
      emit(event, ctx, "info", "Input requested (HITL)", {
        "eve.input.request_count": requests?.length,
      });

      return;
    }

    case "authorization.required":
      emit(event, ctx, "warn", `Authorization required: ${stringField(data, "name") ?? ""}`, {
        "eve.authorization.name": stringField(data, "name"),
      });

      return;
    case "authorization.completed": {
      const outcome = stringField(data, "outcome");
      emit(
        event,
        ctx,
        outcome === "authorized" ? "info" : "warn",
        `Authorization ${outcome ?? "completed"}: ${stringField(data, "name") ?? ""}`,
        {
          "eve.authorization.name": stringField(data, "name"),
          "eve.authorization.outcome": outcome,
          "eve.authorization.reason": stringField(data, "reason"),
        },
      );

      return;
    }

    case "subagent.started":
      emit(event, ctx, "info", "Subagent started", {
        "eve.subagent.name": stringField(data, "subagentName"),
        "gen_ai.tool.call.id": stringField(data, "callId"),
      });

      return;
    case "subagent.event": {
      const child = asRecord(data.event as Value);
      const childData = asRecord(child?.data);
      const childType = stringField(child, "type");

      const attrs = {
        "eve.subagent.name": stringField(data, "subagentName"),
        "gen_ai.tool.call.id": stringField(data, "callId"),
      };

      if (
        childType === "step.failed" ||
        childType === "turn.failed" ||
        childType === "session.failed"
      ) {
        emit(
          event,
          ctx,
          "error",
          `Subagent ${childType.replace(".", " ")}: ${stringField(childData, "message") ?? ""}`,
          {
            ...attrs,
            "error.code": stringField(childData, "code"),
            "eve.error.details": jsonField(childData, "details"),
          },
          {
            ...child,
            data: childData,
            type: childType,
          } as MessageStreamEvent,
        );
      }

      return;
    }

    case "subagent.called": {
      const remote = asRecord(data.remote);
      emit(event, ctx, "info", `Subagent called: ${stringField(data, "name") ?? ""}`, {
        "eve.subagent.name": stringField(data, "name"),
        "gen_ai.tool.name": stringField(data, "toolName"),
        "gen_ai.tool.call.id": stringField(data, "callId"),
        "eve.child.session_id": stringField(data, "childSessionId"),
        "eve.workflow.id": stringField(data, "workflowId"),
        "eve.remote.url": stringField(remote, "url"),
      });

      return;
    }

    case "subagent.completed":
      emit(event, ctx, "info", "Subagent completed", {
        "eve.subagent.name": stringField(data, "subagentName"),
        "gen_ai.tool.call.id": stringField(data, "callId"),
      });

      return;
    case "compaction.requested":
      emit(event, ctx, "info", `Compaction requested (${stringField(data, "modelId") ?? ""})`, {
        "gen_ai.request.model": stringField(data, "modelId"),
        "gen_ai.usage.input_tokens": numberField(data, "usageInputTokens"),
      });

      return;
    case "compaction.completed":
      emit(event, ctx, "debug", "Compaction completed", {
        "gen_ai.request.model": stringField(data, "modelId"),
      });

      return;
    case "turn.completed":
      emit(event, ctx, "info", "Turn completed");

      return;
    case "turn.cancelled":
      emit(event, ctx, "warn", "Turn cancelled");

      return;
    case "turn.failed":
      emit(event, ctx, "error", `Turn failed: ${stringField(data, "message") ?? ""}`, {
        "error.code": stringField(data, "code"),
        "eve.error.details": jsonField(data, "details"),
      });

      return;
    case "session.waiting":
      emit(event, ctx, "debug", "Session waiting");

      return;
    case "session.completed":
      emit(event, ctx, "info", "Session completed");

      return;
    case "session.failed":
      emit(event, ctx, "error", `Session failed: ${stringField(data, "message") ?? ""}`, {
        "error.code": stringField(data, "code"),
        "eve.error.details": jsonField(data, "details"),
      });

      return;
    default:
      return;
  }
}

export function telemetryDevHook(
  options: TelemetryDevEveOptions = {},
  overrides?: ClientOverrides,
): HookDefinition {
  return {
    events: {
      "*"(event, ctx) {
        try {
          ensureInit(options, overrides);
          logEvent(event, ctx);
        } catch (error) {
          reportError(options.onError, error instanceof Error ? error : new Error(String(error)));
        }
      },
    },
  };
}
