import type { Span, SpanData, Trace, TracingProcessor } from "@openai/agents";
import {
  captureEnabled,
  flush,
  type SpanFields,
  type SpanHandle,
  startSpan,
  type StartSpanOptions,
  type TokenUsage,
} from "@telemetry-dev/sdk";

import {
  type ClientOverrides,
  ensureInit,
  type TelemetryDevOpenAIAgentsOptions,
} from "./config.ts";
import { SANDBOX_ID_KEY } from "./sandbox.ts";

type AgentsSpan = Span<SpanData>;

type Attrs = NonNullable<SpanFields["attributes"]>;

// Bounds the processor's memory when a host never ends its spans; past the cap new spans are
// dropped, not exported.
const MAX_OPEN_SPANS = 10_000;

// How long a trace with no open spans waits for `onTraceEnd` before its root ends anyway, and how
// long a finished trace is kept afterwards so that late spans still join it. The Agents SDK ends a
// trace only when its run returns; a run that throws never ends it.
const IDLE_TRACE_MS = 60_000;

// Bounds the finished traces kept for late spans when runs finish faster than they expire.
const MAX_TRACES = 10_000;

// `formatExecResponse` in agents-core writes this line in the header of a shell tool's output,
// above `Output:`, once the command has exited. A command still running gets a session id line
// instead, and the command's own output below the header can contain any text.
const EXIT_CODE_LINE = /^Process exited with code (-?\d+)$/m;
const OUTPUT_MARKER = "\nOutput:";

const SANDBOX_COMMAND_TOOLS = new Set(["exec_command", "write_stdin"]);

const SANDBOX_SPAN_PREFIX = "sandbox.";

interface SandboxRef {
  backend?: string;
  id?: string;
}

/** Sandbox of each agent in one run, keyed by agent name. */
type RunSandboxes = Map<string, SandboxRef>;

interface TraceState {
  root: SpanHandle;
  open: number;
  /** When `onTraceEnd` arrived. The root ends once this is set and no span is open. */
  endTime?: Date;
  lastSpanEnd?: Date;
  /** The root has ended. The state stays until it expires, so a late span joins the root. */
  finished: boolean;
  timer?: ReturnType<typeof setTimeout>;
}

interface SpanState {
  handle: SpanHandle;
  /** The trace state this span was counted in, if any. */
  trace?: TraceState;
  /** `shutdown` ended the handle; the SDK's own end of the span only uncounts it. */
  ended?: boolean;
  /** Name of the agent this span runs under, for looking up its sandbox. */
  agent?: string;
  /** Sandboxes of the run this span belongs to; shared by every span of the run. */
  sandboxes?: RunSandboxes;
}

type JsonRecord = Record<string, unknown>;

const asRecord = (value: unknown): JsonRecord | undefined =>
  value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : undefined;

const asString = (value: unknown): string | undefined =>
  typeof value === "string" && value.length > 0 ? value : undefined;

const asNumber = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined;

const toTime = (iso: string | null): Date | undefined => (iso ? new Date(iso) : undefined);

const compact = (usage: TokenUsage): TokenUsage | undefined =>
  Object.values(usage).some((v) => v !== undefined) ? usage : undefined;

/**
 * Usage of a Responses API response (`input_tokens_details`), or of a `generation` span, whose
 * adapters put token details under `details`.
 */
function usageOf(value: unknown): TokenUsage | undefined {
  const usage = asRecord(value);

  if (!usage) return undefined;
  const details = asRecord(usage.details);
  const inputDetails = asRecord(usage.input_tokens_details) ?? details;
  const outputDetails = asRecord(usage.output_tokens_details) ?? details;

  return compact({
    inputTokens: asNumber(usage.input_tokens),
    outputTokens: asNumber(usage.output_tokens),
    totalTokens: asNumber(usage.total_tokens),
    cacheReadInputTokens: asNumber(inputDetails?.cached_tokens),
    reasoningOutputTokens: asNumber(outputDetails?.reasoning_tokens),
  });
}

/** Usage of a Chat Completions response, which a successful `generation` span keeps in its output. */
function chatCompletionUsageOf(value: unknown): TokenUsage | undefined {
  const usage = asRecord(value);

  if (!usage) return undefined;

  return compact({
    inputTokens: asNumber(usage.prompt_tokens),
    outputTokens: asNumber(usage.completion_tokens),
    totalTokens: asNumber(usage.total_tokens),
    cacheReadInputTokens: asNumber(asRecord(usage.prompt_tokens_details)?.cached_tokens),
    reasoningOutputTokens: asNumber(asRecord(usage.completion_tokens_details)?.reasoning_tokens),
  });
}

function primitiveMetadata(value: unknown): Record<string, string | number | boolean> | undefined {
  const record = asRecord(value);

  if (!record) return undefined;
  const out: Record<string, string | number | boolean> = {};

  for (const [key, v] of Object.entries(record)) {
    if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") out[key] = v;
  }

  return Object.keys(out).length > 0 ? out : undefined;
}

function agentOf(data: SpanData): string | undefined {
  switch (data.type) {
    case "agent":
      return data.name;
    case "turn":
      return data.agent_name;
    case "custom":
      return asString(data.data.agent_name);
    default:
      return undefined;
  }
}

/** Name and fields known when the span starts. */
function startOf(data: SpanData): { name: string; options: StartSpanOptions } {
  switch (data.type) {
    case "agent":
      return { name: data.name, options: { type: "agent", agentName: data.name } };
    case "task":
      return { name: `task ${data.name}`, options: {} };
    case "turn":
      return {
        name: `turn ${data.turn}`,
        options: { attributes: { "openai_agents.turn": data.turn } },
      };
    case "response":
      return { name: "chat", options: { type: "generation", provider: "openai" } };
    case "generation":
      return {
        name: data.model ? `chat ${data.model}` : "chat",
        options: { type: "generation", model: data.model },
      };
    case "function":
      return { name: data.name, options: { type: "tool", toolName: data.name } };
    case "handoff":
      return { name: "handoff", options: {} };
    case "guardrail":
      return {
        name: `guardrail ${data.name}`,
        options: { attributes: { "openai_agents.guardrail.name": data.name } },
      };
    case "mcp_tools":
      return { name: data.server ? `mcp_tools ${data.server}` : "mcp_tools", options: {} };
    case "custom":
      return { name: data.name, options: {} };
    default:
      return { name: data.type, options: {} };
  }
}

/** Fields known when the span ends, apart from sandbox attributes. */
function endOf(data: SpanData): SpanFields {
  switch (data.type) {
    // The SDK fills the tool list once the agent's tools are resolved, after the span starts.
    case "agent":
      return {
        attributes: {
          ...(data.tools?.length ? { "openai_agents.agent.tools": data.tools } : {}),
          ...(data.handoffs?.length ? { "openai_agents.agent.handoffs": data.handoffs } : {}),
          ...(data.output_type ? { "openai_agents.agent.output_type": data.output_type } : {}),
        },
      };
    case "response": {
      const response = asRecord(data._response);
      const model = asString(response?.model);

      return {
        ...(model ? { name: `chat ${model}`, model } : {}),
        responseModel: model,
        responseId: data.response_id ?? asString(response?.id),
        usage: usageOf(response?.usage),
        input: data._input,
        output: response?.output as SpanFields["output"],
      };
    }

    // The Chat Completions model sets the model after the span starts, and on success leaves
    // `usage` empty: the usage is only in the completion it stores as output.
    case "generation": {
      const completion = asRecord(data.output?.[0]);
      const model = asString(data.model);

      return {
        ...(model ? { name: `chat ${model}`, model } : {}),
        responseModel: asString(completion?.model),
        responseId: asString(completion?.id),
        usage: usageOf(data.usage) ?? chatCompletionUsageOf(completion?.usage),
        input: data.input as SpanFields["input"],
        output: data.output as SpanFields["output"],
      };
    }

    case "function": {
      // `mcp_data` is typed as a string but holds `{ server }` at runtime.
      const mcpServer = asString(asRecord(data.mcp_data)?.server) ?? asString(data.mcp_data);

      return {
        input: data.input,
        output: data.output,
        ...(mcpServer ? { attributes: { "openai_agents.mcp.server": mcpServer } } : {}),
      };
    }

    case "handoff":
      return {
        name:
          data.from_agent && data.to_agent
            ? `handoff ${data.from_agent} -> ${data.to_agent}`
            : undefined,
        attributes: {
          ...(data.from_agent ? { "openai_agents.handoff.from": data.from_agent } : {}),
          ...(data.to_agent ? { "openai_agents.handoff.to": data.to_agent } : {}),
        },
      };
    case "guardrail":
      return { attributes: { "openai_agents.guardrail.triggered": data.triggered } };
    case "mcp_tools":
      return data.result
        ? { attributes: { "openai_agents.mcp.tool_count": data.result.length } }
        : {};
    case "custom":
      return data.name.startsWith(SANDBOX_SPAN_PREFIX)
        ? {}
        : { metadata: primitiveMetadata(data.data) };
    default:
      return {};
  }
}

function errorOf(span: AgentsSpan): Error | undefined {
  const error = span.error;

  if (!error) return undefined;
  const sandboxError =
    span.spanData.type === "custom" ? asRecord(span.spanData.data.error) : undefined;
  const result = new Error(asString(sandboxError?.message) ?? error.message);
  result.name = asString(sandboxError?.name) ?? "Error";

  return result;
}

function exitCodeOf(output: string | undefined): number | undefined {
  if (!output) return undefined;
  const marker = output.indexOf(OUTPUT_MARKER);
  const header = marker < 0 ? output : output.slice(0, marker);
  const code = EXIT_CODE_LINE.exec(header)?.[1];

  return code === undefined ? undefined : Number(code);
}

function mergeFields(...parts: SpanFields[]): SpanFields {
  const merged: SpanFields = {};

  for (const part of parts) {
    const { attributes, ...rest } = part;
    Object.assign(merged, rest);

    if (attributes) merged.attributes = { ...merged.attributes, ...attributes };
  }

  return merged;
}

const sandboxAttributes = (ref: SandboxRef | undefined): Attrs => ({
  ...(ref?.backend ? { "sandbox.backend": ref.backend } : {}),
  ...(ref?.id ? { "sandbox.id": ref.id } : {}),
});

/**
 * Sends OpenAI Agents SDK traces to telemetry.dev. Register it with `addTraceProcessor`, which
 * keeps the SDK's default OpenAI exporter, or `setTraceProcessors([...])`, which replaces it.
 *
 * Each Agents trace gets one root span, and its spans keep their parents. Model calls are `chat`
 * spans and the only spans with token usage; task, turn, and agent spans repeat the totals of their
 * calls and carry none. Sandbox spans keep their `sandbox.*` names and get `sandbox.backend`,
 * `sandbox.id`, and `sandbox.operation`. The `exec_command` and `write_stdin` tool spans get
 * `process.exit.code` once the command has exited. Command text and working directory are input
 * content: they are sent only when `captureInput` is on, and go through the mask.
 *
 * The root span ends when the trace has ended and every one of its spans has ended, since other
 * processors can delay span ends past the trace end. A trace of a run that threw ends after
 * `IDLE_TRACE_MS` without open spans, or at the next `forceFlush` or `shutdown`.
 */
export function telemetryDevProcessor(
  options: TelemetryDevOpenAIAgentsOptions = {},
  overrides?: ClientOverrides,
): TracingProcessor {
  const onError = options.onError;

  const report = (cause: unknown): void => {
    try {
      onError?.(cause instanceof Error ? cause : new Error(String(cause)));
    } catch {
      // A failing onError must not fail an agent run.
    }
  };

  try {
    ensureInit(options, overrides);
  } catch (error) {
    report(error);
  }

  const traces = new Map<string, TraceState>();
  const spans = new Map<string, SpanState>();
  let reportedCap = false;

  function finish(state: TraceState): void {
    if (state.finished) return;
    state.finished = true;
    state.root.end({ endTime: state.endTime ?? state.lastSpanEnd });
  }

  function remove(traceId: string, state: TraceState): void {
    clearTimeout(state.timer);

    if (traces.get(traceId) === state) traces.delete(traceId);
  }

  /** Ends the root or schedules its end or expiry, from the trace's open span count. */
  function settle(traceId: string, state: TraceState): void {
    clearTimeout(state.timer);
    state.timer = undefined;

    if (state.open > 0) return;

    if (state.endTime) finish(state);

    // An unfinished trace waits for `onTraceEnd`; a finished one waits for late spans.
    state.timer = setTimeout(() => {
      try {
        finish(state);
        remove(traceId, state);
      } catch (error) {
        report(error);
      }
    }, IDLE_TRACE_MS);
    state.timer.unref?.();
  }

  function uncount(
    traceId: string,
    state: TraceState | undefined,
    endTime: Date | undefined,
  ): void {
    if (!state) return;
    state.open--;

    if (endTime && (!state.lastSpanEnd || endTime > state.lastSpanEnd)) state.lastSpanEnd = endTime;
    settle(traceId, state);
  }

  function evictFinished(): void {
    for (const [traceId, state] of traces) {
      if (traces.size < MAX_TRACES) return;

      if (state.finished && state.open === 0) remove(traceId, state);
    }
  }

  function sandboxSpanEnd(span: AgentsSpan, state: SpanState): SpanFields {
    const data = span.spanData;

    if (data.type !== "custom" || !data.name.startsWith(SANDBOX_SPAN_PREFIX)) return {};
    const backend = asString(data.data.backend_id);
    const id = asString(data.data[SANDBOX_ID_KEY]);
    const own: SandboxRef = { ...(backend ? { backend } : {}), ...(id ? { id } : {}) };

    if (state.agent && state.sandboxes && (backend || id)) {
      state.sandboxes.set(state.agent, { ...state.sandboxes.get(state.agent), ...own });
    }

    const attributes: Attrs = {
      "sandbox.operation": data.name.slice(SANDBOX_SPAN_PREFIX.length),
      ...sandboxAttributes({ ...sandboxFor(state), ...own }),
    };

    if (data.name === "sandbox.exec" && captureEnabled("input")) {
      const command = asString(data.data.cmd);
      const workdir = asString(data.data.workdir);
      const commandLine =
        command && state.handle.serializeContent?.(command, "process.command_line");
      const directory =
        workdir && state.handle.serializeContent?.(workdir, "process.working_directory");

      if (commandLine) attributes["process.command_line"] = commandLine;

      if (directory) attributes["process.working_directory"] = directory;
    }

    return { attributes };
  }

  function commandToolEnd(span: AgentsSpan, state: SpanState): SpanFields {
    const data = span.spanData;

    if (data.type !== "function" || !SANDBOX_COMMAND_TOOLS.has(data.name)) return {};
    const exitCode = exitCodeOf(data.output);
    const attributes: Attrs = sandboxAttributes(sandboxFor(state));

    // A non-zero exit is recorded, not raised: `grep` exits 1 when nothing matches.
    if (exitCode !== undefined) attributes["process.exit.code"] = exitCode;

    return Object.keys(attributes).length > 0 ? { attributes } : {};
  }

  const sandboxFor = (state: SpanState): SandboxRef | undefined =>
    state.agent ? state.sandboxes?.get(state.agent) : undefined;

  return {
    async onTraceStart(trace: Trace) {
      try {
        const existing = traces.get(trace.traceId);

        // A trace id seen again, such as a resumed run's or a fixed `traceId`, keeps its root while
        // the processor still holds it, even when that root has already ended.
        if (existing) {
          existing.endTime = undefined;
          settle(trace.traceId, existing);

          return;
        }

        evictFinished();

        const attributes: Attrs = { "openai_agents.trace_id": trace.traceId };

        if (trace.groupId) attributes["gen_ai.conversation.id"] = trace.groupId;
        const root = startSpan(trace.name, {
          attributes,
          metadata: primitiveMetadata(trace.metadata),
        });
        const state: TraceState = { root, open: 0, finished: false };
        traces.set(trace.traceId, state);
        settle(trace.traceId, state);
      } catch (error) {
        report(error);
      }
    },

    async onTraceEnd(trace: Trace) {
      try {
        const state = traces.get(trace.traceId);

        if (!state) return;
        state.endTime = new Date();
        settle(trace.traceId, state);
      } catch (error) {
        report(error);
      }
    },

    async onSpanStart(span: AgentsSpan) {
      try {
        if (spans.size >= MAX_OPEN_SPANS) {
          if (!reportedCap) {
            reportedCap = true;
            report(new Error(`more than ${MAX_OPEN_SPANS} open Agents spans; dropping new spans`));
          }

          return;
        }

        const data = span.spanData;
        const parent = span.parentId ? spans.get(span.parentId) : undefined;
        const traceState = traces.get(span.traceId);
        const { name, options } = startOf(data);
        const handle = startSpan(name, {
          ...options,
          parent: parent?.handle ?? traceState?.root,
          startTime: toTime(span.startedAt),
        });

        // A task span is one run; an agent span without one is a run without task spans.
        const sandboxes =
          data.type === "task" || (data.type === "agent" && !parent?.sandboxes)
            ? new Map<string, SandboxRef>()
            : parent?.sandboxes;

        spans.set(span.spanId, {
          handle,
          trace: traceState,
          agent: agentOf(data) ?? parent?.agent,
          sandboxes,
        });

        if (traceState) {
          traceState.open++;
          settle(span.traceId, traceState);
        }
      } catch (error) {
        report(error);
      }
    },

    async onSpanEnd(span: AgentsSpan) {
      try {
        const state = spans.get(span.spanId);

        if (!state) return;
        spans.delete(span.spanId);

        if (state.ended) {
          uncount(span.traceId, state.trace, undefined);

          return;
        }

        const fields = mergeFields(
          endOf(span.spanData),
          sandboxSpanEnd(span, state),
          commandToolEnd(span, state),
        );
        const error = errorOf(span);
        const endTime = toTime(span.endedAt);

        state.handle.end({ ...fields, ...(error ? { error } : {}), endTime });
        uncount(span.traceId, state.trace, endTime ?? new Date());
      } catch (error) {
        report(error);
      }
    },

    // Ends the roots of traces whose run threw. A trace that has not ended a span yet may be a
    // run that is still preparing its first span, so it keeps waiting.
    async forceFlush() {
      try {
        for (const state of traces.values()) {
          if (state.open === 0 && (state.endTime || state.lastSpanEnd)) finish(state);
        }
      } catch (error) {
        report(error);
      }

      await flush();
    },

    // The SDK also calls this on SIGTERM, SIGINT, and unhandled rejections, after which a server
    // that drains requests keeps running. So spans and traces stay known: later spans join their
    // parents, and the SDK's own ends of the spans ended here only uncount them.
    async shutdown() {
      try {
        for (const state of spans.values()) {
          if (state.ended) continue;
          state.ended = true;
          state.handle.end();
        }

        for (const state of traces.values()) finish(state);
      } catch (error) {
        report(error);
      }

      await flush();
    },
  };
}
