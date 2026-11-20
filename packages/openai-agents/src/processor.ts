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

type AgentsSpan = Span<SpanData>;

type Attrs = NonNullable<SpanFields["attributes"]>;

// Bounds the processor's memory when a host never ends its spans; past the cap new spans are
// dropped, not exported.
const MAX_OPEN_SPANS = 10_000;

// Bounds the trace roots kept so that spans arriving after their trace ended still find their root.
// Past the cap the root of the trace whose spans started or ended least recently is forgotten, and
// ended first if it is still open.
const MAX_TRACES = 10_000;

// `formatExecResponse` in agents-core writes this line in the header of a shell tool's output,
// above `Output:`, once the command has exited. A command still running gets a session id line
// instead, and the command's own output below the header can contain any text.
const EXIT_CODE_LINE = /^Process exited with code (-?\d+)$/m;
const OUTPUT_MARKER = "\nOutput:";

const SANDBOX_COMMAND_TOOLS = new Set(["exec_command", "write_stdin"]);

const SANDBOX_SPAN_PREFIX = "sandbox.";

interface TraceState {
  root: SpanHandle;
  ended: boolean;
  /** Latest observed span end, used to timestamp the root when the trace never ends. */
  lastEnd?: Date;
}

interface OpenSpan {
  handle: SpanHandle;
  /** The root state the span started under; a later trace can reuse the same trace id. */
  trace?: TraceState;
}

// The top-level spans of a run. Upstream ends them with the run's error even when the trace never
// ends; a failed custom span at the top level may be handled by the caller.
const RUN_SPAN_TYPES = new Set(["task", "agent"]);

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

/** Name and fields known when the span starts. */
function startOf(data: SpanData): { name: string; options: StartSpanOptions } {
  switch (data.type) {
    case "agent":
      return {
        name: `invoke_agent ${data.name}`,
        options: { type: "agent", agentName: data.name },
      };
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
      return {
        name: `execute_tool ${data.name}`,
        options: { type: "tool", toolName: data.name },
      };
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

/**
 * Sends OpenAI Agents SDK traces to telemetry.dev. Register it with `addTraceProcessor`, which
 * keeps the SDK's default OpenAI exporter, or `setTraceProcessors([...])`, which replaces it.
 *
 * Each Agents trace gets one root span, and its spans keep their parents. Model calls are `chat`
 * spans and the only spans with token usage; task, turn, and agent spans repeat the totals of their
 * calls and carry none. Sandbox spans keep their `sandbox.*` names and get `sandbox.operation`; the
 * spans that create or resume a session also get `sandbox.backend`. The `exec_command` and
 * `write_stdin` tool spans get `process.exit.code` once the command has exited. Command text and
 * working directory are input content: they are sent only when `captureInput` is on, and go
 * through the mask.
 *
 * The root span ends when the trace ends. A run that throws before the SDK ends its trace leaves
 * it open, so the root also ends when the run's task or agent span ends with an error, and
 * `forceFlush` and `shutdown` end the root of every trace with no open span at the end of its last
 * span. Spans that end after the root still nest under it while the processor holds the root (the
 * `MAX_TRACES` most recently active traces): telemetry.dev takes a trace's start and end from all
 * of its spans.
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
  const spans = new Map<string, OpenSpan>();
  let reportedCap = false;

  function endRoot(state: TraceState, endTime?: Date): void {
    if (state.ended) return;
    state.ended = true;
    state.root.end({ endTime });
  }

  /** Moves a trace to the end of the map, which eviction reads as least recently active first. */
  function touch(traceId: string, state: TraceState): void {
    traces.delete(traceId);
    traces.set(traceId, state);
  }

  /** Ends, at their last span's end, the roots of traces that have no open span. */
  function endIdleRoots(): void {
    const open = new Set<TraceState | undefined>();

    for (const { trace } of spans.values()) open.add(trace);

    for (const state of traces.values()) {
      if (!open.has(state)) endRoot(state, state.lastEnd);
    }
  }

  function sandboxSpanEnd(span: AgentsSpan, handle: SpanHandle): SpanFields {
    const data = span.spanData;

    if (data.type !== "custom" || !data.name.startsWith(SANDBOX_SPAN_PREFIX)) return {};
    const backend = asString(data.data.backend_id);
    const attributes: Attrs = {
      "sandbox.operation": data.name.slice(SANDBOX_SPAN_PREFIX.length),
      ...(backend ? { "sandbox.backend": backend } : {}),
    };

    if (data.name === "sandbox.exec" && captureEnabled("input")) {
      const command = asString(data.data.cmd);
      const workdir = asString(data.data.workdir);
      const commandLine = command && handle.serializeContent?.(command, "process.command_line");
      const directory = workdir && handle.serializeContent?.(workdir, "process.working_directory");

      if (commandLine) attributes["process.command_line"] = commandLine;

      if (directory) attributes["process.working_directory"] = directory;
    }

    return { attributes };
  }

  function commandToolEnd(span: AgentsSpan): SpanFields {
    const data = span.spanData;

    if (data.type !== "function" || !SANDBOX_COMMAND_TOOLS.has(data.name)) return {};
    const exitCode = exitCodeOf(data.output);

    // A non-zero exit is recorded, not raised: `grep` exits 1 when nothing matches.
    return exitCode === undefined ? {} : { attributes: { "process.exit.code": exitCode } };
  }

  return {
    async onTraceStart(trace: Trace) {
      try {
        // A trace id seen again, such as a resumed run's or a fixed `traceId`, keeps its root while
        // the processor still holds it, even when that root has already ended.
        if (traces.has(trace.traceId)) return;

        // Map order is least recently active first: span starts and ends move a trace to the end.
        const oldest = traces.size >= MAX_TRACES ? traces.entries().next() : undefined;

        if (oldest && !oldest.done) {
          const [oldestId, state] = oldest.value;
          endRoot(state, state.lastEnd);
          traces.delete(oldestId);
        }

        const attributes: Attrs = { "openai_agents.trace_id": trace.traceId };

        if (trace.groupId) attributes["gen_ai.conversation.id"] = trace.groupId;
        const root = startSpan(trace.name, {
          attributes,
          metadata: primitiveMetadata(trace.metadata),
        });
        traces.set(trace.traceId, { root, ended: false });
      } catch (error) {
        report(error);
      }
    },

    async onTraceEnd(trace: Trace) {
      try {
        const state = traces.get(trace.traceId);

        if (state) endRoot(state);
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

        const state = traces.get(span.traceId);

        if (state) touch(span.traceId, state);

        const { name, options } = startOf(span.spanData);
        const parent =
          (span.parentId ? spans.get(span.parentId)?.handle : undefined) ?? state?.root;
        const handle = startSpan(name, {
          ...options,
          parent,
          startTime: toTime(span.startedAt),
        });
        spans.set(span.spanId, { handle, trace: state });
      } catch (error) {
        report(error);
      }
    },

    async onSpanEnd(span: AgentsSpan) {
      try {
        const open = spans.get(span.spanId);

        if (!open) return;
        spans.delete(span.spanId);
        const { handle, trace: state } = open;

        const fields = mergeFields(
          endOf(span.spanData),
          sandboxSpanEnd(span, handle),
          commandToolEnd(span),
        );
        const error = errorOf(span);
        const endTime = toTime(span.endedAt);

        handle.end({ ...fields, ...(error ? { error } : {}), endTime });

        if (!state) return;
        const end = endTime ?? new Date();

        if (!state.lastEnd || end > state.lastEnd) state.lastEnd = end;

        if (traces.get(span.traceId) === state) touch(span.traceId, state);

        // The top-level span of a run that throws (its task span, or its agent span without task
        // spans) ends with the error, but the trace never ends.
        if (error && !span.parentId && RUN_SPAN_TYPES.has(span.spanData.type)) endRoot(state, end);
      } catch (error) {
        report(error);
      }
    },

    // Also ends the roots of traces that threw outside any span, such as a run that failed before
    // its first span or a `withTrace` callback that threw after its run returned: no span carries
    // the error, and the trace never ends. A trace between two of its runs is ended too; spans of
    // the next run still nest under its root.
    async forceFlush() {
      try {
        endIdleRoots();
      } catch (error) {
        report(error);
      }

      await flush();
    },

    // The SDK also calls this on SIGTERM, SIGINT, and unhandled rejections, so a process that exits
    // right after still exports what it recorded. Roots stay known: spans of a server that keeps
    // draining requests still join their trace.
    async shutdown() {
      try {
        endIdleRoots();

        for (const { handle } of spans.values()) handle.end();
        spans.clear();

        for (const state of traces.values()) endRoot(state);
      } catch (error) {
        report(error);
      }

      await flush();
    },
  };
}
