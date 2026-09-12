import { context as apiContext, SpanStatusCode, trace } from "@opentelemetry/api";
import { InMemoryLogRecordExporter } from "@opentelemetry/sdk-logs";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  type ReadableSpan,
} from "@opentelemetry/sdk-trace-base";
import { flush, type ClientOverrides } from "@telemetry-dev/sdk";
import { Client } from "eve/client";
import type { MessageStreamEvent } from "eve/client";
import type {
  InstrumentationDefinition,
  InstrumentationStepStartedEventInput,
} from "eve/instrumentation";
import type { HookContext } from "eve/hooks";
import { afterEach, expect, test, vi } from "vitest";

import { wrapEveClient } from "../src/client.ts";
import { resetForTesting } from "../src/config.ts";
import { telemetryDevHook } from "../src/hook.ts";
import { telemetryDevOtelIntegration } from "../src/otel.ts";
import {
  telemetryDevInstrumentation,
  type TelemetryDevInstrumentationOptions,
} from "../src/instrumentation.ts";

type ExportedServiceName = ReadableSpan["resource"]["attributes"][string] | undefined;

type TestValue =
  | string
  | number
  | boolean
  | bigint
  | null
  | undefined
  | readonly TestValue[]
  | { readonly [key: string]: TestValue };

const originalOtelServiceName = process.env.OTEL_SERVICE_NAME;

interface Exporters extends ClientOverrides {
  logs: InMemoryLogRecordExporter;
  spans: InMemorySpanExporter;
}

function makeExporters(): Exporters {
  const spans = new InMemorySpanExporter();
  const logs = new InMemoryLogRecordExporter();

  return { logRecordExporter: logs, logs, spanExporter: spans, spans };
}

function telemetryOptions<TOptions extends object>(options?: TOptions) {
  return {
    environment: "test",
    exportMode: "immediate" as const,
    logLevel: "silent" as const,
    fetch: async () => new Response(null, { status: 200 }),
    ...options,
  };
}

function setupInstrumentation(
  options: TelemetryDevInstrumentationOptions,
  overrides: Exporters,
): InstrumentationDefinition {
  const definition = telemetryDevInstrumentation(telemetryOptions(options), overrides);
  definition.setup?.({ agentName: "agent-from-setup" });

  return definition;
}

async function exportedServiceName(
  options: TelemetryDevInstrumentationOptions,
  envServiceName: string | undefined,
  agentName: string,
): Promise<ExportedServiceName> {
  await resetForTesting();

  if (envServiceName === undefined) {
    delete process.env.OTEL_SERVICE_NAME;
  } else {
    process.env.OTEL_SERVICE_NAME = envServiceName;
  }

  const exporters = makeExporters();

  try {
    const definition = telemetryDevInstrumentation(telemetryOptions(options), exporters);
    definition.setup?.({ agentName });
    trace.getTracer("eve").startSpan(`service-${agentName}`).end();
    await flush();

    return exporters.spans.getFinishedSpans()[0]?.resource.attributes["service.name"];
  } finally {
    await resetForTesting();
  }
}

function stepInput(auth?: {
  current?: string;
  initiator?: string;
}): InstrumentationStepStartedEventInput {
  const channel = { kind: "unit" } as never;

  return {
    channel,
    modelInput: { instructions: undefined, messages: [] },
    session: {
      auth: {
        current: auth?.current
          ? {
              attributes: {},
              authenticator: "test",
              principalId: auth.current,
              principalType: "user",
            }
          : null,
        initiator: auth?.initiator
          ? {
              attributes: {},
              authenticator: "test",
              principalId: auth.initiator,
              principalType: "user",
            }
          : null,
      },
      id: "session-1",
    },
    step: { index: 1 },
    turn: { id: "turn-1", sequence: 2 },
  };
}

function hookContext(): HookContext {
  return {
    agent: { name: "support-agent" },
    channel: { kind: "web" },
    getSandbox: async () => {
      throw new Error("not used");
    },
    getSkill: () => {
      throw new Error("not used");
    },
    session: {
      auth: { current: null, initiator: null },
      id: "session-log",
      turn: { id: "turn-ctx", sequence: 7 },
    },
  };
}

async function invokeHook(event: MessageStreamEvent, exporters: Exporters): Promise<void> {
  const hook = telemetryDevHook(telemetryOptions(), exporters);
  await hook.events?.["*"]?.(event, hookContext());
  await flush();
}

type EventPayload<TType extends MessageStreamEvent["type"]> =
  Extract<MessageStreamEvent, { type: TType }> extends { data: infer TData } ? TData : never;

function event<TType extends MessageStreamEvent["type"]>(
  type: TType,
  ...args: EventPayload<TType> extends never ? [] : [data: EventPayload<TType>]
): Extract<MessageStreamEvent, { type: TType }> {
  const [data] = args;

  const result = {
    meta: { at: "2026-01-02T03:04:05.000Z", id: `evt_${type}` },
    type,
  };

  if (data !== undefined) Object.assign(result, { data });

  return result as Extract<MessageStreamEvent, { type: TType }>;
}

type UnsafeEventData = TestValue;

function unsafeEvent(type: string, data?: UnsafeEventData): MessageStreamEvent {
  return {
    data,
    meta: { at: "2026-01-02T03:04:05.000Z", id: `evt_${type}` },
    type,
  } as MessageStreamEvent;
}

function jsonResponse(body: TestValue, status = 200): Response {
  return new Response(JSON.stringify(body), {
    headers: { "content-type": "application/json" },
    status,
  });
}

function textResponse(body: string, status: number): Response {
  return new Response(body, { status });
}

function ndjsonResponse(events: readonly MessageStreamEvent[]): Response {
  const encoder = new TextEncoder();

  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const item of events) {
        controller.enqueue(encoder.encode(`${JSON.stringify(item)}\n`));
      }

      controller.close();
    },
  });

  return new Response(body, {
    headers: {
      "content-type": "application/x-ndjson; charset=utf-8",
      "x-eve-stream-version": "25",
    },
    status: 200,
  });
}

function stubFetchSequence(responses: Response[]) {
  const requests: Array<Parameters<typeof fetch>> = [];

  const fetchMock = vi.fn(async (...args: Parameters<typeof fetch>) => {
    requests.push(args);
    const response = responses.shift();

    if (!response) throw new Error("unexpected fetch");

    return response;
  });

  vi.stubGlobal("fetch", fetchMock);

  return { requests };
}

function requestUrl(input: Parameters<typeof fetch>[0]): string {
  if (input instanceof URL) return input.href;

  if (input instanceof Request) return input.url;

  return String(input);
}

function wrapClient(exporters: Exporters) {
  return wrapEveClient(
    new Client({ host: "https://eve.test" }),
    telemetryOptions({
      apiKey: "td_live_test",
      agentName: "wrapped-agent",
      serviceName: "wrapped-service",
    }),
    exporters,
  );
}

function successfulTurnEvents(): MessageStreamEvent[] {
  return [
    event("step.completed", {
      finishReason: "stop",
      sequence: 1,
      stepIndex: 0,
      turnId: "turn-success",
      usage: {
        cacheReadTokens: 2,
        cacheWriteTokens: 3,
        costUsd: 0.01,
        outputTokens: 13,
        inputTokens: 11,
      },
    }),
    event("step.completed", {
      finishReason: "stop",
      sequence: 2,
      stepIndex: 1,
      turnId: "turn-success",
      usage: { costUsd: 0.02 },
    }),
    event("message.completed", {
      finishReason: "stop",
      message: "final answer",
      sequence: 3,
      stepIndex: 0,
      turnId: "turn-success",
    }),
    event("result.completed", {
      result: { ok: true },
      sequence: 4,
      stepIndex: 0,
      turnId: "turn-success",
    }),
    event("session.completed"),
  ];
}

function onlySpan(exporters: Exporters): ReadableSpan {
  const spans = exporters.spans.getFinishedSpans();
  expect(spans).toHaveLength(1);

  return spans[0]!;
}

function onlyLog(exporters: Exporters) {
  const records = exporters.logs.getFinishedLogRecords();
  expect(records).toHaveLength(1);

  return records[0]!;
}

afterEach(async () => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();

  if (originalOtelServiceName === undefined) {
    delete process.env.OTEL_SERVICE_NAME;
  } else {
    process.env.OTEL_SERVICE_NAME = originalOtelServiceName;
  }

  await resetForTesting();
});

test("instrumentation setup chooses serviceName by explicit option, OTEL_SERVICE_NAME, then agentName", async () => {
  await expect(
    exportedServiceName({ serviceName: "explicit-service" }, "env-service", "agent-service"),
  ).resolves.toBe("explicit-service");
  await expect(exportedServiceName({}, "env-service", "agent-service")).resolves.toBe(
    "env-service",
  );
  await expect(exportedServiceName({}, undefined, "agent-service")).resolves.toBe("agent-service");
});

test("instrumentation default global filter keeps eve and SDK spans, drops other libraries", async () => {
  const exporters = makeExporters();
  setupInstrumentation({}, exporters);

  trace.getTracer("eve").startSpan("eve-span").end();
  trace.getTracer("eve.agent").startSpan("agent-span").end();
  trace.getTracer("gen_ai").startSpan("ai.streamText").end();
  trace.getTracer("@telemetry-dev/sdk").startSpan("sdk-span").end();
  trace.getTracer("workflow").startSpan("step.execute").end();
  trace.getTracer("better-auth").startSpan("GET /get-session").end();
  await flush();

  expect(exporters.spans.getFinishedSpans().map((span) => span.name)).toEqual([
    "eve-span",
    "agent-span",
    "ai.streamText",
    "sdk-span",
  ]);
});

test("instrumentation custom spanFilter narrows global eve spans", async () => {
  const exporters = makeExporters();
  setupInstrumentation(
    { spanFilter: (span) => span.instrumentationScope.name === "eve.agent" },
    exporters,
  );

  trace.getTracer("eve").startSpan("drop-eve").end();
  trace.getTracer("eve.agent").startSpan("keep-agent").end();
  await flush();

  expect(exporters.spans.getFinishedSpans().map((span) => span.name)).toEqual(["keep-agent"]);
});

test("instrumentation puts every ai.eve.turn of a session in the session trace", async () => {
  const exporters = makeExporters();
  setupInstrumentation({ apiKey: "td_live_test" }, exporters);
  const eve = trace.getTracer("eve");
  const genAi = trace.getTracer("gen_ai");

  for (const turnId of ["turn_0", "turn_1"]) {
    // eve runs each turn inside a workflow-engine span that starts its own trace.
    const engine = trace.getTracer("workflow").startSpan("workflow.execute");

    const turn = eve.startSpan(
      "ai.eve.turn",
      { attributes: { "eve.session.id": "session-1", "eve.turn.id": turnId } },
      trace.setSpan(apiContext.active(), engine),
    );

    genAi.startSpan("ai.streamText", {}, trace.setSpan(apiContext.active(), turn)).end();
    turn.end();
    engine.end();
  }

  await flush();

  // sha256("td_live_test\0session-1"): trace id = digest[0:16], session parent = digest[16:24]
  const spans = exporters.spans.getFinishedSpans();
  expect(spans.map((span) => span.name)).toEqual([
    "ai.streamText",
    "ai.eve.turn",
    "ai.streamText",
    "ai.eve.turn",
  ]);

  for (const span of spans) {
    expect(span.spanContext().traceId).toBe("1382eccf84291f6161a535bdd642e06c");
  }

  for (const span of spans.filter((span) => span.name === "ai.eve.turn")) {
    expect(span.parentSpanContext?.spanId).toBe("da259c4698ed3d9f");
  }
});

test("instrumentation captureInput:false maps to eve recordInputs:false", () => {
  const definition = telemetryDevInstrumentation(telemetryOptions({ captureInput: false }));

  expect(definition.recordInputs).toBe(false);
});

test("provider integration forwards content capture flags", () => {
  const integration = telemetryDevOtelIntegration({
    recordInputs: false,
    recordOutputs: false,
  });

  expect(integration.content).toEqual({ recordInputs: false, recordOutputs: false });
});

test.each([
  { name: "omitted", options: {}, httpExports: 0, totalExports: 1 },
  {
    name: "explicitly undefined",
    options: { spanFilter: undefined },
    httpExports: 0,
    totalExports: 1,
  },
  {
    name: "custom allow-all",
    options: { spanFilter: () => true },
    httpExports: 1,
    totalExports: 2,
  },
  {
    name: "custom HTTP-only",
    options: {
      spanFilter: (span: ReadableSpan) => span.instrumentationScope.name === "unrelated-http",
    },
    httpExports: 1,
    totalExports: 1,
  },
])(
  "provider integration applies the $name filter to actual exports",
  async ({ options, httpExports, totalExports }) => {
    const requests: string[] = [];

    const integration = telemetryDevOtelIntegration({
      apiKey: "td_live_test",
      baseUrl: "https://ingest.test",
      exportMode: "immediate",
      metrics: false,
      fetch: async (input) => {
        requests.push(requestUrl(input));

        return new Response(null, { status: 200 });
      },
      ...options,
    });

    const provider = new BasicTracerProvider({
      spanProcessors: integration.spanProcessors.filter(
        (processor) => typeof processor !== "string",
      ),
    });

    try {
      provider.getTracer("unrelated-http").startSpan("GET /health").end();
      await provider.forceFlush();
      expect(requests).toEqual(Array(httpExports).fill("https://ingest.test/v1/traces"));

      provider.getTracer("eve").startSpan("ai.eve.turn").end();
      await provider.forceFlush();
      expect(requests).toEqual(Array(totalExports).fill("https://ingest.test/v1/traces"));
    } finally {
      await provider.shutdown();
    }
  },
);

test("step.started merges integration context and lets the user callback win", () => {
  const definition = telemetryDevInstrumentation(
    telemetryOptions({
      runtimeContext: { tenant: "acme", "user.id": "static-user" },
      stepStarted: () => ({
        runtimeContext: { requestId: "request-1", "user.id": "callback-user" },
      }),
    }),
  );

  expect(definition.events?.["step.started"]?.(stepInput({ initiator: "initiator-user" }))).toEqual(
    {
      runtimeContext: {
        requestId: "request-1",
        tenant: "acme",
        "user.id": "callback-user",
      },
    },
  );
});

test("step.started adds the current user id when there is no initiator", () => {
  const definition = telemetryDevInstrumentation(telemetryOptions());

  expect(definition.events?.["step.started"]?.(stepInput({ current: "current-user" }))).toEqual({
    runtimeContext: { "user.id": "current-user" },
  });
});

test("step.started returns undefined when no context is available", () => {
  const definition = telemetryDevInstrumentation(telemetryOptions());

  expect(definition.events?.["step.started"]?.(stepInput())).toBeUndefined();
});

test("step.started swallows user callback errors and reports them", () => {
  const onError = vi.fn();
  const thrown = new Error("callback failed");

  const definition = telemetryDevInstrumentation(
    telemetryOptions({
      onError,
      stepStarted: () => {
        throw thrown;
      },
    }),
  );

  expect(definition.events?.["step.started"]?.(stepInput())).toBeUndefined();
  expect(onError).toHaveBeenCalledWith(thrown);
});

test("step.started swallows callback and onError failures while preserving integration context", () => {
  const definition = telemetryDevInstrumentation(
    telemetryOptions({
      onError: () => {
        throw new Error("onError failed");
      },
      stepStarted: () => {
        throw new Error("callback failed");
      },
    }),
  );

  expect(definition.events?.["step.started"]?.(stepInput({ current: "current-user" }))).toEqual({
    runtimeContext: { "user.id": "current-user" },
  });
});

test("instrumentation setup called twice does not reinitialize the SDK", async () => {
  const first = makeExporters();
  const second = makeExporters();
  telemetryDevInstrumentation(telemetryOptions({ serviceName: "first-service" }), first).setup?.({
    agentName: "first-agent",
  });
  telemetryDevInstrumentation(telemetryOptions({ serviceName: "second-service" }), second).setup?.({
    agentName: "second-agent",
  });

  trace.getTracer("eve").startSpan("after-second-setup").end();
  await flush();

  expect(first.spans.getFinishedSpans()).toHaveLength(1);
  expect(second.spans.getFinishedSpans()).toHaveLength(0);
  expect(first.spans.getFinishedSpans()[0]?.resource.attributes["service.name"]).toBe(
    "first-service",
  );
});

test("hook logs subagent session parent turn id", async () => {
  const exporters = makeExporters();

  await invokeHook(
    event("session.started", {
      invocation: {
        kind: "subagent",
        name: "researcher",
        parentCallId: "call-1",
        parentSessionId: "parent-session",
        parentTurnId: "parent-turn",
      },
      runtime: { agentId: "agent-1", eveVersion: "0.50.0" },
    }),
    exporters,
  );

  const record = onlyLog(exporters);
  expect(record?.attributes).toMatchObject({
    "eve.parent.call_id": "call-1",
    "eve.parent.session_id": "parent-session",
    "eve.parent.turn_id": "parent-turn",
    "eve.subagent.name": "researcher",
  });
});

test("hook logs step.completed with severity, message, event name, base attrs, and usage attrs", async () => {
  const exporters = makeExporters();

  await invokeHook(
    event("step.completed", {
      finishReason: "stop",
      sequence: 4,
      stepIndex: 2,
      turnId: "turn-log",
      usage: { cacheReadTokens: 1, cacheWriteTokens: 2, inputTokens: 9, outputTokens: 10 },
    }),
    exporters,
  );

  const record = onlyLog(exporters);
  expect(record?.severityText).toBe("INFO");
  expect(record?.body).toBe("Step completed (stop): 9 in / 10 out tokens");
  expect(record?.eventName).toBe("step.completed");
  expect(record?.attributes).toMatchObject({
    "eve.channel.kind": "web",
    "eve.step.index": 2,
    "eve.turn.id": "turn-log",
    "eve.turn.sequence": 4,
    "eve.usage.cache_read_tokens": 1,
    "eve.usage.cache_write_tokens": 2,
    "gen_ai.agent.name": "support-agent",
    "gen_ai.conversation.id": "session-log",
    "gen_ai.usage.input_tokens": 9,
    "gen_ai.usage.output_tokens": 10,
  });
});

test("hook logs turn.failed with error attrs and JSON details", async () => {
  const exporters = makeExporters();

  await invokeHook(
    event("turn.failed", {
      code: "MODEL_ERROR",
      details: { retryable: false },
      message: "model exploded",
      sequence: 5,
      turnId: "turn-failed",
    }),
    exporters,
  );

  const record = onlyLog(exporters);
  expect(record?.severityText).toBe("ERROR");
  expect(record?.body).toBe("Turn failed: model exploded");
  expect(record?.eventName).toBe("turn.failed");
  expect(record?.attributes).toMatchObject({
    "error.code": "MODEL_ERROR",
    "eve.error.details": JSON.stringify({ retryable: false }),
    "eve.turn.id": "turn-failed",
    "eve.turn.sequence": 5,
    "gen_ai.agent.name": "support-agent",
    "gen_ai.conversation.id": "session-log",
  });
});

test("hook logs turn.cancelled with turn correlation attributes", async () => {
  const exporters = makeExporters();

  await invokeHook(event("turn.cancelled", { sequence: 6, turnId: "turn-cancelled" }), exporters);

  const record = onlyLog(exporters);
  expect(record.severityText).toBe("WARN");
  expect(record.body).toBe("Turn cancelled");
  expect(record.eventName).toBe("turn.cancelled");
  expect(record.attributes).toMatchObject({
    "eve.turn.id": "turn-cancelled",
    "eve.turn.sequence": 6,
    "gen_ai.agent.name": "support-agent",
    "gen_ai.conversation.id": "session-log",
  });
});

test("hook logs subagent.called with child and tool attrs", async () => {
  const exporters = makeExporters();

  await invokeHook(
    event("subagent.called", {
      callId: "call-1",
      childSessionId: "child-1",
      childStreamPath: "/eve/v1/session/child-1/stream",
      sessionId: "parent-session",
      name: "researcher",
      remote: { url: "https://remote.eve" },
      sequence: 6,
      toolName: "delegate_research",
      turnId: "turn-subagent",
      workflowId: "workflow-1",
    }),
    exporters,
  );

  const record = onlyLog(exporters);
  expect(record?.severityText).toBe("INFO");
  expect(record?.body).toBe("Subagent called: researcher");
  expect(record?.eventName).toBe("subagent.called");
  expect(record?.attributes).toMatchObject({
    "eve.child.session_id": "child-1",
    "eve.remote.url": "https://remote.eve",
    "eve.subagent.name": "researcher",
    "eve.turn.id": "turn-subagent",
    "eve.turn.sequence": 6,
    "eve.workflow.id": "workflow-1",
    "gen_ai.agent.name": "support-agent",
    "gen_ai.conversation.id": "session-log",
    "gen_ai.tool.call.id": "call-1",
    "gen_ai.tool.name": "delegate_research",
  });
});

test("hook logs subagent.completed with subagent name and call id", async () => {
  const exporters = makeExporters();

  await invokeHook(
    event("subagent.completed", {
      callId: "call-1",
      output: "summary",
      subagentName: "researcher",
    }),
    exporters,
  );

  const record = onlyLog(exporters);
  expect(record?.severityText).toBe("INFO");
  expect(record?.body).toBe("Subagent completed");
  expect(record?.eventName).toBe("subagent.completed");
  expect(record?.attributes).toMatchObject({
    "eve.subagent.name": "researcher",
    "gen_ai.tool.call.id": "call-1",
  });
});

test("hook preserves failed subagent action result identity", async () => {
  const exporters = makeExporters();

  await invokeHook(
    event("action.result", {
      error: { code: "SUBAGENT_EXECUTION_FAILED", message: "subagent failed to start" },
      result: {
        callId: "call-subagent",
        isError: true,
        kind: "subagent-result",
        origin: "dispatch",
        output: { code: "SUBAGENT_EXECUTION_FAILED" },
        subagentName: "researcher",
      },
      sequence: 9,
      stepIndex: 3,
      status: "failed",
      turnId: "turn-action",
    }),
    exporters,
  );

  const record = onlyLog(exporters);
  expect(record?.severityText).toBe("ERROR");
  expect(record?.body).toBe("Tool failed: subagent failed to start");
  expect(record?.eventName).toBe("action.result");
  expect(record?.attributes).toMatchObject({
    "error.code": "SUBAGENT_EXECUTION_FAILED",
    "eve.subagent.name": "researcher",
    "gen_ai.tool.call.id": "call-subagent",
    "gen_ai.tool.name": "eve:subagent:researcher",
  });
});

test("hook logs subagent.started and failed child events with subagent attrs", async () => {
  const exporters = makeExporters();
  const hook = telemetryDevHook(telemetryOptions(), exporters);

  await hook.events?.["*"]?.(
    event("subagent.started", { callId: "call-start", subagentName: "researcher" }),
    hookContext(),
  );
  await hook.events?.["*"]?.(
    event("subagent.event", {
      callId: "call-start",
      event: {
        data: {
          code: "CHILD_TURN_FAILED",
          details: { retryable: false },
          message: "child turn broke",
          sequence: 8,
          turnId: "child-turn",
        },
        type: "turn.failed",
      },
      subagentName: "researcher",
    }),
    hookContext(),
  );
  await flush();

  const records = exporters.logs.getFinishedLogRecords();
  expect(records.map((record) => record.eventName)).toEqual(["subagent.started", "subagent.event"]);
  expect(records[0]?.severityText).toBe("INFO");
  expect(records[0]?.body).toBe("Subagent started");
  expect(records[0]?.attributes).toMatchObject({
    "eve.subagent.name": "researcher",
    "gen_ai.tool.call.id": "call-start",
  });
  expect(records[1]?.severityText).toBe("ERROR");
  expect(records[1]?.body).toBe("Subagent turn failed: child turn broke");
  expect(records[1]?.attributes).toMatchObject({
    "error.code": "CHILD_TURN_FAILED",
    "eve.error.details": JSON.stringify({ retryable: false }),
    "eve.subagent.name": "researcher",
    "eve.turn.id": "child-turn",
    "eve.turn.sequence": 8,
    "gen_ai.tool.call.id": "call-start",
  });
});

test("hook ignores message.appended and malformed event data does not throw", async () => {
  const exporters = makeExporters();
  const hook = telemetryDevHook(telemetryOptions(), exporters);

  expect(
    hook.events?.["*"]?.(unsafeEvent("message.appended", "bad-data"), hookContext()),
  ).toBeUndefined();
  await flush();
  expect(exporters.logs.getFinishedLogRecords()).toHaveLength(0);

  expect(hook.events?.["*"]?.(unsafeEvent("turn.failed", null), hookContext())).toBeUndefined();
  await flush();
  expect(onlyLog(exporters).body).toBe("Turn failed: ");
});

test("hook swallows event processing errors even when onError throws", async () => {
  const exporters = makeExporters();
  const thrown = new Error("event data failed");

  const onError = vi.fn(() => {
    throw new Error("onError failed");
  });

  const hook = telemetryDevHook(telemetryOptions({ onError }), exporters);

  const brokenEvent = {
    get data() {
      throw thrown;
    },
    meta: { at: "2026-01-02T03:04:05.000Z" },
    type: "turn.failed",
  } as never;

  await hook.events?.["*"]?.(brokenEvent, hookContext());
  await flush();

  expect(onError).toHaveBeenCalledWith(thrown);
  expect(exporters.logs.getFinishedLogRecords()).toHaveLength(0);
});

test("wrapped eve client streams a successful turn and records usage on the turn span", async () => {
  const exporters = makeExporters();
  stubFetchSequence([
    jsonResponse({ sessionId: "session-1" }),
    ndjsonResponse([
      event("turn.started", {
        sequence: 0,
        turnId: "turn-success",
        trace: {
          traceId: "1382eccf84291f6161a535bdd642e06c",
          spanId: "0123456789abcdef",
          traceFlags: 1,
        },
      }),
      ...successfulTurnEvents(),
    ]),
  ]);
  const client = wrapClient(exporters);

  const { response } = await client.sessions.create({ message: "hello" });
  const seen: string[] = [];

  for await (const item of response) seen.push(item.type);
  await flush();

  expect(seen).toEqual([
    "turn.started",
    "step.completed",
    "step.completed",
    "message.completed",
    "result.completed",
    "session.completed",
  ]);
  const span = onlySpan(exporters);
  expect(span.name).toBe("invoke_agent");
  expect(span.attributes).toMatchObject({
    "td.eve.turn_root": "1382eccf84291f6161a535bdd642e06c/0123456789abcdef",
    "eve.turn.id": "turn-success",
    "gen_ai.agent.name": "wrapped-agent",
    "gen_ai.conversation.id": "session-1",
    "gen_ai.input.messages": "hello",
    "gen_ai.operation.name": "invoke_agent",
    "gen_ai.output.messages": JSON.stringify({ ok: true }),
    "gen_ai.response.finish_reasons": ["stop"],
    "gen_ai.usage.cache_creation.input_tokens": 3,
    "gen_ai.usage.cache_read.input_tokens": 2,
    "gen_ai.usage.input_tokens": 11,
    "gen_ai.usage.output_tokens": 13,
    "gen_ai.usage.cost": 0.03,
  });
});

test.each([
  { name: "missing later cost", costs: [0.01, undefined], expected: undefined },
  { name: "missing earlier cost", costs: [undefined, 0.02], expected: undefined },
  { name: "all known costs", costs: [0.01, 0.02], expected: 0.03 },
  { name: "known zero costs", costs: [0, 0], expected: 0 },
  { name: "null cost", costs: [0.01, null], expected: undefined },
  { name: "string cost", costs: [0.01, "0.02"], expected: undefined },
  { name: "negative cost", costs: [0.01, -0.02], expected: undefined },
  {
    name: "overflowing total",
    costs: [Number.MAX_VALUE, Number.MAX_VALUE],
    expected: undefined,
  },
])("wrapped eve client publishes only complete cost totals: $name", async ({ costs, expected }) => {
  const exporters = makeExporters();
  stubFetchSequence([
    jsonResponse({ sessionId: "session-cost" }),
    ndjsonResponse([
      unsafeEvent("step.completed", {
        finishReason: "tool-calls",
        sequence: 1,
        stepIndex: 0,
        turnId: "turn-cost",
        usage: { costUsd: costs[0], inputTokens: 10, outputTokens: 5 },
      }),
      unsafeEvent("step.completed", {
        finishReason: "stop",
        sequence: 2,
        stepIndex: 1,
        turnId: "turn-cost",
        usage: { costUsd: costs[1], inputTokens: 20, outputTokens: 10 },
      }),
      event("session.completed"),
    ]),
  ]);
  const client = wrapClient(exporters);
  await (await client.sessions.create({ message: "calculate cost" })).response.result();
  await flush();

  const span = onlySpan(exporters);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(30);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(15);
  expect(span.attributes["gen_ai.usage.cost"]).toBe(expected);
});

test("wrapped eve client puts every turn of one session in the session trace", async () => {
  const exporters = makeExporters();
  stubFetchSequence([
    jsonResponse({ sessionId: "session-1" }),
    ndjsonResponse(successfulTurnEvents()),
    jsonResponse({ sessionId: "session-1" }),
    ndjsonResponse(successfulTurnEvents()),
  ]);
  const client = wrapClient(exporters);

  const before = Date.now();
  const { response, session } = await client.sessions.create({ message: "first" });
  await response.result();
  await (await session.send("second")).result();
  await flush();

  // sha256("td_live_test\0session-1"): trace id = digest[0:16], session parent = digest[16:24]
  const expected = { traceId: "1382eccf84291f6161a535bdd642e06c", spanId: "da259c4698ed3d9f" };
  const spans = exporters.spans.getFinishedSpans();
  expect(spans.map((s) => s.name)).toEqual(["invoke_agent", "invoke_agent"]);
  const ms = ([sec, nano]: [number, number]) => sec * 1000 + nano / 1e6;

  for (const span of spans) {
    expect(span.spanContext().traceId).toBe(expected.traceId);
    expect(span.parentSpanContext?.spanId).toBe(expected.spanId);
    expect(span.attributes["gen_ai.conversation.id"]).toBe("session-1");
    // The span keeps the pre-send start time even though it is created after the response.
    expect(ms(span.startTime)).toBeGreaterThanOrEqual(before);
    expect(ms(span.startTime)).toBeLessThanOrEqual(ms(span.endTime));
  }
});

test("wrapped eve client marks streams without terminal events as errored spans", async () => {
  const exporters = makeExporters();
  stubFetchSequence([
    jsonResponse({ sessionId: "session-incomplete" }),
    ndjsonResponse([
      event("message.completed", {
        finishReason: "stop",
        message: "partial answer",
        sequence: 1,
        stepIndex: 0,
        turnId: "turn-incomplete",
      }),
    ]),
  ]);
  const client = wrapClient(exporters);

  const result = await (
    await client.sessions.create({
      message: "hello",
      streamReconnectPolicy: { reconnect: false },
    })
  ).response.result();

  await flush();

  expect(result.status).toBe("completed");
  const span = onlySpan(exporters);
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes["error.type"]).toBe("stream_incomplete");
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["error"]);
});

test("wrapped eve client result() aggregates output, message, input requests, and status", async () => {
  const exporters = makeExporters();

  const inputRequest = {
    action: { callId: "call-approval", input: {}, kind: "tool-call" as const, toolName: "approve" },
    kind: "tool-approval" as const,
    prompt: "Approve?",
    requestId: "approval-1",
  };

  stubFetchSequence([
    jsonResponse({ sessionId: "session-2" }),
    ndjsonResponse([
      event("input.requested", {
        requests: [inputRequest],
        sequence: 1,
        stepIndex: 0,
        turnId: "turn-waiting",
      }),
      event("message.completed", {
        finishReason: "stop",
        message: "needs approval",
        sequence: 2,
        stepIndex: 0,
        turnId: "turn-waiting",
      }),
      event("result.completed", {
        result: { needsApproval: true },
        sequence: 3,
        stepIndex: 0,
        turnId: "turn-waiting",
      }),
      event("session.waiting", { continuationToken: "cont-2", wait: "next-user-message" }),
    ]),
  ]);
  const client = wrapClient(exporters);

  const result = await (
    await client.sessions.create({ message: "approve this" })
  ).response.result();

  await flush();

  expect(result).toMatchObject({
    data: { needsApproval: true },
    inputRequests: [inputRequest],
    message: "needs approval",
    sessionId: "session-2",
    status: "waiting",
  });
  expect(result.events.map((item: MessageStreamEvent) => item.type)).toEqual([
    "input.requested",
    "message.completed",
    "result.completed",
    "session.waiting",
  ]);
  expect(onlySpan(exporters).events.map((item) => item.name)).toEqual(["input.requested"]);
});

test("wrapped eve client marks turn.failed streams as errored spans", async () => {
  const exporters = makeExporters();
  stubFetchSequence([
    jsonResponse({ sessionId: "session-3" }),
    ndjsonResponse([
      event("turn.failed", {
        code: "TURN_FAILED",
        details: { reason: "bad" },
        message: "turn broke",
        sequence: 1,
        turnId: "turn-broken",
      }),
      event("session.failed", {
        code: "SESSION_FAILED",
        message: "session broke",
        sessionId: "session-3",
      }),
    ]),
  ]);
  const client = wrapClient(exporters);

  const result = await (await client.sessions.create({ message: "break" })).response.result();
  await flush();

  expect(result.status).toBe("failed");
  const span = onlySpan(exporters);
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes["error.code"]).toBe("TURN_FAILED");
  expect(span.attributes["error.type"]).toBe("TURN_FAILED");
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["error"]);
});

test("wrapped eve client marks turn.cancelled streams as cancelled spans", async () => {
  const exporters = makeExporters();
  stubFetchSequence([
    jsonResponse({ sessionId: "session-cancelled" }),
    ndjsonResponse([
      event("turn.cancelled", { sequence: 1, turnId: "turn-cancelled" }),
      event("session.waiting", { continuationToken: "cont", wait: "next-user-message" }),
    ]),
  ]);
  const client = wrapClient(exporters);

  const result = await (await client.sessions.create({ message: "cancel" })).response.result();
  await flush();

  expect(result.status).toBe("waiting");
  const span = onlySpan(exporters);
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes["error.type"]).toBe("cancelled");
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["cancelled"]);
});

test("wrapped eve client marks aborted streams as cancelled spans", async () => {
  const exporters = makeExporters();
  const controller = new AbortController();
  const requests: Array<Parameters<typeof fetch>> = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (...args: Parameters<typeof fetch>) => {
      requests.push(args);

      if (requests.length === 1) return jsonResponse({ sessionId: "session-4" });
      controller.abort();
      const abortError = new DOMException("The operation was aborted.", "AbortError");

      return new Response(
        new ReadableStream<Uint8Array>({
          start(streamController) {
            streamController.error(abortError);
          },
        }),
        { headers: { "content-type": "application/x-ndjson; charset=utf-8" } },
      );
    }),
  );
  const client = wrapClient(exporters);

  const result = await (
    await client.sessions.create({ message: "cancel", signal: controller.signal })
  ).response.result();

  expect(result.events).toEqual([]);
  await flush();

  const span = onlySpan(exporters);
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes["error.type"]).toBe("cancelled");
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["cancelled"]);
});

test.each(["create", "send", "respond"] as const)(
  "wrapped eve client closes an unconsumed %s span when its signal aborts",
  async (method) => {
    const exporters = makeExporters();
    const controller = new AbortController();

    const inputRequest = {
      action: { callId: "call-1", input: {}, kind: "tool-call" as const, toolName: "approve" },
      kind: "tool-approval" as const,
      prompt: "Approve?",
      requestId: "approval-1",
    };

    stubFetchSequence([
      ...(method === "create" ? [] : [jsonResponse({ sessionId: "session-abort" })]),
      jsonResponse({ sessionId: "session-abort" }),
    ]);
    const client = wrapClient(exporters);

    if (method === "create") {
      await client.sessions.create({ message: "hello", signal: controller.signal });
    } else {
      const session = client.sessions.attach("session-abort");

      if (method === "send") {
        await session.send("hello", { signal: controller.signal });
      } else {
        await session.respond([inputRequest], { signal: controller.signal });
      }
    }

    controller.abort();
    await flush();

    const span = onlySpan(exporters);
    expect(span.status.code).toBe(SpanStatusCode.ERROR);
    expect(span.attributes["error.type"]).toBe("cancelled");
    expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["cancelled"]);
  },
);

test("wrapped eve client closes an unconsumed span for an already aborted signal", async () => {
  const exporters = makeExporters();
  const controller = new AbortController();
  stubFetchSequence([jsonResponse({ sessionId: "session-abort" })]);
  const client = wrapClient(exporters);
  controller.abort();

  await client.sessions.create({ message: "hello", signal: controller.signal });
  await flush();

  const span = onlySpan(exporters);
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes["error.type"]).toBe("cancelled");
});

test("wrapped eve client keeps completed streams successful when abort races after terminal event", async () => {
  const exporters = makeExporters();
  const controller = new AbortController();
  const requests: Array<Parameters<typeof fetch>> = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (...args: Parameters<typeof fetch>) => {
      requests.push(args);

      if (requests.length === 1) return jsonResponse({ sessionId: "session-4b" });
      controller.abort();

      return ndjsonResponse([event("session.completed")]);
    }),
  );
  const client = wrapClient(exporters);

  const result = await (
    await client.sessions.create({ message: "complete despite abort", signal: controller.signal })
  ).response.result();

  await flush();

  expect(result.status).toBe("completed");
  const span = onlySpan(exporters);
  expect(span.status.code).not.toBe(SpanStatusCode.ERROR);
  expect(span.attributes["error.type"]).toBeUndefined();
  expect(span.attributes["gen_ai.response.finish_reasons"]).toBeUndefined();
});

test("wrapped eve client marks an already aborted POST rejection as a cancelled span", async () => {
  const exporters = makeExporters();
  const controller = new AbortController();
  const abortError = new DOMException("The operation was aborted.", "AbortError");
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => {
      throw abortError;
    }),
  );
  const client = wrapClient(exporters);
  controller.abort();

  await expect(
    client.sessions.create({ message: "cancel before post", signal: controller.signal }),
  ).rejects.toBe(abortError);
  await flush();

  const span = onlySpan(exporters);
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes["error.type"]).toBe("cancelled");
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["cancelled"]);
});

test("wrapped eve client rethrows POST 500 errors and records the failed send", async () => {
  const exporters = makeExporters();
  stubFetchSequence([textResponse("server exploded", 500)]);
  const client = wrapClient(exporters);

  await expect(client.sessions.create({ message: "explode" })).rejects.toMatchObject({
    body: "server exploded",
    name: "ClientError",
    status: 500,
  });
  await flush();

  const span = onlySpan(exporters);
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes["error.type"]).toBe("ClientError");
});

test("wrapped eve client binds private-field methods for health and session state", async () => {
  const exporters = makeExporters();

  const { requests } = stubFetchSequence([
    jsonResponse({ ok: true, status: "ready", workflowId: "wf-1" }),
  ]);

  const client = wrapClient(exporters);

  await expect(client.health()).resolves.toEqual({ ok: true, status: "ready", workflowId: "wf-1" });
  expect(client.sessions.attach("session-9").state).toEqual({
    sessionId: "session-9",
    streamIndex: 0,
  });
  expect(requests.map(([url]) => requestUrl(url))).toEqual(["https://eve.test/eve/v1/health"]);
});
