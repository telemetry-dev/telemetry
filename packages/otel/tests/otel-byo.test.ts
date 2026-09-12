import {
  context,
  createTraceState,
  propagation,
  ROOT_CONTEXT,
  SpanKind,
  trace,
  TraceFlags,
} from "@opentelemetry/api";
import {
  AlwaysOffSampler,
  AlwaysOnSampler,
  BasicTracerProvider,
  InMemorySpanExporter,
  ParentBasedSampler,
  SamplingDecision,
  SimpleSpanProcessor,
  TraceIdRatioBasedSampler,
} from "@opentelemetry/sdk-trace-base";
import { afterEach, expect, test, vi } from "vitest";

import { als, AlsContextManager, propagateAttributes } from "../src/context.ts";
import { createTelemetrySpanExporter, TelemetrySpanProcessor } from "../src/otel.ts";
import {
  sessionRootTracerProvider,
  sessionSampler,
  sessionSpanContext,
  sha256,
  withSessionParent,
} from "../src/session.ts";

afterEach(() => {
  context.disable();
  vi.unstubAllEnvs();
});

function byoSetup(options?: ConstructorParameters<typeof TelemetrySpanProcessor>[0]) {
  const exporter = new InMemorySpanExporter();

  const processor = new TelemetrySpanProcessor({
    exportMode: "immediate",
    spanExporter: exporter,
    ...options,
  });

  const provider = new BasicTracerProvider({
    sampler: sessionSampler(),
    spanProcessors: [processor],
  });

  return { exporter, processor, provider, tracer: provider.getTracer("user-app") };
}

test("exports every span from a user-owned provider by default", async () => {
  const { exporter, processor, tracer } = byoSetup();
  tracer.startSpan("a").end();
  tracer.startSpan("b").end();
  await processor.forceFlush();
  expect(exporter.getFinishedSpans()).toHaveLength(2);
});

test("stamps propagated attributes when a global context manager is present", async () => {
  // BYO hosts (NodeSDK, @vercel/otel) register a real context manager; simulate that.
  expect(als).toBeDefined();
  context.setGlobalContextManager(new AlsContextManager(als!));
  const { exporter, processor, tracer } = byoSetup();
  propagateAttributes({ userId: "u_byo", sessionId: "conv_byo" }, () => {
    tracer.startSpan("instrumented").end();
  });
  await processor.forceFlush();
  const span = exporter.getFinishedSpans()[0]!;
  expect(span.attributes["user.id"]).toBe("u_byo");
  expect(span.attributes["gen_ai.conversation.id"]).toBe("conv_byo");
});

test("stamps propagated attributes with explicit ROOT_CONTEXT and no global context manager", async () => {
  context.disable();
  const { exporter, processor, tracer } = byoSetup();

  propagateAttributes({ userId: "u_local", sessionId: "conv_local" }, () => {
    tracer.startSpan("instrumented", undefined, ROOT_CONTEXT).end();
  });

  await processor.forceFlush();
  const span = exporter.getFinishedSpans()[0]!;
  expect(span.attributes["user.id"]).toBe("u_local");
  expect(span.attributes["gen_ai.conversation.id"]).toBe("conv_local");
});

test("spanFilter narrows what gets exported", async () => {
  const { exporter, processor, tracer } = byoSetup({
    spanFilter: (span) => span.name.startsWith("keep"),
  });

  tracer.startSpan("keep-me").end();
  tracer.startSpan("drop-me").end();
  await processor.forceFlush();
  expect(exporter.getFinishedSpans().map((s) => s.name)).toEqual(["keep-me"]);
});

test("without an api key or exporter override the processor is a silent no-op", async () => {
  const previous = process.env.TELEMETRY_DEV_API_KEY;
  delete process.env.TELEMETRY_DEV_API_KEY;

  try {
    const processor = new TelemetrySpanProcessor();
    const provider = new BasicTracerProvider({ spanProcessors: [processor] });
    provider.getTracer("x").startSpan("lost").end();
    await processor.forceFlush();
    await processor.shutdown();
  } finally {
    if (previous !== undefined) process.env.TELEMETRY_DEV_API_KEY = previous;
  }
});

test("forceFlush and shutdown resolve", async () => {
  const { processor } = byoSetup();
  await expect(processor.forceFlush()).resolves.toBeUndefined();
  await expect(processor.shutdown()).resolves.toBeUndefined();
});

test("createTelemetrySpanExporter is a no-op without a key and real otherwise", async () => {
  const previous = process.env.TELEMETRY_DEV_API_KEY;
  delete process.env.TELEMETRY_DEV_API_KEY;

  try {
    const noop = createTelemetrySpanExporter();

    const code = await new Promise<number>((resolve) =>
      noop.export([], (result) => resolve(result.code)),
    );

    expect(code).toBe(0);
    await noop.shutdown();

    const calls: Array<{ url: string; method?: string }> = [];

    const fetchImpl: typeof fetch = (input, init) => {
      calls.push({
        url: input instanceof URL ? input.href : input instanceof Request ? input.url : input,
        method: init?.method,
      });

      return Promise.resolve(new Response(null, { status: 200 }));
    };

    const exporter = createTelemetrySpanExporter({
      apiKey: "td_live_x",
      baseUrl: "https://ingest.example/",
      fetch: fetchImpl,
    });

    const provider = new BasicTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });

    try {
      provider.getTracer("public-exporter").startSpan("exported").end();
      await provider.forceFlush();
      expect(calls).toEqual([{ url: "https://ingest.example/v1/traces", method: "POST" }]);
    } finally {
      await provider.shutdown();
    }
  } finally {
    if (previous !== undefined) process.env.TELEMETRY_DEV_API_KEY = previous;
  }
});

test("createTelemetrySpanExporter honors a timeout beyond the default", async () => {
  vi.useFakeTimers();

  try {
    let signal: AbortSignal | undefined;
    const results: number[] = [];
    const source = new InMemorySpanExporter();

    const provider = new BasicTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(source)],
    });

    provider.getTracer("timeout-source").startSpan("exported").end();

    const exporter = createTelemetrySpanExporter({
      apiKey: "td_live_x",
      exportTimeoutMillis: 35_000,
      fetch: (_input, init) => {
        signal = init!.signal!;

        return new Promise(() => undefined);
      },
    });

    exporter.export(source.getFinishedSpans(), (result) => results.push(result.code));
    await Promise.resolve();
    await Promise.resolve();

    await vi.advanceTimersByTimeAsync(30_000);
    expect(results).toEqual([]);
    expect(signal!.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(5_000);
    expect(results).toEqual([1]);
    expect(signal!.aborted).toBe(true);
    await exporter.shutdown();
    await provider.shutdown();
  } finally {
    vi.useRealTimers();
  }
});

test("metrics are skipped without an api key even when enabled", async () => {
  const previous = process.env.TELEMETRY_DEV_API_KEY;
  delete process.env.TELEMETRY_DEV_API_KEY;

  try {
    const calls: string[] = [];

    const fetchImpl: typeof fetch = (input) => {
      calls.push(input instanceof URL ? input.href : input instanceof Request ? input.url : input);

      return Promise.resolve(new Response(null, { status: 200 }));
    };

    const exporter = new InMemorySpanExporter();

    const processor = new TelemetrySpanProcessor({
      spanExporter: exporter,
      metrics: true,
      fetch: fetchImpl,
    });

    const provider = new BasicTracerProvider({ spanProcessors: [processor] });

    try {
      const span = provider.getTracer("x").startSpan("chat");
      span.setAttribute("gen_ai.operation.name", "chat");
      span.end();
      await provider.forceFlush();
      expect(exporter.getFinishedSpans().map((finished) => finished.name)).toEqual(["chat"]);
      expect(calls).toHaveLength(0);
    } finally {
      await provider.shutdown();
    }
  } finally {
    if (previous !== undefined) process.env.TELEMETRY_DEV_API_KEY = previous;
  }
});

test("global tracer interop: registered provider receives raw OTel API spans", async () => {
  const { exporter, processor, provider } = byoSetup();
  trace.setGlobalTracerProvider(provider);

  try {
    trace.getTracer("third-party").startSpan("via-global").end();
    await processor.forceFlush();
    expect(exporter.getFinishedSpans().map((s) => s.name)).toEqual(["via-global"]);
  } finally {
    trace.disable();
  }
});

test("sha256 matches the FIPS vector", () => {
  const hex = Array.from(sha256(new TextEncoder().encode("abc")), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");

  expect(hex).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
});

test("sessionSpanContext is deterministic per (apiKey, sessionId) and pinned to Python", () => {
  const a = sessionSpanContext("td_live_k", "s1");
  // python -c "import hashlib;print(hashlib.sha256(b'td_live_k\x00s1').hexdigest()[:48])"
  expect(a.traceId + a.spanId).toBe("9e21d32713f05502cf537dba02d3d6fee1dbbf7220b0fab7");
  expect(sessionSpanContext("td_live_k", "s2").traceId).not.toBe(a.traceId);
  expect(sessionSpanContext("other", "s1").traceId).not.toBe(a.traceId);
  expect(a.isRemote).toBe(true);
});

test("sessionSpanContext replaces lone surrogates like TextEncoder and Python", () => {
  const ctx = sessionSpanContext("td_live_k", "surrogate-\ud800");
  expect(ctx.traceId + ctx.spanId).toBe("9df9d5baf81b578cec335645f2d63a8e345189a4d765e6d5");
  const pair = sessionSpanContext("td_live_k", "pair-\ud83d\ude00");
  expect(pair.traceId + pair.spanId).toBe("76b57d1fe53960b397a8f94a09fdb9a57ac606d0c477796b");
});

test("withSessionParent keeps the context without a key or session id", () => {
  expect(withSessionParent(ROOT_CONTEXT, "s1", undefined)).toBe(ROOT_CONTEXT);
  expect(withSessionParent(ROOT_CONTEXT, "s1", "")).toBe(ROOT_CONTEXT);
  expect(withSessionParent(ROOT_CONTEXT, "", "td_live_k")).toBe(ROOT_CONTEXT);
});

test("sessionRootTracerProvider keeps roots for missing inputs and callback errors", async () => {
  const { exporter, processor, provider } = byoSetup();
  const errors: Error[] = [];
  const noKey = sessionRootTracerProvider(provider, undefined, () => "s1");
  const noSession = sessionRootTracerProvider(provider, "td_live_k", () => "");

  const broken = sessionRootTracerProvider(
    provider,
    "td_live_k",
    () => {
      throw new Error("session root failed");
    },
    (error) => errors.push(error),
  );

  noKey.getTracer("test").startSpan("no-key").end();
  noSession.getTracer("test").startSpan("no-session").end();
  broken.getTracer("test").startSpan("broken").end();
  broken.getTracer("test").startActiveSpan("active-broken", (span) => span.end());
  await processor.forceFlush();

  const finished = exporter.getFinishedSpans();
  expect(finished.map((span) => span.name)).toEqual([
    "no-key",
    "no-session",
    "broken",
    "active-broken",
  ]);

  for (const span of finished) expect(span.parentSpanContext).toBeUndefined();
  expect(errors.map((error) => error.message)).toEqual([
    "session root failed",
    "session root failed",
  ]);
});

test("session roots use the BYO root sampler, not the synthetic parent flags", async () => {
  for (const root of [
    new AlwaysOffSampler(),
    new TraceIdRatioBasedSampler(0.5),
    new AlwaysOnSampler(),
  ]) {
    const policy = new ParentBasedSampler({ root });
    const provider = new BasicTracerProvider({ sampler: sessionSampler(policy) });

    try {
      const tracer = provider.getTracer("byo");

      for (let i = 0; i < 20; i++) {
        const parent = withSessionParent(ROOT_CONTEXT, `session-${i}`, "td_live_k");
        const id = trace.getSpanContext(parent)!.traceId;

        const expected = policy.shouldSample(
          ROOT_CONTEXT,
          id,
          "turn",
          SpanKind.INTERNAL,
          {},
          [],
        ).decision;

        const span = tracer.startSpan("turn", {}, parent);
        expect(span.isRecording()).toBe(expected !== SamplingDecision.NOT_RECORD);
        expect(span.spanContext().traceFlags & TraceFlags.SAMPLED).toBe(
          expected === SamplingDecision.RECORD_AND_SAMPLED ? TraceFlags.SAMPLED : TraceFlags.NONE,
        );
        expect(span.spanContext().traceId).toBe(id);
        span.end();
      }
    } finally {
      await provider.shutdown();
    }
  }
});

test("session roots and children keep real parent sampling, tracestate, and baggage", async () => {
  context.setGlobalContextManager(new AlsContextManager(als!));

  const provider = new BasicTracerProvider({
    sampler: sessionSampler(new ParentBasedSampler({ root: new AlwaysOffSampler() })),
  });

  const tracer = sessionRootTracerProvider(provider, "td_live_k", (name) =>
    name === "turn" ? "s1" : undefined,
  ).getTracer("framework");

  const baggage = propagation.createBaggage({ tenant: { value: "kept" } });
  const traceState = createTraceState("vendor=kept");

  try {
    for (const isRemote of [false, true]) {
      for (const traceFlags of [TraceFlags.NONE, TraceFlags.SAMPLED]) {
        const parent = {
          traceId: "12345678901234567890123456789012",
          spanId: "1234567890123456",
          traceFlags,
          isRemote,
          traceState,
        };

        const ctx = propagation.setBaggage(trace.setSpanContext(ROOT_CONTEXT, parent), baggage);
        expect(withSessionParent(ctx, "s1", "td_live_k")).toBe(ctx);
        context.with(ctx, () =>
          tracer.startActiveSpan("turn", (span) => {
            expect(span.isRecording()).toBe(traceFlags === TraceFlags.SAMPLED);
            expect(span.spanContext().traceId).toBe("9e21d32713f05502cf537dba02d3d6fe");
            expect(span.spanContext().traceState?.serialize()).toBe("vendor=kept");
            expect(propagation.getBaggage(context.active())).toBe(baggage);
            const child = tracer.startSpan("child");
            expect(child.isRecording()).toBe(span.isRecording());
            expect(child.spanContext().traceId).toBe(span.spanContext().traceId);
            child.end();
            span.end();
          }),
        );
        tracer.startActiveSpan("turn", {}, ctx, (span) => {
          expect(span.isRecording()).toBe(traceFlags === TraceFlags.SAMPLED);
          span.end();
        });
        context.with(ctx, () =>
          tracer.startActiveSpan("turn", {}, (span) => {
            expect(span.isRecording()).toBe(traceFlags === TraceFlags.SAMPLED);
            span.end();
          }),
        );
        const root = tracer.startSpan("turn", { root: true }, ctx);
        expect(root.isRecording()).toBe(false);
        expect(root.spanContext().traceId).toBe("9e21d32713f05502cf537dba02d3d6fe");
        root.end();
      }
    }
  } finally {
    await provider.shutdown();
  }
});

test("session sampling keeps baggage changed after attaching the session parent", async () => {
  const exporter = new InMemorySpanExporter();

  const provider = new BasicTracerProvider({
    sampler: sessionSampler(
      new ParentBasedSampler({
        root: {
          shouldSample(ctx) {
            return {
              decision:
                propagation.getBaggage(ctx)?.getEntry("tenant")?.value === "allowed"
                  ? SamplingDecision.RECORD_AND_SAMPLED
                  : SamplingDecision.NOT_RECORD,
            };
          },
          toString: () => "TenantSampler",
        },
      }),
    ),
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });

  try {
    const tracer = provider.getTracer("byo");

    const before = propagation.setBaggage(
      ROOT_CONTEXT,
      propagation.createBaggage({
        tenant: { value: "denied" },
      }),
    );

    const parent = withSessionParent(before, "s1", "td_live_k");

    const allowed = propagation.setBaggage(
      parent,
      propagation.createBaggage({
        tenant: { value: "allowed" },
      }),
    );

    tracer.startSpan("allowed", {}, allowed).end();
    tracer.startSpan("denied", {}, parent).end();
    tracer.startSpan("removed", {}, propagation.deleteBaggage(allowed)).end();
    await provider.forceFlush();
    expect(exporter.getFinishedSpans().map((span) => span.name)).toEqual(["allowed"]);
  } finally {
    await provider.shutdown();
  }
});

test("session samplers keep record-only results, attributes, and tracestate", async () => {
  const provider = new BasicTracerProvider({
    sampler: sessionSampler({
      shouldSample: () => ({
        decision: SamplingDecision.RECORD,
        attributes: { "sampling.policy": "record-only" },
        traceState: createTraceState("vendor=record"),
      }),
      toString: () => "RecordOnly",
    }),
    spanProcessors: [
      {
        onStart(span) {
          expect(span.attributes["sampling.policy"]).toBe("record-only");
        },
        onEnd() {},
        forceFlush: async () => {},
        shutdown: async () => {},
      },
    ],
  });

  try {
    const span = provider
      .getTracer("byo")
      .startSpan("turn", {}, withSessionParent(ROOT_CONTEXT, "s1", "td_live_k"));

    expect(span.isRecording()).toBe(true);
    expect(span.spanContext().traceFlags & TraceFlags.SAMPLED).toBe(TraceFlags.NONE);
    expect(span.spanContext().traceState?.serialize()).toBe("vendor=record");
    span.end();
  } finally {
    await provider.shutdown();
  }
});

test.each([
  { mode: undefined, arg: undefined, root: true, sampled: true, unsampled: false },
  { mode: "always_on", arg: undefined, root: true, sampled: true, unsampled: true },
  { mode: "always_off", arg: undefined, root: false, sampled: false, unsampled: false },
  { mode: "traceidratio", arg: "0", root: false, sampled: false, unsampled: false },
  { mode: "parentbased_always_on", arg: undefined, root: true, sampled: true, unsampled: false },
  { mode: "parentbased_always_off", arg: undefined, root: false, sampled: true, unsampled: false },
  { mode: "parentbased_traceidratio", arg: "0", root: false, sampled: true, unsampled: false },
  { mode: "unknown", arg: undefined, root: true, sampled: true, unsampled: false },
])("session roots obey environment mode $mode", async ({ mode, arg, root, sampled, unsampled }) => {
  vi.stubEnv("OTEL_TRACES_SAMPLER", mode);
  vi.stubEnv("OTEL_TRACES_SAMPLER_ARG", arg);
  const provider = new BasicTracerProvider({ sampler: sessionSampler() });

  try {
    const tracer = provider.getTracer("byo");
    const ordinary = tracer.startSpan("ordinary", {}, ROOT_CONTEXT);
    expect(ordinary.isRecording()).toBe(root);
    ordinary.end();
    const span = tracer.startSpan("turn", {}, withSessionParent(ROOT_CONTEXT, "s1", "td_live_k"));
    expect(span.isRecording()).toBe(root);
    expect(span.spanContext().traceId).toBe("9e21d32713f05502cf537dba02d3d6fe");
    span.end();

    for (const [traceFlags, expected] of [
      [TraceFlags.NONE, unsampled],
      [TraceFlags.SAMPLED, sampled],
    ] as const) {
      const ctx = trace.setSpanContext(ROOT_CONTEXT, {
        traceId: "12345678901234567890123456789012",
        spanId: "1234567890123456",
        traceFlags,
        isRemote: true,
      });

      const child = tracer.startSpan("child", {}, withSessionParent(ctx, "s1", "td_live_k"));
      expect(child.isRecording()).toBe(expected);
      child.end();
    }
  } finally {
    await provider.shutdown();
  }
});

test.each([undefined, "", "NaN", "Infinity", "-0.1", "1.1"])(
  "invalid ratio argument %s keeps the OTel probability-one default",
  async (arg) => {
    vi.stubEnv("OTEL_TRACES_SAMPLER", "parentbased_traceidratio");
    vi.stubEnv("OTEL_TRACES_SAMPLER_ARG", arg);
    const provider = new BasicTracerProvider({ sampler: sessionSampler() });

    try {
      const span = provider
        .getTracer("byo")
        .startSpan("turn", {}, withSessionParent(ROOT_CONTEXT, "s1", "td_live_k"));

      expect(span.isRecording()).toBe(true);
      span.end();
    } finally {
      await provider.shutdown();
    }
  },
);
