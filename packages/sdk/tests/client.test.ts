import { ExportResultCode } from "@opentelemetry/core";
import { trace } from "@opentelemetry/api";
import { InMemorySpanExporter, type SpanExporter } from "@opentelemetry/sdk-trace-base";
import { afterEach, expect, test } from "vitest";

import { flush, init, log, observe, shutdown, startSpan } from "../src/index.ts";
import { setup } from "./helpers.ts";

afterEach(async () => {
  await shutdown();
});

test("primitives are safe no-ops before init", async () => {
  const handle = startSpan("orphan");
  expect(handle.isRecording).toBe(false);
  expect(handle.traceparent).toBeNull();
  handle.update({ output: "x" }).end();
  log("nobody listening");
  const fn = observe((a: number) => a * 2);
  expect(fn(21)).toBe(42);
  await expect(flush()).resolves.toBeUndefined();
  await expect(shutdown()).resolves.toBeUndefined();
});

test("enabled:false yields a no-op client even with a key", () => {
  const spans = new InMemorySpanExporter();
  const client = init(
    { apiKey: "td_live_test", enabled: false, logLevel: "silent" },
    { spanExporter: spans },
  );
  expect(client.enabled).toBe(false);
  startSpan("nope").end();
  expect(spans.getFinishedSpans()).toHaveLength(0);
});

test("missing api key disables the client", () => {
  const previous = process.env.TELEMETRY_DEV_API_KEY;
  delete process.env.TELEMETRY_DEV_API_KEY;
  try {
    const client = init({ logLevel: "silent" });
    expect(client.enabled).toBe(false);
    const handle = startSpan("nope");
    expect(handle.isRecording).toBe(false);
  } finally {
    if (previous !== undefined) process.env.TELEMETRY_DEV_API_KEY = previous;
  }
});

test("an exporter override enables the client without an api key", async () => {
  const previous = process.env.TELEMETRY_DEV_API_KEY;
  delete process.env.TELEMETRY_DEV_API_KEY;
  try {
    const spans = new InMemorySpanExporter();
    const client = init({ exportMode: "immediate", logLevel: "silent" }, { spanExporter: spans });
    expect(client.enabled).toBe(true);
    startSpan("works").end();
    await flush();
    expect(spans.getFinishedSpans()).toHaveLength(1);
  } finally {
    if (previous !== undefined) process.env.TELEMETRY_DEV_API_KEY = previous;
  }
});

test("double init replaces the previous client", async () => {
  const first = setup();
  const second = setup();
  startSpan("after-replace").end();
  await flush();
  expect(first.spans.getFinishedSpans()).toHaveLength(0);
  expect(second.spans.getFinishedSpans()).toHaveLength(1);
});

test("flush and shutdown are idempotent", async () => {
  const { client } = setup();
  startSpan("s").end();
  await client.flush();
  await client.flush();
  await client.shutdown();
  await client.shutdown();
  expect(client.enabled).toBe(false);
  // post-shutdown primitives no-op
  const handle = startSpan("late");
  expect(handle.isRecording).toBe(false);
});

test("concurrent flush calls both resolve and export the span exactly once", async () => {
  const { spans } = setup({ exportMode: "batched" });
  startSpan("buffered-once").end();
  await Promise.all([flush(), flush()]);
  expect(spans.getFinishedSpans()).toHaveLength(1);
});

test("waitUntil receives the flush promise instead of awaiting", async () => {
  let resultCallback: Parameters<SpanExporter["export"]>[1] | undefined;
  let resolveExportStarted!: () => void;
  const exportStarted = new Promise<void>((resolve) => {
    resolveExportStarted = resolve;
  });
  const exporter: SpanExporter = {
    export: (_spans, callback) => {
      resultCallback = callback;
      resolveExportStarted();
    },
    shutdown: () => Promise.resolve(),
  };
  let handoff: Promise<unknown> | undefined;
  setup(
    {
      exportMode: "batched",
      waitUntil: (promise) => {
        handoff = promise;
      },
    },
    { spanExporter: exporter },
  );
  startSpan("buffered").end();
  const flushPromise = flush();
  await exportStarted;

  const exportPromise = handoff;
  const callback = resultCallback;
  try {
    expect(exportPromise).toBeDefined();
    expect(callback).toBeDefined();
    if (!exportPromise || !callback) {
      throw new Error("waitUntil did not receive an in-flight export");
    }

    let flushSettled = false;
    void flushPromise.then(() => {
      flushSettled = true;
    });
    let exportSettled = false;
    void exportPromise.then(() => {
      exportSettled = true;
    });
    await Promise.resolve();
    expect(flushSettled).toBe(true);
    expect(exportSettled).toBe(false);
  } finally {
    callback?.({ code: ExportResultCode.SUCCESS });
    await Promise.all([flushPromise, exportPromise ?? Promise.resolve()]);
  }
});

test("registerGlobal exports only this SDK's scope by default and cleans up on shutdown", async () => {
  const { spans, client } = setup({ registerGlobal: true });
  // Foreign instrumentation routed to our (now global) provider is filtered out.
  trace.getTracer("some-http-lib").startSpan("infra-noise").end();
  startSpan("ours").end();
  await flush();
  const exported = spans.getFinishedSpans();
  expect(exported).toHaveLength(1);
  expect(exported[0]!.name).toBe("ours");
  await client.shutdown();
  // After shutdown the global tracer provider registration is released.
  const orphan = trace.getTracer("some-http-lib").startSpan("post-shutdown");
  expect(orphan.isRecording()).toBe(false);
  orphan.end();
});

test("spanFilter overrides the default global filter", async () => {
  const { spans } = setup({ registerGlobal: true, spanFilter: () => true });
  trace.getTracer("some-http-lib").startSpan("kept").end();
  await flush();
  expect(spans.getFinishedSpans()).toHaveLength(1);
});

test("resource carries service name and environment", async () => {
  const { spans } = setup();
  startSpan("s").end();
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.resource.attributes["service.name"]).toBe("svc");
  expect(span.resource.attributes["deployment.environment.name"]).toBe("test");
  expect(span.instrumentationScope.name).toBe("@telemetry-dev/sdk");
});
test("OTLP requests identify the calling SDK package", async () => {
  const requests: Request[] = [];
  init({
    apiKey: "td_live_test",
    baseUrl: "https://ingest.test",
    exportMode: "immediate",
    sdkName: "@telemetry-dev/cursor",
    fetch: async (input, init) => {
      requests.push(new Request(input, init));
      return new Response(null, { status: 200 });
    },
  });

  startSpan("identified").end();
  await flush();

  expect(requests[0]?.headers.get("x-telemetry-dev-sdk")).toBe("@telemetry-dev/cursor");
});
