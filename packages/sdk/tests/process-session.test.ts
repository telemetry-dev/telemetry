import { TraceFlags } from "@opentelemetry/api";
import { afterEach, expect, test } from "vitest";

import {
  flush,
  init,
  log,
  propagateAttributes,
  shutdown,
  startActiveSpan,
  startSpan,
} from "../src/index.ts";
import { setup } from "./helpers.ts";

afterEach(async () => shutdown());

test("process mode groups otherwise independent roots and standalone logs", async () => {
  const { spans, logs } = setup({ sessionMode: "process" });
  startSpan("a").end();
  startSpan("b").end();
  log("outside");
  await flush();
  const [a, b] = spans.getFinishedSpans();
  const sessionId = a!.attributes["gen_ai.conversation.id"];
  expect(sessionId).toMatch(
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
  );
  expect(b!.spanContext().traceId).toBe(a!.spanContext().traceId);
  expect(b!.attributes["gen_ai.conversation.id"]).toBe(sessionId);
  expect(logs.getFinishedLogRecords()[0]!.attributes["gen_ai.conversation.id"]).toBe(sessionId);
});

test("explicit sessions win and remain consistent on nested spans and logs", async () => {
  const { spans, logs } = setup({ sessionMode: "process" });
  await Promise.all([
    Promise.resolve().then(() =>
      propagateAttributes({ sessionId: "one" }, () =>
        startActiveSpan("one-root", () => {
          startSpan("one-child").end();
          log("one-log");
        }),
      ),
    ),
    Promise.resolve().then(() =>
      propagateAttributes({ sessionId: "two" }, () =>
        startActiveSpan("two-root", () => {
          startSpan("two-child").end();
          log("two-log");
        }),
      ),
    ),
  ]);
  startActiveSpan(
    "attribute-root",
    { attributes: { "gen_ai.conversation.id": "attribute" } },
    () => {
      startSpan("attribute-child").end();
      log("attribute-log");
    },
  );
  await flush();
  for (const span of spans.getFinishedSpans()) {
    expect(span.attributes["gen_ai.conversation.id"]).toBe(span.name.split("-")[0]);
  }
  for (const record of logs.getFinishedLogRecords()) {
    expect(typeof record.body).toBe("string");
    expect(record.attributes["gen_ai.conversation.id"]).toBe((record.body as string).split("-")[0]);
  }
});

test("process mode honors an existing remote parent", async () => {
  const { spans } = setup({ sessionMode: "process" });
  const parent = {
    traceId: "0123456789abcdef0123456789abcdef",
    spanId: "0123456789abcdef",
    traceFlags: TraceFlags.SAMPLED,
    isRemote: true,
  };
  startSpan("joined", { parent }).end();
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.spanContext().traceId).toBe(parent.traceId);
  expect(span.parentSpanContext?.spanId).toBe(parent.spanId);
});

test("reinitializing process mode generates a new session", async () => {
  const first = setup({ sessionMode: "process" });
  startSpan("first").end();
  await flush();
  const firstId = first.spans.getFinishedSpans()[0]!.attributes["gen_ai.conversation.id"];
  await shutdown();
  const second = setup({ sessionMode: "process" });
  startSpan("second").end();
  await flush();
  expect(second.spans.getFinishedSpans()[0]!.attributes["gen_ai.conversation.id"]).not.toBe(
    firstId,
  );
});

test("invalid mode fails open through onError", () => {
  const errors: Error[] = [];
  const client = init({
    apiKey: "td_live_test",
    sessionMode: "invalid" as "process",
    onError: (error) => errors.push(error),
    logLevel: "silent",
  });
  expect(client.enabled).toBe(false);
  expect(errors[0]).toBeInstanceOf(TypeError);
});
