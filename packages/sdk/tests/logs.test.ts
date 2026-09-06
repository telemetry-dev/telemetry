import { afterEach, expect, test } from "vitest";

import { flush, log, shutdown, startActiveSpan } from "../src/index.ts";
import { setup } from "./helpers.ts";

afterEach(async () => {
  await shutdown();
});

test("severity mapping matches the ingest buckets", async () => {
  const { logs } = setup();
  log("d", { level: "debug" });
  log("i");
  log("w", { level: "warn" });
  log("e", { level: "error" });
  await flush();
  const records = logs.getFinishedLogRecords();
  expect(records.map((r) => [r.severityNumber, r.severityText, r.body])).toEqual([
    [5, "DEBUG", "d"],
    [9, "INFO", "i"],
    [13, "WARN", "w"],
    [17, "ERROR", "e"],
  ]);
});

test("logs inside a span are trace-correlated; outside they are standalone", async () => {
  const { logs } = setup();
  let traceId = "";
  let spanId = "";
  startActiveSpan("op", (span) => {
    traceId = span.traceId;
    spanId = span.spanId;
    log("inside");
  });
  log("outside");
  await flush();
  const records = logs.getFinishedLogRecords();
  const inside = records.find((r) => r.body === "inside")!;
  expect(inside.spanContext?.traceId).toBe(traceId);
  expect(inside.spanContext?.spanId).toBe(spanId);
  const outside = records.find((r) => r.body === "outside")!;
  expect(outside.spanContext?.traceId).toBeUndefined();
});

test("eventName, scalar attributes, and structured attribute stringification", async () => {
  const { logs } = setup();
  log("checkout completed", {
    eventName: "checkout.completed",
    attributes: { items: 3, fast: true, cart: { id: "c1" }, label: "primary" },
  });
  await flush();
  const record = logs.getFinishedLogRecords()[0]!;
  expect(record.eventName).toBe("checkout.completed");
  expect(record.attributes.items).toBe(3);
  expect(record.attributes.fast).toBe(true);
  expect(record.attributes.cart).toBe(JSON.stringify({ id: "c1" }));
  expect(record.attributes.label).toBe("primary");
});

test("boxed number and boolean attributes are serialized instead of treated as primitives", async () => {
  const { logs } = setup();
  log("boxed attributes", {
    attributes: {
      primitiveNumber: 3,
      primitiveBoolean: true,
      boxedNumber: new Number(3),
      boxedBoolean: new Boolean(true),
    } as never,
  });
  await flush();
  const record = logs.getFinishedLogRecords()[0]!;
  expect(record.attributes.primitiveNumber).toBe(3);
  expect(record.attributes.primitiveBoolean).toBe(true);
  expect(record.attributes.boxedNumber).toBe("3");
  expect(record.attributes.boxedBoolean).toBe("true");
});

test("the message passes through the mask hook", async () => {
  const { logs } = setup({
    mask: (value, ctx) => (ctx.key === "log.message" ? "[redacted]" : value),
  });
  log("user email is foo@bar.com");
  await flush();
  expect(logs.getFinishedLogRecords()[0]!.body).toBe("[redacted]");
});

test("log records carry the resource", async () => {
  const { logs } = setup();
  log("hello");
  await flush();
  const record = logs.getFinishedLogRecords()[0]!;
  expect(record.resource.attributes["service.name"]).toBe("svc");
  expect(record.resource.attributes["deployment.environment.name"]).toBe("test");
});
