import { createServer, type RequestListener } from "node:http";
import type { AddressInfo } from "node:net";

import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { score, ScoreError, type ScoreOptions, shutdown } from "../src/index.ts";
import { setup } from "./helpers.ts";

const TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736";
const SPAN_ID = "00f067aa0ba902b7";

function fakeFetch(reply: () => Response) {
  const calls: { url: URL; init: RequestInit | undefined }[] = [];

  const fetch = vi.fn(async (input: string | URL, init?: RequestInit) => {
    calls.push({ url: new URL(input), init });

    return reply();
  });

  return { fetch: fetch as unknown as typeof globalThis.fetch, calls };
}

async function serve(handler: RequestListener): Promise<{ url: string; close: () => void }> {
  const server = createServer(handler);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;

  return {
    url: `http://127.0.0.1:${port}`,
    close: () => {
      server.closeAllConnections();
      server.close();
    },
  };
}

function errorsOf() {
  const errors: ScoreError[] = [];
  setup({ onError: (error) => errors.push(error as ScoreError) });

  return errors;
}

beforeEach(() => {
  vi.stubEnv("TELEMETRY_DEV_API_KEY", undefined);
  vi.stubEnv("TELEMETRY_DEV_BASE_URL", undefined);
});

afterEach(async () => {
  await shutdown();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

test("posts the score with the key and base URL of init, and resolves with its id", async () => {
  const { fetch, calls } = fakeFetch(() => Response.json({ id: "score-1" }, { status: 202 }));
  setup({ apiKey: "td_live_key", baseUrl: "https://ingest.test/", fetch });

  const id = await score({
    name: " user_feedback ",
    traceId: TRACE_ID,
    spanId: SPAN_ID,
    value: 0,
    comment: "Wrong refund amount",
  });

  expect(id).toBe("score-1");
  expect(calls).toHaveLength(1);
  expect(calls[0]!.url.href).toBe("https://ingest.test/v1/scores");
  expect(calls[0]!.init).toMatchObject({
    method: "POST",
    headers: {
      authorization: "Bearer td_live_key",
      "x-telemetry-dev-sdk": "@telemetry-dev/sdk",
      "content-type": "application/json",
    },
  });
  expect(JSON.parse(calls[0]!.init!.body as string)).toEqual({
    name: "user_feedback",
    traceId: TRACE_ID,
    spanId: SPAN_ID,
    value: 0,
    comment: "Wrong refund amount",
  });
});

test("calls the global fetch as a plain function, as Workers and browsers need", async () => {
  vi.stubEnv("TELEMETRY_DEV_API_KEY", "td_live_key");
  // A strict function called as a method of an object gets that object as `this`.
  vi.stubGlobal("fetch", function (this: unknown) {
    if (this !== undefined && this !== globalThis) throw new TypeError("Illegal invocation");

    return Promise.resolve(Response.json({ id: "score-3" }, { status: 202 }));
  });

  try {
    await expect(score({ name: "helpful", traceId: TRACE_ID, value: 1 })).resolves.toBe("score-3");
  } finally {
    vi.unstubAllGlobals();
  }
});

test("score with no options reports invalid_score and resolves undefined", async () => {
  const errors = errorsOf();

  await expect((score as (options?: ScoreOptions) => Promise<unknown>)()).resolves.toBeUndefined();
  expect(errors.map((error) => [error.code, error.message])).toEqual([
    ["invalid_score", "name must be a string of 1 to 64 characters"],
  ]);
});

test("a session score needs no trace, and a label goes trimmed", async () => {
  const { fetch, calls } = fakeFetch(() => Response.json({ id: "score-2" }, { status: 202 }));

  await expect(
    score({ apiKey: "td_live_key", fetch, name: "resolved", sessionId: "chat-9", label: " yes " }),
  ).resolves.toBe("score-2");
  expect(JSON.parse(calls[0]!.init!.body as string)).toEqual({
    name: "resolved",
    sessionId: "chat-9",
    label: "yes",
  });
});

test.each<[string, Partial<ScoreOptions>, string]>([
  ["no trace and no session", { traceId: undefined }, "Pass traceId or sessionId"],
  [
    "the all-zero trace id of a span that did not record",
    { traceId: "0".repeat(32) },
    "traceId must be 32 lowercase hex digits and not all zeros",
  ],
  [
    "an uppercase trace id",
    { traceId: TRACE_ID.toUpperCase() },
    "traceId must be 32 lowercase hex digits and not all zeros",
  ],
  [
    "a span without its trace",
    { traceId: undefined, sessionId: "s", spanId: SPAN_ID },
    "Pass the traceId of the span",
  ],
  ["no value and no label", { value: undefined }, "Pass a value or a label"],
  ["a value that is not finite", { value: Number.NaN }, "value must be a finite number"],
  [
    "a label for user feedback",
    { name: "user_feedback", value: 1, label: "up" },
    "user_feedback takes the value 1 for up or 0 for down, and no label",
  ],
  [
    "user feedback of 0.5",
    { name: "user_feedback", value: 0.5 },
    "user_feedback takes the value 1 for up or 0 for down, and no label",
  ],
  ["a pass of 0", { value: 0, label: "pass" }, "The label pass goes with the value 1"],
  ["a fail of 1", { value: 1, label: "fail" }, "The label fail goes with the value 0"],
  ["a blank name", { name: "  " }, "name must be a string of 1 to 64 characters"],
  [
    "a comment of 4,001 characters",
    { comment: "x".repeat(4_001) },
    "comment must be a string of at most 4000 characters",
  ],
])("reports invalid_score for %s and sends nothing", async (_, options, message) => {
  const errors = errorsOf();
  const { fetch, calls } = fakeFetch(() => Response.json({ id: "never" }));

  await expect(
    score({
      apiKey: "td_live_key",
      fetch,
      name: "helpful",
      traceId: TRACE_ID,
      value: 1,
      ...options,
    }),
  ).resolves.toBeUndefined();
  expect(calls).toEqual([]);
  expect(errors.map((error) => [error.code, error.message])).toEqual([["invalid_score", message]]);
});

test.each([
  [
    "the API error",
    () =>
      Response.json(
        { error: { code: "insufficient_scope", message: "This API key cannot send scores" } },
        { status: 403 },
      ),
    ["insufficient_scope", 403],
  ],
  ["a status with no API error", () => new Response("down", { status: 503 }), ["http_503", 503]],
  ["a body with no id", () => Response.json({}, { status: 202 }), ["invalid_response", undefined]],
])("reports %s, tries once, and resolves undefined", async (_, reply, [code, status]) => {
  const errors = errorsOf();
  const { fetch, calls } = fakeFetch(reply);

  await expect(
    score({ apiKey: "td_live_key", fetch, name: "helpful", traceId: TRACE_ID, value: 1 }),
  ).resolves.toBeUndefined();
  expect(calls).toHaveLength(1);
  expect(errors.map((error) => [error.code, error.status])).toEqual([[code, status]]);
});

test("without an API key it reports missing_api_key", async () => {
  const errors = errorsOf();
  const { fetch, calls } = fakeFetch(() => Response.json({ id: "never" }));

  await score({ apiKey: "", fetch, name: "helpful", traceId: TRACE_ID, value: 1 });

  expect(calls).toEqual([]);
  expect(errors.map((error) => error.code)).toEqual(["missing_api_key"]);
});

test("does not follow a redirect, so the key stays with the API host", async () => {
  const errors = errorsOf();
  let reached = 0;
  const other = await serve((_, res) => {
    reached += 1;
    res.end(JSON.stringify({ id: "stolen" }));
  });
  const api = await serve((_, res) => {
    res.writeHead(307, { location: `${other.url}/v1/scores` });
    res.end();
  });

  try {
    await expect(
      // The real fetch: setup() gives the client a stub.
      score({
        apiKey: "td_live_key",
        baseUrl: api.url,
        fetch: globalThis.fetch,
        name: "helpful",
        traceId: TRACE_ID,
        value: 1,
      }),
    ).resolves.toBeUndefined();
  } finally {
    api.close();
    other.close();
  }

  expect(reached).toBe(0);
  expect(errors.map((error) => [error.code, error.status])).toEqual([["http_307", 307]]);
});
