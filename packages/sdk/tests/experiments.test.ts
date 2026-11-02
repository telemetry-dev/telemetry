import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, expect, test, vi } from "vitest";

import {
  type ExperimentItem,
  ExperimentError,
  log,
  propagateAttributes,
  runExperiment,
  type RunExperimentOptions,
  shutdown,
  startActiveSpan,
  startSpan,
} from "../src/index.ts";
import { setup } from "./helpers.ts";

interface Request {
  method: string;
  url: URL;
  body: any;
  signal: AbortSignal | undefined;
}

const EXPERIMENT_ID = "6f9619ff-8b86-4011-b42d-00c04fc964ff";
const URL_OF_EXPERIMENT = `https://telemetry.dev/o/org/p/project/evals/experiments/${EXPERIMENT_ID}`;

function itemId(number: number): string {
  return `00000000-0000-4000-8000-${String(number).padStart(12, "0")}`;
}

function datasetItem(number: number, split: string | null = null) {
  return {
    id: itemId(number),
    number,
    input: { question: `q${number}` } as Record<string, unknown>,
    expectedOutput: `a${number}`,
    metadata: { source: "test" },
    tags: ["golden"],
    split,
  };
}

function result(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: EXPERIMENT_ID,
    name: "support-golden-v3",
    status: "completed",
    checkStatus: null,
    itemsTotal: 2,
    itemsDone: 2,
    itemsFailed: 0,
    error: null,
    scores: [],
    url: URL_OF_EXPERIMENT,
    ...overrides,
  };
}

/**
 * The Evals routes of telemetry.dev, in memory: pages of a dataset version with a cursor, an
 * experiment that takes item posts of at most 100 and skips repeats, and its result.
 */
function fakeApi(
  options: {
    items?: ReturnType<typeof datasetItem>[];
    pageSize?: number;
    results?: Record<string, unknown>[];
    override?: (request: Request) => Response | Promise<Response> | undefined;
  } = {},
) {
  const items = options.items ?? [datasetItem(1), datasetItem(2)];
  const pageSize = options.pageSize ?? 100;
  const results = options.results ?? [result()];
  const requests: Request[] = [];
  const stored = new Set<string>();

  const fetch = vi.fn(async (input: string | URL, init?: RequestInit) => {
    const url = new URL(input);
    const request: Request = {
      method: init?.method ?? "GET",
      url,
      body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
      signal: init?.signal ?? undefined,
    };
    requests.push(request);

    const overridden = options.override?.(request);

    if (overridden) return overridden;

    const route = `${request.method} ${url.pathname}`;

    if (route === "GET /v1/datasets/support-golden") {
      const split = url.searchParams.get("split");
      const start = Number(url.searchParams.get("cursor") ?? 0);
      const matching = items.filter((item) => split === null || item.split === split);

      return Response.json({
        dataset: {
          id: "dataset-1",
          name: "support-golden",
          version: Number(url.searchParams.get("version") ?? 3),
        },
        items: matching.slice(start, start + pageSize),
        nextCursor: start + pageSize < matching.length ? String(start + pageSize) : null,
      });
    }

    if (route === "POST /v1/experiments") {
      const count = items.filter(
        (item) => request.body.split === undefined || item.split === request.body.split,
      ).length;

      return Response.json(
        {
          id: EXPERIMENT_ID,
          name: request.body.name ?? "support-golden-v3",
          datasetVersion: request.body.datasetVersion,
          itemsTotal: count * (request.body.runsPerItem ?? 1),
          url: URL_OF_EXPERIMENT,
        },
        { status: 201 },
      );
    }

    if (route === `POST /v1/experiments/${EXPERIMENT_ID}/items`) {
      if (request.body.items.length > 100) {
        return Response.json(
          { error: { code: "invalid_request", message: "at most 100 items" } },
          { status: 400 },
        );
      }

      let accepted = 0;

      for (const run of request.body.items) {
        const key = `${run.itemId}:${run.runIndex}`;

        if (!stored.has(key)) accepted += 1;
        stored.add(key);
      }

      return Response.json({ accepted, duplicates: request.body.items.length - accepted });
    }

    if (route === `POST /v1/experiments/${EXPERIMENT_ID}/complete`) {
      return Response.json({ id: EXPERIMENT_ID, status: "running" });
    }

    if (route === `GET /v1/experiments/${EXPERIMENT_ID}`) {
      return Response.json(results.length > 1 ? results.shift() : results[0]);
    }

    return Response.json({ error: { code: "not_found", message: route } }, { status: 404 });
  });

  const posts = () =>
    requests
      .filter((request) => request.url.pathname.endsWith("/items"))
      .map((request) => request.body.items);

  return { fetch: fetch as unknown as typeof globalThis.fetch, requests, posts };
}

function routes(requests: Request[]): string[] {
  return requests.map((request) => `${request.method} ${request.url.pathname}`);
}

function experiment(
  api: ReturnType<typeof fakeApi>,
  options: Partial<RunExperimentOptions> = {},
): ReturnType<typeof runExperiment> {
  return runExperiment({
    apiKey: "td_live_test",
    baseUrl: "https://ingest.test",
    fetch: api.fetch,
    dataset: "support-golden",
    task: (item) => `answer to ${String(item.input.question)}`,
    ...options,
  });
}

beforeEach(() => {
  for (const name of [
    "TELEMETRY_DEV_API_KEY",
    "TELEMETRY_DEV_BASE_URL",
    "TELEMETRY_DEV_EXPERIMENT_DATASET",
    "TELEMETRY_DEV_EXPERIMENT_MIN_SCORES",
    "TELEMETRY_DEV_EXPERIMENT_RESULTS",
    "CI",
    "GITHUB_ACTIONS",
    "GITHUB_REPOSITORY",
    "GITHUB_SHA",
    "GITHUB_EVENT_PATH",
    "GITHUB_RUN_ID",
    "GITHUB_SERVER_URL",
  ]) {
    vi.stubEnv(name, undefined);
  }

  vi.spyOn(console, "log").mockImplementation(() => {});
});

afterEach(async () => {
  await shutdown();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

test("runs each item of each page, posts batches of 100 on one version, then completes", async () => {
  const api = fakeApi({
    items: Array.from({ length: 75 }, (_, index) => datasetItem(index + 1)),
    pageSize: 50,
    results: [result({ itemsTotal: 150, itemsDone: 146, itemsFailed: 4 })],
  });

  const outcome = await experiment(api, { runsPerItem: 2, evaluators: ["Correctness"] });

  const sent = routes(api.requests);
  expect(sent.slice(0, 2)).toEqual(["GET /v1/datasets/support-golden", "POST /v1/experiments"]);
  // The second page and the first post of items run at the same time, in either order.
  expect(sent.slice(2, -2).sort()).toEqual([
    "GET /v1/datasets/support-golden",
    `POST /v1/experiments/${EXPERIMENT_ID}/items`,
    `POST /v1/experiments/${EXPERIMENT_ID}/items`,
  ]);
  expect(sent.slice(-2)).toEqual([
    `POST /v1/experiments/${EXPERIMENT_ID}/complete`,
    `GET /v1/experiments/${EXPERIMENT_ID}`,
  ]);

  // The first page names no version; the next one asks for the version of the first.
  const pages = api.requests.filter((request) => request.url.pathname.startsWith("/v1/datasets/"));
  expect(Object.fromEntries(pages[0]!.url.searchParams)).toEqual({});
  expect(Object.fromEntries(pages[1]!.url.searchParams)).toEqual({ version: "3", cursor: "50" });

  expect(api.requests[1]!.body).toEqual({
    dataset: "support-golden",
    datasetVersion: 3,
    evaluators: ["Correctness"],
    runsPerItem: 2,
  });

  const posts = api.posts();
  expect(posts.map((items) => items.length)).toEqual([100, 50]);

  const runs = posts
    .flat()
    .map((run: { itemId: string; runIndex: number }) => `${run.itemId}:${run.runIndex}`);
  expect(new Set(runs).size).toBe(150);
  expect(runs).toContain(`${itemId(75)}:1`);

  expect(posts[0]![0]).toEqual({
    itemId: itemId(1),
    runIndex: 0,
    output: "answer to q1",
    latencyMs: expect.any(Number),
  });

  expect(outcome).toEqual({
    id: EXPERIMENT_ID,
    name: "support-golden-v3",
    status: "completed",
    checkStatus: null,
    itemsTotal: 150,
    itemsDone: 146,
    itemsFailed: 4,
    error: null,
    scores: [],
    url: URL_OF_EXPERIMENT,
  });
});

test("each item run is its own trace, and each span of the task carries the experiment and the item", async () => {
  const { spans, logs } = setup({ environment: "experiment", sessionMode: "process" });
  const api = fakeApi();

  // An active span, an ambient session, and the process session would each put the runs in one
  // trace.
  await propagateAttributes({ sessionId: "ambient-session", userId: "u-1" }, () =>
    startActiveSpan("eval script", () =>
      experiment(api, {
        runsPerItem: 2,
        task: (item) => {
          startSpan("chat gpt-4o", { type: "generation" }).end();
          log("replied", { attributes: { number: item.number } });

          return { reply: `r${item.number}` };
        },
      }),
    ),
  );

  const posted = api.posts().flat();
  const finished = spans.getFinishedSpans();
  const chats = finished.filter((span) => span.name === "chat gpt-4o");
  const roots = finished.filter((span) => span.name === "experiment support-golden-v3");

  expect(chats).toHaveLength(4);
  expect(roots).toHaveLength(4);
  expect(new Set(roots.map((span) => span.spanContext().traceId)).size).toBe(4);
  expect(roots.map((span) => span.parentSpanContext?.isRemote)).toEqual([true, true, true, true]);

  for (const run of posted) {
    expect(run.traceId).toMatch(/^[0-9a-f]{32}$/);
    const root = roots.find((span) => span.spanContext().traceId === run.traceId)!;
    const chat = chats.find((span) => span.spanContext().traceId === run.traceId)!;

    expect(chat.parentSpanContext?.spanId).toBe(root.spanContext().spanId);

    for (const span of [root, chat]) {
      expect(span.attributes).toMatchObject({
        "td.experiment.id": EXPERIMENT_ID,
        "td.dataset.item_id": run.itemId,
        "gen_ai.conversation.id": `${EXPERIMENT_ID}:${run.itemId}:${run.runIndex}`,
        "user.id": "u-1",
      });
    }
  }

  const records = logs.getFinishedLogRecords();
  expect(records).toHaveLength(4);

  for (const record of records) {
    const run = posted.find(
      (candidate: { traceId: string }) => candidate.traceId === record.spanContext?.traceId,
    );

    expect(record.attributes).toMatchObject({
      "td.experiment.id": EXPERIMENT_ID,
      "td.dataset.item_id": run.itemId,
    });
  }

  expect(posted.map((run: { output: string }) => run.output).sort()).toEqual([
    '{"reply":"r1"}',
    '{"reply":"r1"}',
    '{"reply":"r2"}',
    '{"reply":"r2"}',
  ]);
});

test("runs at most `concurrency` tasks at a time", async () => {
  const api = fakeApi({ items: Array.from({ length: 12 }, (_, index) => datasetItem(index + 1)) });
  let running = 0;
  let most = 0;

  await experiment(api, {
    concurrency: 3,
    task: async () => {
      running += 1;
      most = Math.max(most, running);
      await new Promise((resolve) => setTimeout(resolve, 5));
      running -= 1;

      return "ok";
    },
  });

  expect(most).toBe(3);
  expect(api.posts().flat()).toHaveLength(12);
});

test("a task that throws fails only its item", async () => {
  const api = fakeApi();

  await experiment(api, {
    task: (item) => {
      if (item.number === 1) throw new RangeError("no reply");

      return "fine";
    },
  });

  const [failed, passed] = api.posts().flat();

  expect(failed).toMatchObject({ itemId: itemId(1), output: null, error: "RangeError: no reply" });
  expect(failed).not.toHaveProperty("scores");
  expect(passed).toMatchObject({ itemId: itemId(2), output: "fine" });
  expect(passed).not.toHaveProperty("error");
});

test("an output or an error that cannot be encoded does not stop the run", async () => {
  const api = fakeApi({ items: [datasetItem(1), datasetItem(2), datasetItem(3)] });
  const cycle: Record<string, unknown> = {};
  cycle.self = cycle;
  const unreadable = Object.defineProperty(new Error(), "message", {
    get() {
      throw new Error("no message");
    },
  });

  const outcome = await experiment(api, {
    task: (item) => {
      if (item.number === 3) throw unreadable;

      return item.number === 1 ? cycle : 10n;
    },
    // A lone surrogate, which UTF-8 cannot hold.
    scorers: { tone: () => "\ud83d" },
  });

  expect(outcome.status).toBe("completed");
  expect(
    api
      .posts()
      .flat()
      .sort((a: { itemId: string }, b: { itemId: string }) => a.itemId.localeCompare(b.itemId))
      .map((run: { output: unknown; error?: unknown; scores?: unknown }) => [
        run.output,
        run.error,
        run.scores,
      ]),
  ).toEqual([
    [null, undefined, [{ name: "tone", label: "\ud83d" }]],
    [null, undefined, [{ name: "tone", label: "\ud83d" }]],
    [null, "A value that cannot be converted to a string was thrown", undefined],
  ]);
  expect(routes(api.requests)).toContain(`POST /v1/experiments/${EXPERIMENT_ID}/complete`);
});

test("each run of an item starts from the dataset input, also when a task changes it", async () => {
  const api = fakeApi({
    items: [{ ...datasetItem(1), input: { messages: [{ role: "user", content: "Hi" }] } }],
  });
  const seen: number[] = [];

  await experiment(api, {
    runsPerItem: 3,
    concurrency: 1,
    task: (item) => {
      const messages = item.input.messages as unknown[];
      seen.push(messages.length);
      messages.push({ role: "assistant", content: "Hello" });

      return "ok";
    },
  });

  expect(seen).toEqual([1, 1, 1]);
});

test("scorers give pass/fail, numbers, labels, and reasons, and a bad result drops only its score", async () => {
  const errors: Error[] = [];
  setup({ environment: "experiment", onError: (error) => errors.push(error) });
  const api = fakeApi({ items: [datasetItem(1)] });

  await experiment(api, {
    scorers: {
      exact: (output, item) => output === `answer to ${String(item.input.question)}`,
      polite: () => false,
      length: (output) => String(output).length,
      tone: () => " friendly ",
      judged: () => ({ value: 0.25, reason: "too short" }),
      none: () => null,
      thrown: () => {
        throw new Error("judge down");
      },
      mismatched: () => ({ value: 0.5, label: "pass" }),
    },
  });

  expect(api.posts()[0]![0].scores).toEqual([
    { name: "exact", value: 1, label: "pass" },
    { name: "polite", value: 0, label: "fail" },
    { name: "length", value: 12 },
    { name: "tone", label: "friendly" },
    { name: "judged", value: 0.25, reason: "too short" },
  ]);

  expect(errors.map((error) => [(error as ExperimentError).code, error.message])).toEqual([
    ["scorer_failed", "The scorer thrown failed on item 1: Error: judge down"],
    [
      "scorer_failed",
      "The scorer mismatched failed on item 1: TypeError: returned a score that is not valid: The label pass goes with the value 1",
    ],
  ]);
});

test("a failed post of items stops new items, completes the experiment, and rejects with the API error", async () => {
  const api = fakeApi({
    items: Array.from({ length: 300 }, (_, index) => datasetItem(index + 1)),
    override: (request) =>
      request.url.pathname.endsWith("/items")
        ? Response.json(
            { error: { code: "item_not_found", message: "support-golden v3 has no item" } },
            { status: 422 },
          )
        : undefined,
  });
  let started = 0;

  const error = await experiment(api, {
    concurrency: 1,
    task: () => {
      started += 1;

      return "ok";
    },
  }).catch((caught: unknown) => caught);

  expect(error).toBeInstanceOf(ExperimentError);
  expect(error).toMatchObject({ code: "item_not_found", status: 422 });
  // The one worker waits for the post of the first 100, which fails, so no item starts after it.
  expect(started).toBe(100);
  // The complete ends the experiment, which would else stay running.
  expect(routes(api.requests).slice(-2)).toEqual([
    `POST /v1/experiments/${EXPERIMENT_ID}/items`,
    `POST /v1/experiments/${EXPERIMENT_ID}/complete`,
  ]);
});

test("a complete that fails after a failed run goes to onError, and the run rejects with the first error", async () => {
  const errors: Error[] = [];
  setup({ environment: "experiment", onError: (error) => errors.push(error) });
  const api = fakeApi({
    override: (request) =>
      request.url.pathname.endsWith("/items") || request.url.pathname.endsWith("/complete")
        ? Response.json(
            { error: { code: "invalid_api_key", message: "The API key was revoked" } },
            { status: 401 },
          )
        : undefined,
  });

  await expect(experiment(api)).rejects.toMatchObject({
    code: "invalid_api_key",
    message: "The API key was revoked",
  });
  expect(routes(api.requests).filter((route) => route.endsWith("/complete"))).toHaveLength(1);
  expect(errors.map((error) => (error as ExperimentError).code)).toEqual(["invalid_api_key"]);
});

test("an item post that fails with 500 goes again, and the server skips the runs it has", async () => {
  let failures = 1;
  const api = fakeApi({
    override: (request) =>
      request.url.pathname.endsWith("/items") && failures-- > 0
        ? new Response("upstream", { status: 500 })
        : undefined,
  });

  await experiment(api);

  expect(api.posts()).toHaveLength(2);
  expect(api.posts()[1]).toEqual(api.posts()[0]);
});

test("a post of items holds at most 4 MiB of JSON", async () => {
  const api = fakeApi({ items: Array.from({ length: 20 }, (_, index) => datasetItem(index + 1)) });

  // 100,000 characters of 3 UTF-8 bytes: about 300 KB of JSON for each item run.
  await experiment(api, { task: () => "中".repeat(100_000) });

  expect(api.posts().map((items) => items.length)).toEqual([13, 7]);
});

test.each([
  ["10,000 items in pages run", 10_000, undefined],
  ["10,001 items in pages stop the run", 10_001, "invalid_response"],
])("%s", async (_, count, code) => {
  const api = fakeApi({
    items: Array.from({ length: count }, (_, index) => datasetItem(index + 1)),
    pageSize: 5_000,
  });

  const outcome = await experiment(api, { concurrency: 20, task: () => "ok" }).catch(
    (caught: unknown) => caught,
  );

  expect((outcome as ExperimentError).code).toBe(code);
  expect(api.posts().flat()).toHaveLength(10_000);
  expect(routes(api.requests).filter((route) => route.endsWith("/complete"))).toHaveLength(1);
});

test.each([
  ["a dataset page", "GET /v1/datasets/support-golden", "timeout", 2, undefined],
  ["an item post", `POST /v1/experiments/${EXPERIMENT_ID}/items`, "network", 2, undefined],
  ["the create", "POST /v1/experiments", "timeout", 1, "timeout"],
  ["the create", "POST /v1/experiments", "network", 1, "network_error"],
])("a %s %s, after a %s, sends %i times in all", async (_, route, failure, sends, code) => {
  vi.useFakeTimers();

  try {
    let failures = 1;
    const api = fakeApi({
      override: (request) => {
        if (`${request.method} ${request.url.pathname}` !== route || failures-- <= 0) {
          return undefined;
        }

        if (failure === "network") return Promise.reject(new TypeError("fetch failed"));

        // A server that never answers: the request ends when its timeout aborts it.
        return new Promise<Response>((_, reject) =>
          request.signal!.addEventListener("abort", () => reject(request.signal!.reason)),
        );
      },
    });

    const outcome = experiment(api).catch((caught: unknown) => caught);
    await vi.advanceTimersByTimeAsync(60_000);

    expect(((await outcome) as ExperimentError).code).toBe(code);
    expect(routes(api.requests).filter((sent) => sent === route)).toHaveLength(sends);
  } finally {
    vi.useRealTimers();
  }
});

test.each([
  ["500", () => new Response("upstream", { status: 500 }), "http_500"],
  // The server answers 503 when its database work times out, which can be after the commit.
  [
    "503",
    () =>
      Response.json(
        { error: { code: "evals_unavailable", message: "Evals are unavailable right now" } },
        { status: 503 },
      ),
    "evals_unavailable",
  ],
])(
  "a create that fails with %s does not go again, so it cannot start two experiments",
  async (status, reply, code) => {
    const api = fakeApi({
      override: (request) =>
        request.method === "POST" && request.url.pathname === "/v1/experiments"
          ? reply()
          : undefined,
    });

    await expect(experiment(api)).rejects.toMatchObject({ code, status: Number(status) });
    expect(routes(api.requests)).toEqual([
      "GET /v1/datasets/support-golden",
      "POST /v1/experiments",
    ]);
  },
);

test("a create that gets a 429 goes again after its Retry-After", async () => {
  let limited = 1;
  const api = fakeApi({
    override: (request) =>
      request.method === "POST" && request.url.pathname === "/v1/experiments" && limited-- > 0
        ? Response.json(
            { error: { code: "rate_limited", message: "Too many requests for this project" } },
            { status: 429, headers: { "retry-after": "2" } },
          )
        : undefined,
  });
  const started = Date.now();

  await experiment(api);

  expect(routes(api.requests).filter((route) => route === "POST /v1/experiments")).toHaveLength(2);
  // Without the header, the first retry waits 1 second.
  expect(Date.now() - started).toBeGreaterThanOrEqual(2_000);
});

test("waits until the server evaluators end the experiment", async () => {
  const api = fakeApi({ results: [result({ status: "running", itemsDone: 1 }), result()] });

  const outcome = await experiment(api);

  expect(outcome.status).toBe("completed");
  expect(
    routes(api.requests).filter((route) => route.startsWith("GET /v1/experiments/")),
  ).toHaveLength(2);
});

test("gives up when the experiment does not end in 30 minutes, with the result", async () => {
  vi.useFakeTimers();

  try {
    const polls: number[] = [];
    const api = fakeApi({
      override: (request) => {
        if (request.method === "GET" && request.url.pathname.startsWith("/v1/experiments/")) {
          polls.push(Date.now());

          return Response.json(result({ status: "running", itemsDone: 1 }));
        }

        return undefined;
      },
    });

    const outcome = experiment(api).catch((caught: unknown) => caught);
    await vi.advanceTimersByTimeAsync(31 * 60_000);
    const error = (await outcome) as ExperimentError;

    expect(error.code).toBe("timeout");
    expect(error.result?.status).toBe("running");
    // The last poll is the one whose next wait of 5 seconds would end past the 30 minutes.
    const waited = polls.at(-1)! - polls[0]!;
    expect(waited).toBeGreaterThan(30 * 60_000 - 5_000);
    expect(waited).toBeLessThanOrEqual(30 * 60_000);
  } finally {
    vi.useRealTimers();
  }
});

test.each([
  [
    "the API error",
    { error: { code: "dataset_not_found", message: "No dataset named support-golden" } },
    404,
    "dataset_not_found",
  ],
  [
    "a page that cannot be read",
    { dataset: { version: 1 }, items: [{ id: 1 }] },
    200,
    "invalid_response",
  ],
  [
    "an empty dataset",
    { dataset: { id: "d", name: "support-golden", version: 0 }, items: [], nextCursor: null },
    200,
    "dataset_empty",
  ],
])("rejects with %s before it creates an experiment", async (_, body, status, code) => {
  const api = fakeApi({
    override: (request) =>
      request.url.pathname.startsWith("/v1/datasets/")
        ? Response.json(body, { status })
        : undefined,
  });

  await expect(experiment(api)).rejects.toMatchObject({ code });
  expect(routes(api.requests)).toEqual(["GET /v1/datasets/support-golden"]);
});

test("a cursor that repeats is an invalid response, not an endless fetch", async () => {
  const api = fakeApi({
    override: (request) =>
      request.url.pathname.startsWith("/v1/datasets/")
        ? Response.json({
            dataset: { id: "d", name: "support-golden", version: 3 },
            items: [datasetItem(1)],
            nextCursor: "same",
          })
        : undefined,
  });

  await expect(experiment(api)).rejects.toMatchObject({ code: "invalid_response" });
  expect(
    routes(api.requests).filter((route) => route.startsWith("GET /v1/datasets/")),
  ).toHaveLength(2);
});

const failedCheck = result({
  checkStatus: "failed",
  scores: [
    {
      name: "correctness",
      kind: "pass_fail",
      value: 0.85,
      passCount: 17,
      failCount: 3,
      baselineValue: 0.9,
      delta: -5,
      minScore: 0.9,
      passed: false,
    },
    {
      name: "length",
      kind: null,
      value: null,
      passCount: 0,
      failCount: 0,
      baselineValue: null,
      delta: null,
      minScore: null,
      passed: null,
    },
  ],
});

test("in CI mode a failed check rejects with the result", async () => {
  vi.stubEnv("CI", "true");
  const api = fakeApi({ results: [failedCheck] });

  const error = (await experiment(api, { minScores: { correctness: 0.9 } }).catch(
    (caught: unknown) => caught,
  )) as ExperimentError;

  expect(error).toBeInstanceOf(ExperimentError);
  expect(error.code).toBe("check_failed");
  expect(error.message).toBe(
    `The check of support-golden-v3 failed: correctness is 85%, under the pass mark 90%. ${URL_OF_EXPERIMENT}`,
  );
  expect(error.result?.scores[1]?.kind).toBeNull();
});

test("in CI mode a check that failed items failed names their count, and a mean with one decimal, in the log too", async () => {
  vi.stubEnv("CI", "true");

  const api = fakeApi({
    results: [
      result({
        checkStatus: "failed",
        itemsTotal: 6,
        itemsDone: 6,
        itemsFailed: 5,
        scores: [
          {
            name: "correctness",
            kind: "pass_fail",
            value: 1,
            passCount: 1,
            failCount: 0,
            baselineValue: null,
            delta: null,
            minScore: 0.9,
            passed: true,
            itemsFailed: 5,
          },
          {
            // toFixed(1) of the exact value, as in the comment of the action: the float 4.35 is just under 4.35.
            name: "helpfulness",
            kind: "number",
            value: 4.35,
            passCount: 0,
            failCount: 0,
            baselineValue: null,
            delta: null,
            minScore: 4.5,
            passed: false,
            itemsFailed: 5,
          },
          {
            // An exact tie rounds away from zero.
            name: "tone",
            kind: "number",
            value: 4.25,
            passCount: 0,
            failCount: 0,
            baselineValue: null,
            delta: null,
            minScore: null,
            passed: null,
            itemsFailed: 5,
          },
          {
            name: "length",
            kind: "number",
            value: 12,
            passCount: 0,
            failCount: 0,
            baselineValue: null,
            delta: null,
            minScore: 10,
            passed: true,
            itemsFailed: 5,
          },
        ],
      }),
    ],
  });

  const error = (await experiment(api, { minScores: { correctness: 0.9 } }).catch(
    (caught: unknown) => caught,
  )) as ExperimentError;

  expect(error.code).toBe("check_failed");
  expect(error.message).toBe(
    `The check of support-golden-v3 failed: 5 of 6 items failed; helpfulness is 4.3, under the pass mark 4.5. ${URL_OF_EXPERIMENT}`,
  );
  expect(error.result?.scores[0]?.itemsFailed).toBe(5);
  expect(vi.mocked(console.log).mock.calls.at(-1)?.[0]).toBe(
    [
      `Experiment support-golden-v3 completed. The check failed. ${URL_OF_EXPERIMENT}`,
      "  5 of 6 items failed.",
      "  correctness: 100% (pass mark 90%)",
      "  helpfulness: 4.3 (pass mark 4.5)",
      "  tone: 4.3",
      "  length: 12.0 (pass mark 10.0)",
    ].join("\n"),
  );

  const twice = (await experiment(
    fakeApi({
      results: [result({ checkStatus: "failed", itemsTotal: 4, itemsDone: 4, itemsFailed: 3 })],
    }),
    { runsPerItem: 2 },
  ).catch((caught: unknown) => caught)) as ExperimentError;

  expect(twice.message).toBe(
    `The check of support-golden-v3 failed: 3 of 4 runs failed. ${URL_OF_EXPERIMENT}`,
  );
  expect(vi.mocked(console.log).mock.calls.at(-1)?.[0]).toContain("\n  3 of 4 runs failed.");
});

test.each([
  ["CI is off", {}, undefined],
  ["ci: false wins over CI", { ci: false }, "true"],
])("a failed check resolves when %s", async (_, options, ci) => {
  if (ci) vi.stubEnv("CI", ci);
  const api = fakeApi({ results: [failedCheck] });

  await expect(experiment(api, options)).resolves.toMatchObject({ checkStatus: "failed" });
});

test("a result score that leaves out its null fields reads them as null", async () => {
  const api = fakeApi({
    results: [result({ scores: [{ name: "length", passCount: 0, failCount: 0 }] })],
  });

  const outcome = await experiment(api);

  expect(outcome.scores).toEqual([
    {
      name: "length",
      kind: null,
      value: null,
      passCount: 0,
      failCount: 0,
      baselineValue: null,
      delta: null,
      minScore: null,
      passed: null,
      itemsFailed: null,
    },
  ]);
});

test("a result without checkStatus is an invalid response, not a run with no check", async () => {
  const { checkStatus: _, ...missing } = failedCheck;
  const api = fakeApi({ results: [missing] });

  await expect(experiment(api)).rejects.toMatchObject({ code: "invalid_response" });
});

test("in CI mode an experiment that fails rejects with its error", async () => {
  vi.stubEnv("CI", "1");
  const api = fakeApi({
    results: [result({ status: "failed", error: "0 of 2 items ran. Each one failed." })],
  });

  await expect(experiment(api)).rejects.toMatchObject({
    code: "experiment_failed",
    message: `The experiment support-golden-v3 failed: 0 of 2 items ran. Each one failed.. ${URL_OF_EXPERIMENT}`,
  });
});

test("in GitHub Actions the experiment records the pull request, its head commit, and the run", async () => {
  const dir = mkdtempSync(join(tmpdir(), "td-experiment-"));
  const event = join(dir, "event.json");
  const results = join(dir, "results.jsonl");
  writeFileSync(
    event,
    JSON.stringify({
      pull_request: {
        number: 42,
        title: "Shorter replies",
        html_url: "https://github.com/acme/support/pull/42",
        head: { sha: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2" },
      },
    }),
  );
  writeFileSync(results, "");

  vi.stubEnv("CI", "true");
  vi.stubEnv("GITHUB_ACTIONS", "true");
  vi.stubEnv("GITHUB_REPOSITORY", "acme/support");
  vi.stubEnv("GITHUB_SHA", "ffffffffffffffffffffffffffffffffffffffff");
  vi.stubEnv("GITHUB_EVENT_PATH", event);
  vi.stubEnv("GITHUB_RUN_ID", "991");
  vi.stubEnv("TELEMETRY_DEV_EXPERIMENT_DATASET", "support-golden");
  vi.stubEnv("TELEMETRY_DEV_EXPERIMENT_MIN_SCORES", "correctness=0.9\nlength = 12.5,__proto__=1");
  vi.stubEnv("TELEMETRY_DEV_EXPERIMENT_RESULTS", results);
  const api = fakeApi();

  await experiment(api, { dataset: undefined, minScores: { correctness: 0.5, tone: 2 } });

  expect(api.requests[1]!.body).toMatchObject({
    dataset: "support-golden",
    minScores: JSON.parse('{"correctness":0.9,"tone":2,"length":12.5,"__proto__":1}'),
    ci: {
      provider: "github_actions",
      repository: "acme/support",
      commit: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
      prNumber: 42,
      prTitle: "Shorter replies",
      prUrl: "https://github.com/acme/support/pull/42",
      runUrl: "https://github.com/acme/support/actions/runs/991",
    },
  });

  // A second experiment of the script, with the CI mode off: the action does not fail the job for
  // it, and counts its items from runsPerItem.
  await experiment(fakeApi(), { ci: false, runsPerItem: 2 });

  expect(readFileSync(results, "utf8")).toBe(
    `${JSON.stringify({ ...result(), ci: true, runsPerItem: 1 })}\n` +
      `${JSON.stringify({ ...result(), ci: false, runsPerItem: 2 })}\n`,
  );
});

test("a pull request title is cut to 256 UTF-16 units without half of an emoji", async () => {
  const dir = mkdtempSync(join(tmpdir(), "td-experiment-"));
  const event = join(dir, "event.json");
  // GitHub counts the emoji as one character, thus the title fits its limit of 256.
  writeFileSync(event, JSON.stringify({ pull_request: { title: `${"a".repeat(255)}😀` } }));
  vi.stubEnv("CI", "true");
  vi.stubEnv("GITHUB_ACTIONS", "true");
  vi.stubEnv("GITHUB_REPOSITORY", "acme/support");
  vi.stubEnv("GITHUB_SHA", "a1b2c3d");
  vi.stubEnv("GITHUB_EVENT_PATH", event);
  const api = fakeApi();

  await experiment(api);

  expect(api.requests[1]!.body.ci.prTitle).toBe("a".repeat(255));
});

test.each([
  [
    "no dataset",
    {},
    { dataset: undefined },
    "Pass dataset, or set the dataset input of the experiment action",
  ],
  [
    "a dataset that the action does not name",
    { TELEMETRY_DEV_EXPERIMENT_DATASET: "support-v2" },
    {},
    "dataset is support-golden, but the experiment action names support-v2. Name one dataset.",
  ],
  [
    "min-scores that do not parse",
    { TELEMETRY_DEV_EXPERIMENT_MIN_SCORES: "correctness=0x10" },
    {},
    'TELEMETRY_DEV_EXPERIMENT_MIN_SCORES: "correctness=0x10" is not a name=number pair',
  ],
  ["runsPerItem 6", {}, { runsPerItem: 6 }, "runsPerItem must be an integer from 1 to 5"],
  ["concurrency 0", {}, { concurrency: 0 }, "concurrency must be an integer from 1 to 100"],
  [
    "a scorer named user_feedback",
    {},
    { scorers: { user_feedback: () => 1 } },
    "user_feedback is the score of user feedback. Name the scorer another way.",
  ],
  [
    "a scorer with the name of an evaluator",
    {},
    { evaluators: ["correctness"], scorers: { Correctness: () => true } },
    "The scorer Correctness has the name of an evaluator. Rename one of them.",
  ],
])("rejects %s before any request", async (_, env, options, message) => {
  for (const [name, value] of Object.entries(env)) vi.stubEnv(name, value);
  const api = fakeApi();

  await expect(experiment(api, options as Partial<RunExperimentOptions>)).rejects.toThrow(message);
  expect(api.requests).toEqual([]);
});

test("without an API key it rejects with missing_api_key and sends nothing", async () => {
  const api = fakeApi();

  await expect(experiment(api, { apiKey: undefined })).rejects.toMatchObject({
    code: "missing_api_key",
  });
  expect(api.requests).toEqual([]);
});

test("the task gets each run of an item of the split, with its run index", async () => {
  const seen: ExperimentItem[] = [];
  const api = fakeApi({ items: [datasetItem(7, "test")] });

  await experiment(api, {
    split: "test",
    runsPerItem: 2,
    concurrency: 1,
    task: (item) => {
      seen.push(item);

      return "ok";
    },
  });

  expect(seen).toEqual([
    { ...datasetItem(7, "test"), runIndex: 0 },
    { ...datasetItem(7, "test"), runIndex: 1 },
  ]);
  expect(api.requests[0]!.url.searchParams.get("split")).toBe("test");
  expect(api.requests[1]!.body.split).toBe("test");
});
