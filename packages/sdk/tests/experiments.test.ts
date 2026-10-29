import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, expect, test, vi } from "vitest";

import {
  type ExperimentItem,
  ExperimentError,
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
    input: { question: `q${number}` },
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
    override?: (request: Request) => Response | undefined;
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

  expect(routes(api.requests)).toEqual([
    "GET /v1/datasets/support-golden",
    "POST /v1/experiments",
    "GET /v1/datasets/support-golden",
    `POST /v1/experiments/${EXPERIMENT_ID}/items`,
    `POST /v1/experiments/${EXPERIMENT_ID}/items`,
    `POST /v1/experiments/${EXPERIMENT_ID}/complete`,
    `GET /v1/experiments/${EXPERIMENT_ID}`,
  ]);

  // The first page names no version; the next one asks for the version of the first.
  expect(Object.fromEntries(api.requests[0]!.url.searchParams)).toEqual({});
  expect(Object.fromEntries(api.requests[2]!.url.searchParams)).toEqual({
    version: "3",
    cursor: "50",
  });

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
  const { spans } = setup({ environment: "experiment", sessionMode: "process" });
  const api = fakeApi();

  // An active span, an ambient session, and the process session would each put the runs in one
  // trace.
  await propagateAttributes({ sessionId: "ambient-session", userId: "u-1" }, () =>
    startActiveSpan("eval script", () =>
      experiment(api, {
        runsPerItem: 2,
        task: (item) => {
          startSpan("chat gpt-4o", { type: "generation" }).end();

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

test("a failed post of items stops new items, never completes, and rejects with the API error", async () => {
  const api = fakeApi({
    items: Array.from({ length: 300 }, (_, index) => datasetItem(index + 1)),
    override: (request) =>
      request.url.pathname.endsWith("/items")
        ? Response.json(
            { error: { code: "experiment_ended", message: "support-golden-v3 has ended." } },
            { status: 409 },
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
  expect(error).toMatchObject({ code: "experiment_ended", status: 409 });
  // The one worker waits for the post of the first 100, which fails, so no item starts after it.
  expect(started).toBe(100);
  expect(routes(api.requests)).not.toContain(`POST /v1/experiments/${EXPERIMENT_ID}/complete`);
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

test("a create that fails with 500 does not go again, so it cannot start two experiments", async () => {
  const api = fakeApi({
    override: (request) =>
      request.method === "POST" && request.url.pathname === "/v1/experiments"
        ? new Response("upstream", { status: 500 })
        : undefined,
  });

  await expect(experiment(api)).rejects.toMatchObject({ code: "http_500", status: 500 });
  expect(routes(api.requests).filter((route) => route === "POST /v1/experiments")).toHaveLength(1);
});

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

test.each([
  ["CI is off", {}, undefined],
  ["ci: false wins over CI", { ci: false }, "true"],
])("a failed check resolves when %s", async (_, options, ci) => {
  if (ci) vi.stubEnv("CI", ci);
  const api = fakeApi({ results: [failedCheck] });

  await expect(experiment(api, options)).resolves.toMatchObject({ checkStatus: "failed" });
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
  vi.stubEnv("TELEMETRY_DEV_EXPERIMENT_MIN_SCORES", "correctness=0.9\nlength = 12.5");
  vi.stubEnv("TELEMETRY_DEV_EXPERIMENT_RESULTS", results);
  const api = fakeApi();

  await experiment(api, { dataset: undefined, minScores: { correctness: 0.5, tone: 2 } });

  expect(api.requests[1]!.body).toMatchObject({
    dataset: "support-golden",
    minScores: { correctness: 0.9, length: 12.5, tone: 2 },
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
  expect(readFileSync(results, "utf8")).toBe(`${JSON.stringify(result())}\n`);
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
