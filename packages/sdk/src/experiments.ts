import { type Context, trace } from "@opentelemetry/api";
import {
  activeContext,
  diag,
  jsonAttr,
  PROPAGATED_KEY,
  propagatedFromContext,
  reportError,
  resolveEnv,
} from "@telemetry-dev/otel";

import { type Api, type ApiOptions, isRecord, request, resolveApi, sleep } from "./api.ts";
import { truncate } from "./capture.ts";
import { type CiRun, githubRun, inCi, minScoresFromEnv, writeResult } from "./ci.ts";
import { currentClient, flush } from "./client.ts";
import { checkValue, MAX_REASON_LENGTH, MAX_SCORE_NAME_LENGTH, USER_FEEDBACK } from "./scores.ts";
import { type SpanHandle, startActiveSpan } from "./span.ts";

/** An item of a dataset version, as `GET /v1/datasets/{name}` returns it. */
export interface DatasetItem {
  /** The key of the item. It stays the same across dataset versions. */
  readonly id: string;
  readonly number: number;
  readonly input: Readonly<Record<string, unknown>>;
  readonly expectedOutput: string | null;
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly tags: readonly string[];
  readonly split: string | null;
}

/** One run of a dataset item. With `runsPerItem` above 1, an item runs more than one time. */
export interface ExperimentItem extends DatasetItem {
  /** 0 to `runsPerItem` - 1. */
  readonly runIndex: number;
}

/**
 * What a scorer returns. `true` and `false` are a pass and a fail, a number is the value, and a
 * string is the label. Null or undefined gives no score.
 */
export type ScorerResult =
  | boolean
  | number
  | string
  | { value?: number | boolean; label?: string; reason?: string }
  | null
  | undefined;

/** Scores the output of an item on your machine. The score has the name of the scorer. */
export type Scorer = (
  output: unknown,
  item: ExperimentItem,
) => ScorerResult | Promise<ScorerResult>;

export interface RunExperimentOptions extends ApiOptions {
  /** Runs the code under test on one item. Its return value is the output of the item. */
  task: (item: ExperimentItem) => unknown;
  /**
   * The name of the dataset. Default: the `dataset` input of the experiment action, which it
   * sets in `TELEMETRY_DEV_EXPERIMENT_DATASET`. When both are set, they must be the same.
   */
  dataset?: string;
  /** Unique in the project. telemetry.dev picks a name when it is not set. */
  name?: string;
  /** Default: the newest version. */
  datasetVersion?: number;
  /** Run only the items of this split. */
  split?: string;
  /** Evaluators of the project that score the outputs on the server, by name. */
  evaluators?: readonly string[];
  /** Scorers that run here, by score name. */
  scorers?: Readonly<Record<string, Scorer>>;
  /**
   * Pass marks by score name: the lowest pass rate (0 to 1) of a pass/fail score, or the lowest
   * mean of a number score. The `min-scores` input of the experiment action adds to them and wins
   * for the same name.
   */
  minScores?: Readonly<Record<string, number>>;
  /** 1 to 5. Default 1. */
  runsPerItem?: number;
  /** Item runs at the same time, 1 to 100. Default 4. */
  concurrency?: number;
  metadata?: Readonly<Record<string, unknown>>;
  /**
   * Records the GitHub Actions run, and rejects when the check fails or the experiment does not
   * complete. Default: on when the `CI` variable is `true` or `1`.
   */
  ci?: boolean;
}

export interface ExperimentScore {
  readonly name: string;
  /** `pass_fail`, `number`, `labels`, `a_b_tie`, or `up_down`. Null before the score has values. */
  readonly kind: string | null;
  /** The pass rate (0 to 1) of a pass/fail or up/down score, or the mean of a number score. */
  readonly value: number | null;
  readonly passCount: number;
  readonly failCount: number;
  /** The value in the experiment that this one compares with, by default the dataset baseline. */
  readonly baselineValue: number | null;
  /** Whole points of a pass rate, or the change of a mean to one decimal. */
  readonly delta: number | null;
  readonly minScore: number | null;
  /** Null when the score has no pass mark. */
  readonly passed: boolean | null;
  /**
   * The runs that failed. They have no score, so `value` leaves them out, and the check fails.
   * Null from a server that does not send it.
   */
  readonly itemsFailed: number | null;
}

export interface ExperimentResult {
  readonly id: string;
  readonly name: string;
  /** `completed`, `failed`, or `stopped`. */
  readonly status: string;
  /** Null when the experiment has no pass marks. */
  readonly checkStatus: "passed" | "failed" | null;
  readonly itemsTotal: number;
  readonly itemsDone: number;
  readonly itemsFailed: number;
  /** Why the experiment failed. */
  readonly error: string | null;
  readonly scores: readonly ExperimentScore[];
  readonly url: string;
}

/**
 * Why runExperiment failed. `code` is the code of the API error (`dataset_not_found`,
 * `insufficient_scope`, `invalid_api_key`, …), or `missing_api_key`, `timeout`, `network_error`,
 * `invalid_response`, or `dataset_empty`. In CI mode it is also `check_failed` or
 * `experiment_failed`, and `result` holds the result. A response with a different status and no
 * API error gives `http_<status>`. A scorer that throws or returns a value that is not a score
 * reports a `scorer_failed` error through `onError`.
 */
export class ExperimentError extends Error {
  override readonly name = "ExperimentError";
  readonly result?: ExperimentResult;

  constructor(
    readonly code: string,
    message: string,
    readonly status?: number,
    options?: { cause?: unknown; result?: ExperimentResult },
  ) {
    super(message, options);
    this.result = options?.result;
  }
}

interface Run {
  dataset: string;
  task: (item: ExperimentItem) => unknown;
  name?: string;
  datasetVersion?: number;
  split?: string;
  evaluators?: readonly string[];
  scorers: [string, Scorer][];
  minScores?: Record<string, number>;
  runsPerItem: number;
  concurrency: number;
  metadata?: Readonly<Record<string, unknown>>;
  ci: CiRun | undefined;
  inCi: boolean;
}

interface Created {
  id: string;
  name: string;
  itemsTotal: number;
  url: string;
}

interface DatasetPage {
  version: number;
  items: DatasetItem[];
  nextCursor: string | null;
}

interface ItemScore {
  name: string;
  value?: number;
  label?: string;
  reason?: string;
}

interface ItemResult {
  itemId: string;
  runIndex: number;
  output: string | null;
  traceId?: string;
  latencyMs: number;
  error?: string;
  scores?: ItemScore[];
}

type Authed = Api & { apiKey: string };

/** The `experiment` environment, which pages, metrics, and alerts leave out by default. */
const EXPERIMENT_ENVIRONMENT = "experiment";
const MAX_RUNS_PER_ITEM = 5;
const MAX_CONCURRENCY = 100;
/** Server evaluators of one experiment, and scorers: an item takes at most 20 scores. */
const MAX_EVALUATORS = 20;
const MAX_MIN_SCORES = 20;
const MAX_ERROR_LENGTH = 4_000;
const MAX_OUTPUT_LENGTH = 100_000;
const MAX_DATASET_ITEMS = 10_000;
/** Items in one `POST /v1/experiments/{id}/items`. */
const MAX_BATCH_ITEMS = 100;
/** Bytes of JSON in one post of items, well under the 16 MiB body limit of the API. */
const MAX_BATCH_BYTES = 4 * 1024 * 1024;
const REQUEST_TIMEOUT_MS = 30_000;
/** How long runExperiment waits for the server evaluators after the last item. */
const MAX_WAIT_MS = 30 * 60_000;
const MAX_POLL_INTERVAL_MS = 5_000;
const FINISHED = new Set(["completed", "failed", "stopped"]);
const RATE_KINDS = new Set(["pass_fail", "up_down"]);

/**
 * Runs `task` on each item of a dataset and records the result as an experiment in telemetry.dev.
 * Each item run is its own trace, with `td.experiment.id` and `td.dataset.item_id` on each span.
 * The outputs go to telemetry.dev in batches of at most 100, with the scores of `scorers`. Then
 * runExperiment waits for the `evaluators` of the server, prints the result, and returns it.
 *
 * The run cannot start without its dataset, so runExperiment rejects with an `ExperimentError`
 * when a request fails. When the run stops after it made the experiment, it first sends the
 * complete request, so that telemetry.dev ends the experiment with the item runs that it has. A
 * task that throws fails only its item. In CI mode, runExperiment rejects when the check fails, so
 * the process exits with an error.
 */
export async function runExperiment(options: RunExperimentOptions): Promise<ExperimentResult> {
  const env = resolveEnv();
  const run = checkOptions(options, env);
  const resolved = resolveApi(options);

  if (!resolved.apiKey) {
    throw new ExperimentError(
      "missing_api_key",
      "No API key: pass apiKey or set TELEMETRY_DEV_API_KEY",
    );
  }

  const api = { ...resolved, apiKey: resolved.apiKey };
  warnAboutClient();

  const first = await fetchPage(api, run, run.datasetVersion, undefined);

  if (first.items.length === 0) {
    throw new ExperimentError(
      "dataset_empty",
      `The dataset ${run.dataset} has no items${run.split === undefined ? "" : ` in the split ${run.split}`}`,
    );
  }

  const created = parseCreated(
    await request(api, ExperimentError, "POST", "/v1/experiments", {
      body: {
        name: run.name,
        dataset: run.dataset,
        datasetVersion: first.version,
        split: run.split,
        evaluators: run.evaluators,
        minScores: run.minScores,
        runsPerItem: run.runsPerItem,
        ci: run.ci,
        metadata: run.metadata,
      },
      timeoutMs: REQUEST_TIMEOUT_MS,
      // A repeat after a lost response would start a second experiment.
      idempotent: false,
    }),
  );

  try {
    console.log(
      `Experiment ${created.name}: ${created.itemsTotal / run.runsPerItem} items` +
        (run.runsPerItem > 1 ? `, ${run.runsPerItem} runs each` : "") +
        `. ${created.url}`,
    );
    await runItems(api, run, created, itemRuns(api, run, first));
  } catch (error) {
    // Without the complete, the experiment would stay running: the server ends it with the item
    // runs that it has, or fails it when it has none.
    await complete(api, created.id).catch((cause: unknown) => reportError(api.onError, cause));

    throw error;
  }

  // The spans of the items reach ingest before the server reads their cost.
  await flush();
  await complete(api, created.id);

  const { result, body } = await waitForResult(api, created.id);
  writeResult(env, body, api.onError);
  console.log(describe(result));

  if (run.inCi) checkResult(result);

  return result;
}

function checkOptions(options: RunExperimentOptions, env: Record<string, string | undefined>): Run {
  if (typeof options.task !== "function") throw new TypeError("task must be a function");

  for (const [key, value] of [
    ["dataset", options.dataset],
    ["name", options.name],
    ["split", options.split],
  ] as const) {
    if (value !== undefined && (typeof value !== "string" || value.length === 0)) {
      throw new TypeError(`${key} must be a non-empty string`);
    }
  }

  const actionDataset = env.TELEMETRY_DEV_EXPERIMENT_DATASET || undefined;
  const dataset = options.dataset ?? actionDataset;

  if (dataset === undefined) {
    throw new TypeError("Pass dataset, or set the dataset input of the experiment action");
  }

  if (actionDataset !== undefined && actionDataset !== dataset) {
    throw new TypeError(
      `dataset is ${dataset}, but the experiment action names ${actionDataset}. Name one dataset.`,
    );
  }

  const { datasetVersion } = options;

  if (
    datasetVersion !== undefined &&
    !(Number.isSafeInteger(datasetVersion) && datasetVersion >= 1)
  ) {
    throw new RangeError("datasetVersion must be a positive integer");
  }

  const evaluators = options.evaluators;

  if (
    evaluators !== undefined &&
    (!Array.isArray(evaluators) ||
      evaluators.length > MAX_EVALUATORS ||
      !evaluators.every((name) => typeof name === "string" && validScoreName(name)))
  ) {
    throw new TypeError(`evaluators must be a list of at most ${MAX_EVALUATORS} evaluator names`);
  }

  const scorers = Object.entries(options.scorers ?? {});
  const evaluatorNames = new Set(evaluators?.map((name) => name.toLowerCase()));

  if (scorers.length > MAX_EVALUATORS) {
    throw new TypeError(`scorers can have at most ${MAX_EVALUATORS} scorers`);
  }

  for (const [name, scorer] of scorers) {
    if (!validScoreName(name) || typeof scorer !== "function") {
      throw new TypeError(
        `scorers must map score names of 1 to ${MAX_SCORE_NAME_LENGTH} characters to functions`,
      );
    }

    if (name === USER_FEEDBACK) {
      throw new TypeError(
        `${USER_FEEDBACK} is the score of user feedback. Name the scorer another way.`,
      );
    }

    if (evaluatorNames.has(name.toLowerCase())) {
      throw new TypeError(`The scorer ${name} has the name of an evaluator. Rename one of them.`);
    }
  }

  const minScores = { ...options.minScores, ...minScoresFromEnv(env) };

  if (
    Object.keys(minScores).length > MAX_MIN_SCORES ||
    !Object.entries(minScores).every(
      ([name, value]) => validScoreName(name) && Number.isFinite(value),
    )
  ) {
    throw new TypeError(`minScores must map at most ${MAX_MIN_SCORES} score names to numbers`);
  }

  const runsPerItem = options.runsPerItem ?? 1;

  if (!Number.isInteger(runsPerItem) || runsPerItem < 1 || runsPerItem > MAX_RUNS_PER_ITEM) {
    throw new RangeError(`runsPerItem must be an integer from 1 to ${MAX_RUNS_PER_ITEM}`);
  }

  const concurrency = options.concurrency ?? 4;

  if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > MAX_CONCURRENCY) {
    throw new RangeError(`concurrency must be an integer from 1 to ${MAX_CONCURRENCY}`);
  }

  if (options.metadata !== undefined && !isRecord(options.metadata)) {
    throw new TypeError("metadata must be an object");
  }

  if (options.ci !== undefined && typeof options.ci !== "boolean") {
    throw new TypeError("ci must be a boolean");
  }

  const ci = options.ci ?? inCi(env);

  return {
    dataset,
    task: options.task,
    name: options.name,
    datasetVersion,
    split: options.split,
    evaluators,
    scorers,
    minScores: Object.keys(minScores).length > 0 ? minScores : undefined,
    runsPerItem,
    concurrency,
    metadata: options.metadata,
    ci: ci ? githubRun(env) : undefined,
    inCi: ci,
  };
}

/** A trimmed name of 1 to 64 characters, as the API stores score names. */
function validScoreName(name: string): boolean {
  return name.length > 0 && name.length <= MAX_SCORE_NAME_LENGTH && name.trim() === name;
}

function warnAboutClient(): void {
  const environment = currentClient().core?.config.environment;

  if (environment === undefined) {
    diag.warn(
      'runExperiment: no telemetry client, so the items record no traces. Call init({ environment: "experiment" }) first.',
    );
  } else if (environment.trim().toLowerCase() !== EXPERIMENT_ENVIRONMENT) {
    diag.warn(
      `runExperiment: the traces of the items go to the environment ${environment}. Call init({ environment: "experiment" }) to keep them out of it.`,
    );
  }
}

/**
 * The runs of each item of the dataset version, page by page, so that only one page is in memory.
 * Workers share the generator: an async generator answers each `next()` in turn.
 */
async function* itemRuns(
  api: Authed,
  run: Run,
  first: DatasetPage,
): AsyncGenerator<ExperimentItem> {
  let page = first;
  let count = 0;

  for (;;) {
    count += page.items.length;

    if (count > MAX_DATASET_ITEMS) {
      throw new ExperimentError(
        "invalid_response",
        `telemetry.dev returned more than ${MAX_DATASET_ITEMS} items`,
      );
    }

    for (const item of page.items) {
      // A copy for each run, so that a task that changes its input does not change the next run.
      for (let runIndex = 0; runIndex < run.runsPerItem; runIndex++) {
        yield { ...structuredClone(item), runIndex };
      }
    }

    if (page.nextCursor === null) return;

    page = await fetchPage(api, run, page.version, page.nextCursor);
  }
}

/** One page of a dataset version. A page after the first names the version of the first. */
async function fetchPage(
  api: Authed,
  run: Run,
  version: number | undefined,
  cursor: string | undefined,
): Promise<DatasetPage> {
  const page = parseDatasetPage(
    await request(api, ExperimentError, "GET", `/v1/datasets/${encodeURIComponent(run.dataset)}`, {
      query: { version: version?.toString(), split: run.split, cursor },
      timeoutMs: REQUEST_TIMEOUT_MS,
      idempotent: true,
    }),
  );

  if (
    !page ||
    (version !== undefined && page.version !== version) ||
    (page.nextCursor !== null && (page.items.length === 0 || page.nextCursor === cursor))
  ) {
    throw new ExperimentError(
      "invalid_response",
      "telemetry.dev returned a dataset it cannot read",
    );
  }

  return page;
}

/**
 * Runs the items, `concurrency` at a time, and posts each result. A post or a page fetch that
 * fails stops the run: no new item starts, and runItems rejects when the running items end.
 */
async function runItems(
  api: Authed,
  run: Run,
  experiment: Created,
  runs: AsyncGenerator<ExperimentItem>,
): Promise<void> {
  const batch = itemBatch((results) =>
    request(
      api,
      ExperimentError,
      "POST",
      `/v1/experiments/${encodeURIComponent(experiment.id)}/items`,
      // The server skips an item run that it has, so a repeat is safe.
      { body: { items: results }, timeoutMs: REQUEST_TIMEOUT_MS, idempotent: true },
    ),
  );

  let failure: { error: unknown } | undefined;

  const worker = async () => {
    while (failure === undefined) {
      const next = await runs.next().catch((error: unknown) => {
        failure ??= { error };

        return { done: true as const, value: undefined };
      });

      if (next.done) return;

      await runItem(run, experiment, next.value, api.onError)
        .then((result) => batch.add(result))
        .catch((error: unknown) => {
          failure ??= { error };
        });
    }
  };

  await Promise.all(Array.from({ length: run.concurrency }, worker));

  if (failure === undefined) {
    await batch.flush().catch((error: unknown) => {
      failure = { error };
    });
  }

  if (failure !== undefined) {
    await runs.return(undefined);

    throw failure.error;
  }
}

/** Collects item results and posts them in order, one post at a time. */
function itemBatch(post: (results: ItemResult[]) => Promise<unknown>) {
  let pending: ItemResult[] = [];
  let bytes = 0;
  let queue: Promise<unknown> = Promise.resolve();

  const send = async () => {
    const results = pending;
    pending = [];
    bytes = 0;
    await post(results);
  };

  return {
    add(result: ItemResult): Promise<unknown> {
      const size = new TextEncoder().encode(JSON.stringify(result)).length;

      queue = queue.then(async () => {
        if (pending.length > 0 && bytes + size > MAX_BATCH_BYTES) await send();
        pending.push(result);
        bytes += size;

        if (pending.length >= MAX_BATCH_ITEMS) await send();
      });

      return queue;
    },
    flush(): Promise<unknown> {
      queue = queue.then(() => (pending.length > 0 ? send() : undefined));

      return queue;
    },
  };
}

async function runItem(
  run: Run,
  experiment: Created,
  item: ExperimentItem,
  onError: ((cause: Error) => void) | undefined,
): Promise<ItemResult> {
  const started = performance.now();

  const outcome = await startActiveSpan(
    `experiment ${experiment.name}`,
    { parent: itemContext(experiment.id, item), input: item.input },
    async (span): Promise<{ output?: unknown; error?: unknown; traceId?: string }> => {
      try {
        const output: unknown = await run.task(item);
        span.update({ output });

        return { output, traceId: traceIdOf(span) };
      } catch (error) {
        span.update({ error });

        return {
          error: error ?? new Error("The task threw a nullish value"),
          traceId: traceIdOf(span),
        };
      }
    },
  );

  const base = {
    itemId: item.id,
    runIndex: item.runIndex,
    traceId: outcome.traceId,
    latencyMs: Math.round(performance.now() - started),
  };

  if ("error" in outcome) {
    return { ...base, output: null, error: truncate(errorText(outcome.error), MAX_ERROR_LENGTH) };
  }

  const scores = await runScorers(run, outcome.output, item, onError);

  return {
    ...base,
    output: outputText(outcome.output),
    scores: scores.length > 0 ? scores : undefined,
  };
}

/**
 * The context of one item run. It drops the active span, so that the run starts a trace of its
 * own, and gives the run a session id of its own: an ambient session or `sessionMode: "process"`
 * would otherwise put all runs in one session trace.
 */
function itemContext(experimentId: string, item: ExperimentItem): Context {
  const base = trace.deleteSpan(activeContext());

  return base.setValue(PROPAGATED_KEY, {
    ...propagatedFromContext(base),
    "td.experiment.id": experimentId,
    "td.dataset.item_id": item.id,
    "gen_ai.conversation.id": `${experimentId}:${item.id}:${item.runIndex}`,
  });
}

function traceIdOf(span: SpanHandle): string | undefined {
  return span.isRecording ? span.traceId : undefined;
}

async function runScorers(
  run: Run,
  output: unknown,
  item: ExperimentItem,
  onError: ((cause: Error) => void) | undefined,
): Promise<ItemScore[]> {
  const scores: ItemScore[] = [];

  // After the span of the item ends, so that the model calls of a scorer do not count in the
  // cost of the item.
  for (const [name, scorer] of run.scorers) {
    try {
      const score = toScore(name, await scorer(output, item));

      if (score) scores.push(score);
    } catch (error) {
      reportError(
        onError,
        new ExperimentError(
          "scorer_failed",
          `The scorer ${name} failed on item ${item.number}: ${errorText(error)}`,
          undefined,
          { cause: error },
        ),
      );
    }
  }

  return scores;
}

/** The score of a scorer result. It throws for a result that the server would reject. */
function toScore(name: string, result: ScorerResult): ItemScore | undefined {
  if (result === null || result === undefined) return undefined;

  let score: ItemScore;

  if (typeof result === "boolean") score = { name, ...passFail(result) };
  else if (typeof result === "number") score = { name, value: result };
  else if (typeof result === "string") score = { name, label: result.trim() };
  else if (isRecord(result)) {
    const { value, label, reason } = result as Record<string, unknown>;

    for (const [key, field] of [
      ["label", label],
      ["reason", reason],
    ] as const) {
      if (field !== undefined && typeof field !== "string") {
        throw new TypeError(`returned a ${key} that is not a string`);
      }
    }

    score = {
      name,
      ...(typeof value === "boolean" ? passFail(value) : { value: value as number | undefined }),
      ...(label === undefined ? {} : { label: (label as string).trim() }),
      reason: reason === undefined ? undefined : truncate(reason as string, MAX_REASON_LENGTH),
    };
  } else {
    throw new TypeError(`returned a ${typeof result}`);
  }

  const problem = checkValue(name, score.value, score.label);

  if (problem) throw new TypeError(`returned a score that is not valid: ${problem}`);

  return score;
}

function passFail(pass: boolean): { value: number; label: string } {
  return pass ? { value: 1, label: "pass" } : { value: 0, label: "fail" };
}

function outputText(output: unknown): string | null {
  const text = output === null || output === undefined ? undefined : jsonAttr(output);

  return text === undefined ? null : truncate(text, MAX_OUTPUT_LENGTH);
}

function errorText(error: unknown): string {
  try {
    return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  } catch {
    return "A value that cannot be converted to a string was thrown";
  }
}

function complete(api: Authed, id: string): Promise<unknown> {
  return request(
    api,
    ExperimentError,
    "POST",
    `/v1/experiments/${encodeURIComponent(id)}/complete`,
    { timeoutMs: REQUEST_TIMEOUT_MS, idempotent: true },
  );
}

/** Waits until the server evaluators finish and the experiment ends. */
async function waitForResult(
  api: Authed,
  id: string,
): Promise<{ result: ExperimentResult; body: unknown }> {
  const deadline = Date.now() + MAX_WAIT_MS;

  for (let interval = 1000; ; interval = Math.min(interval * 2, MAX_POLL_INTERVAL_MS)) {
    const body = await request(
      api,
      ExperimentError,
      "GET",
      `/v1/experiments/${encodeURIComponent(id)}`,
      { timeoutMs: REQUEST_TIMEOUT_MS, idempotent: true },
    );

    const result = parseResult(body);

    if (FINISHED.has(result.status)) return { result, body };

    if (Date.now() + interval > deadline) {
      throw new ExperimentError(
        "timeout",
        `The experiment ${result.name} did not end in ${MAX_WAIT_MS / 60_000} minutes. ${result.url}`,
        undefined,
        { result },
      );
    }

    await sleep(interval);
  }
}

/** In CI mode, a failed check or an experiment that did not complete fails the run. */
function checkResult(result: ExperimentResult): void {
  if (result.status !== "completed") {
    throw new ExperimentError(
      "experiment_failed",
      `The experiment ${result.name} ${result.status}${result.error ? `: ${result.error}` : ""}. ${result.url}`,
      undefined,
      { result },
    );
  }

  if (result.checkStatus !== "failed") return;

  const failed = result.scores
    .filter((score) => score.passed === false)
    .map(
      (score) =>
        `${score.name} is ${formatScore(score.kind, score.value)}, under the pass mark ${formatScore(score.kind, score.minScore)}`,
    );

  if (result.itemsFailed > 0) failed.unshift(itemsFailedText(result));

  throw new ExperimentError(
    "check_failed",
    `The check of ${result.name} failed${failed.length > 0 ? `: ${failed.join("; ")}` : ""}. ${result.url}`,
    undefined,
    { result },
  );
}

/** "5 of 6 items failed": a failed item has no score, so it fails the check. */
function itemsFailedText(result: ExperimentResult): string {
  return `${result.itemsFailed} of ${result.itemsDone} items failed`;
}

function describe(result: ExperimentResult): string {
  const outcome =
    result.status !== "completed"
      ? `${result.status}${result.error ? `: ${result.error}` : ""}`
      : result.checkStatus === null
        ? "completed"
        : `completed. The check ${result.checkStatus}`;

  return [
    `Experiment ${result.name} ${outcome}. ${result.url}`,
    ...(result.itemsFailed > 0 ? [`  ${itemsFailedText(result)}.`] : []),
    ...result.scores.map(
      (score) =>
        `  ${score.name}: ${formatScore(score.kind, score.value)}` +
        (score.minScore === null ? "" : ` (pass mark ${formatScore(score.kind, score.minScore)})`),
    ),
  ].join("\n");
}

function formatScore(kind: string | null, value: number | null): string {
  if (value === null) return "no score";

  return kind !== null && RATE_KINDS.has(kind)
    ? `${Math.round(value * 100)}%`
    : String(Math.round(value * 100) / 100);
}

function parseCreated(body: unknown): Created {
  if (
    !isRecord(body) ||
    typeof body.id !== "string" ||
    typeof body.name !== "string" ||
    typeof body.url !== "string" ||
    !Number.isSafeInteger(body.itemsTotal)
  ) {
    throw new ExperimentError(
      "invalid_response",
      "telemetry.dev returned an experiment it cannot read",
    );
  }

  return { id: body.id, name: body.name, itemsTotal: body.itemsTotal as number, url: body.url };
}

function parseDatasetPage(body: unknown): DatasetPage | undefined {
  if (!isRecord(body) || !isRecord(body.dataset) || !Array.isArray(body.items)) return undefined;

  const { version } = body.dataset;
  const nextCursor = body.nextCursor ?? null;

  if (!Number.isSafeInteger(version) || (version as number) < 0) return undefined;

  if (nextCursor !== null && typeof nextCursor !== "string") return undefined;

  const items: DatasetItem[] = [];

  for (const item of body.items) {
    if (
      !isRecord(item) ||
      typeof item.id !== "string" ||
      !Number.isSafeInteger(item.number) ||
      !isRecord(item.input) ||
      (item.expectedOutput !== null && typeof item.expectedOutput !== "string") ||
      !isRecord(item.metadata) ||
      !Array.isArray(item.tags) ||
      !item.tags.every((tag) => typeof tag === "string") ||
      (item.split !== null && typeof item.split !== "string")
    ) {
      return undefined;
    }

    items.push({
      id: item.id,
      number: item.number as number,
      input: item.input,
      expectedOutput: item.expectedOutput,
      metadata: item.metadata,
      tags: item.tags as string[],
      split: item.split,
    });
  }

  return { version: version as number, items, nextCursor };
}

function parseResult(body: unknown): ExperimentResult {
  const scores = isRecord(body) && Array.isArray(body.scores) ? body.scores.map(parseScore) : [];

  if (
    !isRecord(body) ||
    typeof body.id !== "string" ||
    typeof body.name !== "string" ||
    typeof body.status !== "string" ||
    typeof body.url !== "string" ||
    // Required: it decides the result of CI mode, so a missing one must not read as no check.
    !(
      body.checkStatus === null ||
      body.checkStatus === "passed" ||
      body.checkStatus === "failed"
    ) ||
    typeof body.itemsTotal !== "number" ||
    typeof body.itemsDone !== "number" ||
    !(body.itemsFailed === undefined || typeof body.itemsFailed === "number") ||
    !(body.error === undefined || body.error === null || typeof body.error === "string") ||
    !Array.isArray(body.scores) ||
    scores.some((score) => score === undefined)
  ) {
    throw new ExperimentError("invalid_response", "telemetry.dev returned a result it cannot read");
  }

  return {
    id: body.id,
    name: body.name,
    status: body.status,
    checkStatus: body.checkStatus,
    itemsTotal: body.itemsTotal,
    itemsDone: body.itemsDone,
    itemsFailed: body.itemsFailed ?? 0,
    error: body.error ?? null,
    scores: scores as ExperimentScore[],
    url: body.url,
  };
}

/** A score of the result. A field that can be null can also be missing. */
function parseScore(raw: unknown): ExperimentScore | undefined {
  if (!isRecord(raw)) return undefined;

  const score = {
    name: raw.name,
    kind: raw.kind ?? null,
    value: raw.value ?? null,
    passCount: raw.passCount,
    failCount: raw.failCount,
    baselineValue: raw.baselineValue ?? null,
    delta: raw.delta ?? null,
    minScore: raw.minScore ?? null,
    passed: raw.passed ?? null,
    itemsFailed: raw.itemsFailed ?? null,
  };
  const numbers = [
    score.value,
    score.baselineValue,
    score.delta,
    score.minScore,
    score.itemsFailed,
  ];

  if (
    typeof score.name !== "string" ||
    !(score.kind === null || typeof score.kind === "string") ||
    typeof score.passCount !== "number" ||
    typeof score.failCount !== "number" ||
    !numbers.every((field) => field === null || typeof field === "number") ||
    !(score.passed === null || typeof score.passed === "boolean")
  ) {
    return undefined;
  }

  return score as ExperimentScore;
}
