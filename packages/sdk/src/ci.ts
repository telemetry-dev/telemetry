import type * as fs from "node:fs";

import { reportError } from "@telemetry-dev/otel";

/** The pull request and the workflow run of an experiment that runs in GitHub Actions. */
export interface CiRun {
  provider: "github_actions";
  /** `owner/name`. */
  repository: string;
  commit: string;
  prNumber?: number;
  prTitle?: string;
  prUrl?: string;
  runUrl?: string;
}

type Env = Record<string, string | undefined>;

type RuntimeGlobal = typeof globalThis & {
  process?: { getBuiltinModule?: (id: string) => unknown };
};

const REPOSITORY = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const COMMIT = /^[0-9a-f]{7,64}$/i;
const HTTPS_URL = /^https:\/\/\S+$/;
/** A plain decimal, so that the TypeScript and Python SDKs read the same marks. */
const DECIMAL = /^-?\d+(\.\d+)?$/;
const MAX_REPOSITORY_LENGTH = 200;
const MAX_URL_LENGTH = 2_048;
const MAX_PR_TITLE_LENGTH = 256;

/** CI mode is on when `CI` is `true` or `1`, as CI services set it. */
export function inCi(env: Env): boolean {
  return env.CI === "true" || env.CI === "1";
}

/**
 * The run that GitHub Actions describes in its variables, or undefined outside GitHub Actions. On a
 * pull request the commit is the head of the pull request: `GITHUB_SHA` is then a merge commit
 * that nobody pushed.
 */
export function githubRun(env: Env): CiRun | undefined {
  const repository = env.GITHUB_REPOSITORY;

  if (
    env.GITHUB_ACTIONS !== "true" ||
    !repository ||
    repository.length > MAX_REPOSITORY_LENGTH ||
    !REPOSITORY.test(repository)
  ) {
    return undefined;
  }

  const pull = pullRequest(env.GITHUB_EVENT_PATH);
  const commit = pull?.commit ?? env.GITHUB_SHA;

  if (!commit || !COMMIT.test(commit)) return undefined;

  const server = (env.GITHUB_SERVER_URL ?? "https://github.com").replace(/\/+$/, "");
  const runId = env.GITHUB_RUN_ID;

  return {
    provider: "github_actions",
    repository,
    commit,
    prNumber: pull?.number,
    prTitle: pull?.title,
    prUrl: pull?.url,
    runUrl:
      runId && /^\d+$/.test(runId)
        ? httpsUrl(`${server}/${repository}/actions/runs/${runId}`)
        : undefined,
  };
}

/**
 * The pass marks in `TELEMETRY_DEV_EXPERIMENT_MIN_SCORES`, which the experiment action sets from
 * its `min-scores` input: `name=value` pairs, separated by commas or new lines.
 */
export function minScoresFromEnv(env: Env): Record<string, number> {
  // Pairs, so that the name __proto__ becomes a key and not the prototype of the object.
  const scores: [string, number][] = [];

  for (const pair of (env.TELEMETRY_DEV_EXPERIMENT_MIN_SCORES ?? "").split(/[,\n]/)) {
    if (pair.trim() === "") continue;
    const [name, value, ...rest] = pair.split("=").map((part) => part.trim());

    if (!name || !value || rest.length > 0 || !DECIMAL.test(value)) {
      throw new TypeError(
        `TELEMETRY_DEV_EXPERIMENT_MIN_SCORES: "${pair.trim()}" is not a name=number pair`,
      );
    }

    scores.push([name, Number(value)]);
  }

  return Object.fromEntries(scores);
}

/**
 * Adds the result to the file in `TELEMETRY_DEV_EXPERIMENT_RESULTS` as one line of JSON. The
 * experiment action reads the file to write the job summary and the comment on the pull request.
 */
export function writeResult(env: Env, result: unknown, onError?: (cause: Error) => void): void {
  const path = env.TELEMETRY_DEV_EXPERIMENT_RESULTS;

  if (!path) return;

  try {
    const fs = nodeFs();

    if (!fs) throw new Error("This runtime cannot write TELEMETRY_DEV_EXPERIMENT_RESULTS");

    fs.appendFileSync(path, `${JSON.stringify(result)}\n`, { mode: 0o600 });
  } catch (error) {
    reportError(onError, error);
  }
}

function pullRequest(
  eventPath: string | undefined,
): { number?: number; title?: string; url?: string; commit?: string } | undefined {
  if (!eventPath) return undefined;

  let event: unknown;

  try {
    event = JSON.parse(nodeFs()?.readFileSync(eventPath, "utf8") ?? "null");
  } catch {
    return undefined;
  }

  const pull = (event as { pull_request?: Record<string, unknown> } | null)?.pull_request;

  if (typeof pull !== "object" || pull === null) return undefined;

  const head = pull.head as { sha?: unknown } | undefined;

  return {
    number:
      Number.isSafeInteger(pull.number) && (pull.number as number) > 0
        ? (pull.number as number)
        : undefined,
    title: typeof pull.title === "string" ? cut(pull.title, MAX_PR_TITLE_LENGTH) : undefined,
    url: typeof pull.html_url === "string" ? httpsUrl(pull.html_url) : undefined,
    commit: typeof head?.sha === "string" ? head.sha : undefined,
  };
}

/**
 * `text` cut to `max` UTF-16 units, with no half of a surrogate pair at the end: the API stores the
 * run as jsonb, and Postgres rejects a lone surrogate there.
 */
function cut(text: string, max: number): string {
  if (text.length <= max) return text;

  const last = text.charCodeAt(max - 1);

  return text.slice(0, last >= 0xd800 && last <= 0xdbff ? max - 1 : max);
}

function httpsUrl(url: string): string | undefined {
  return url.length <= MAX_URL_LENGTH && HTTPS_URL.test(url) ? url : undefined;
}

// Read at call time, with no static `node:` import, so that the package still loads in runtimes
// without a file system.
function nodeFs(): typeof fs | undefined {
  return (globalThis as RuntimeGlobal).process?.getBuiltinModule?.("node:fs") as
    | typeof fs
    | undefined;
}
