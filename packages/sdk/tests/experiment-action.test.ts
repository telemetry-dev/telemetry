import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, expect, test } from "vitest";

// The experiment action: run.sh runs the eval script, and report.mjs writes the job summary and
// the comment on the pull request.
const action = fileURLToPath(new URL("../../../actions/experiment/", import.meta.url));
const closers: (() => void)[] = [];

afterEach(() => {
  for (const close of closers.splice(0)) close();
});

function workdir(): string {
  return mkdtempSync(join(tmpdir(), "td-action-"));
}

function outputs(path: string): Record<string, string> {
  return Object.fromEntries(
    readFileSync(path, "utf8")
      .split("\n")
      .filter(Boolean)
      .map((line) => line.split(/=(.*)/s).slice(0, 2) as [string, string]),
  );
}

function runScript(env: Record<string, string>, files: Record<string, string> = {}) {
  const dir = workdir();
  const output = join(dir, "output");
  writeFileSync(output, "");

  for (const [name, content] of Object.entries(files)) writeFileSync(join(dir, name), content);

  const run = spawnSync("bash", [join(action, "run.sh")], {
    cwd: dir,
    encoding: "utf8",
    env: { PATH: process.env.PATH, RUNNER_TEMP: dir, GITHUB_OUTPUT: output, ...env },
  });

  return { dir, run, outputs: outputs(output) };
}

const result = {
  id: "6f9619ff-8b86-4011-b42d-00c04fc964ff",
  name: "reply-v13-gpt-4o",
  status: "completed",
  checkStatus: "failed",
  itemsTotal: 120,
  itemsDone: 120,
  itemsFailed: 2,
  error: null,
  scores: [
    {
      name: "correctness",
      kind: "pass_fail",
      value: 0.85,
      passCount: 100,
      failCount: 18,
      baselineValue: 0.88,
      delta: -3,
      minScore: 0.9,
      passed: false,
    },
    {
      name: "helpfulness",
      kind: "number",
      value: 4.436,
      passCount: 0,
      failCount: 0,
      baselineValue: 4.1,
      delta: 0.3,
      minScore: null,
      passed: null,
    },
  ],
  url: "https://telemetry.dev/o/org/p/project/evals/experiments/6f9619ff-8b86-4011-b42d-00c04fc964ff",
};

const summary = `### telemetry.dev experiment

**[reply-v13-gpt-4o](https://telemetry.dev/o/org/p/project/evals/experiments/6f9619ff-8b86-4011-b42d-00c04fc964ff)**: Check failed. 118 of 120 items ran, 2 failed.

| Score | Value | Change | Pass mark | Result |
| --- | --- | --- | --- | --- |
| correctness | 85% | ↓ 3 pts | 90% | Failed |
| helpfulness | 4.44 | ↑ 0.3 | — | — |

The experiment script exited with code 1.`;

interface GithubRequest {
  method: string;
  url: string;
  body: unknown;
}

/** The comment routes of the GitHub REST API for one pull request. */
async function fakeGithub(
  options: { comments?: { id: number; body: string }[]; status?: number } = {},
) {
  const requests: GithubRequest[] = [];
  const server = createServer(async (req: IncomingMessage, res) => {
    let raw = "";
    for await (const chunk of req) raw += chunk;
    requests.push({ method: req.method!, url: req.url!, body: raw ? JSON.parse(raw) : undefined });
    res.writeHead(options.status ?? 200, { "content-type": "application/json" });
    res.end(JSON.stringify(req.method === "GET" ? (options.comments ?? []) : { id: 9 }));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  closers.push(() => server.close());

  return { url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, requests };
}

async function report(env: Record<string, string>, lines: unknown[]) {
  const dir = workdir();
  const results = join(dir, "results.jsonl");
  const stepSummary = join(dir, "summary.md");
  const output = join(dir, "output");
  const event = join(dir, "event.json");
  writeFileSync(results, lines.map((line) => `${JSON.stringify(line)}\n`).join(""));
  writeFileSync(stepSummary, "");
  writeFileSync(output, "");
  writeFileSync(event, JSON.stringify({ pull_request: { number: 42 } }));

  const child = spawn("node", [join(action, "report.mjs")], {
    env: {
      PATH: process.env.PATH,
      TELEMETRY_DEV_EXPERIMENT_RESULTS: results,
      GITHUB_STEP_SUMMARY: stepSummary,
      GITHUB_OUTPUT: output,
      GITHUB_EVENT_PATH: event,
      GITHUB_REPOSITORY: "acme/support",
      GITHUB_TOKEN: "ghs_test",
      GITHUB_WORKFLOW: "evals",
      GITHUB_JOB: "reply",
      INPUT_DATASET: "support-golden",
      EXPERIMENT_EXIT_CODE: "1",
      ...env,
    },
  });
  let stdout = "";
  child.stdout.on("data", (chunk) => (stdout += chunk));
  const code = await new Promise<number | null>((resolve) => child.on("close", resolve));

  return { code, stdout, summary: readFileSync(stepSummary, "utf8"), outputs: outputs(output) };
}

test("run.sh passes the inputs to the script and records its exit code without failing", () => {
  const {
    dir,
    run,
    outputs: written,
  } = runScript({
    INPUT_COMMAND: `node -e 'const fs = require("node:fs"); fs.writeFileSync("env.json", JSON.stringify(process.env)); process.exit(3)'`,
    INPUT_API_KEY: "td_live_secret",
    INPUT_DATASET: "support-golden",
    INPUT_MIN_SCORES: "correctness=0.9",
    INPUT_BASE_URL: "https://ingest.test",
  });

  expect(run.status).toBe(0);
  expect(run.stdout).toContain("::add-mask::td_live_secret");
  expect(written["exit-code"]).toBe("3");

  const env = JSON.parse(readFileSync(join(dir, "env.json"), "utf8"));
  expect(env).toMatchObject({
    TELEMETRY_DEV_API_KEY: "td_live_secret",
    TELEMETRY_DEV_BASE_URL: "https://ingest.test",
    TELEMETRY_DEV_EXPERIMENT_DATASET: "support-golden",
    TELEMETRY_DEV_EXPERIMENT_MIN_SCORES: "correctness=0.9",
    TELEMETRY_DEV_ENVIRONMENT: "experiment",
    TELEMETRY_DEV_EXPERIMENT_RESULTS: written.results,
  });
  expect(written.results!.startsWith(dir)).toBe(true);
  expect(statSync(written.results!).mode & 0o777).toBe(0o600);
});

test("run.sh keeps the environment of the job, and runs a .js script with node", () => {
  const { run, outputs: written } = runScript(
    {
      INPUT_SCRIPT: "print-env.mjs",
      TELEMETRY_DEV_API_KEY: "td_live_job",
      TELEMETRY_DEV_ENVIRONMENT: "staging",
    },
    { "print-env.mjs": "console.log(JSON.stringify(process.env));\n" },
  );

  expect(written["exit-code"]).toBe("0");
  expect(JSON.parse(run.stdout)).toMatchObject({
    TELEMETRY_DEV_API_KEY: "td_live_job",
    TELEMETRY_DEV_ENVIRONMENT: "staging",
  });
});

test.each([
  [
    "script and command",
    { INPUT_SCRIPT: "a.ts", INPUT_COMMAND: "true", TELEMETRY_DEV_API_KEY: "k" },
    "Give script or command, not the two.",
  ],
  [
    "no script and no command",
    { TELEMETRY_DEV_API_KEY: "k" },
    "Give the script or the command that runs the experiment.",
  ],
  [
    "no API key",
    { INPUT_COMMAND: "true" },
    "Set the api-key input or the TELEMETRY_DEV_API_KEY variable.",
  ],
  [
    "a script of another type",
    { INPUT_SCRIPT: "eval.rb", TELEMETRY_DEV_API_KEY: "k" },
    "The script must end in .ts, .js, or .py. For another runner, give a command.",
  ],
])("run.sh stops with an error for %s", (_, env, message) => {
  const { run, outputs: written } = runScript(env);

  expect(run.stdout).toContain(`::error title=telemetry.dev experiment::${message}`);
  expect(written["exit-code"]).toBe("2");
});

test("report.mjs writes the summary, posts it on the pull request, and sets the outputs", async () => {
  const github = await fakeGithub();

  const {
    code,
    summary: written,
    outputs: set,
  } = await report({ GITHUB_API_URL: github.url }, [result]);

  expect(code).toBe(0);
  expect(written).toBe(`${summary}\n`);
  expect(set).toEqual({ "check-status": "failed", url: result.url });
  expect(github.requests.map((request) => `${request.method} ${request.url}`)).toEqual([
    "GET /repos/acme/support/issues/42/comments?per_page=100&page=1",
    "POST /repos/acme/support/issues/42/comments",
  ]);
  expect(github.requests[1]!.body).toEqual({
    body: expect.stringMatching(
      /^<!-- telemetry-dev-experiment:[0-9a-f]{12} -->\n### telemetry\.dev experiment\n/,
    ),
  });
  expect((github.requests[1]!.body as { body: string }).body.endsWith(summary)).toBe(true);
});

test("report.mjs changes its comment from an earlier push, and a step with another dataset adds its own", async () => {
  const first = await fakeGithub();
  await report({ GITHUB_API_URL: first.url }, [result]);
  const own = { id: 8, body: (first.requests[1]!.body as { body: string }).body };

  const other = await fakeGithub({ comments: [own] });
  await report({ GITHUB_API_URL: other.url, INPUT_DATASET: "billing-golden" }, [result]);

  const again = await fakeGithub({ comments: [{ id: 7, body: "Looks good" }, own] });
  await report({ GITHUB_API_URL: again.url }, [result]);

  expect(other.requests.map((request) => request.method)).toEqual(["GET", "POST"]);
  expect(again.requests.map((request) => `${request.method} ${request.url}`)).toEqual([
    "GET /repos/acme/support/issues/42/comments?per_page=100&page=1",
    "PATCH /repos/acme/support/issues/comments/8",
  ]);
});

test("report.mjs warns and succeeds when the token cannot write the comment", async () => {
  const github = await fakeGithub({ status: 403 });

  const { code, stdout } = await report({ GITHUB_API_URL: github.url }, [result]);

  expect(code).toBe(0);
  expect(stdout).toContain(
    "::warning title=telemetry.dev experiment::The comment on the pull request failed: GitHub returned 403",
  );
});

test("report.mjs reports a script that ended before it wrote a result, and comment: false posts nothing", async () => {
  const github = await fakeGithub();

  const { summary: written, outputs: set } = await report(
    { GITHUB_API_URL: github.url, INPUT_COMMENT: "false", EXPERIMENT_EXIT_CODE: "1" },
    [],
  );

  expect(github.requests).toEqual([]);

  expect(written).toBe(
    "### telemetry.dev experiment\n\nThe experiment script exited with code 1 before it wrote a result. The log of the step has the error.\n",
  );
  expect(set).toEqual({ "check-status": "none", url: "" });
});

test("report.mjs escapes names, so that they cannot break the table or add HTML", async () => {
  const github = await fakeGithub();
  const { summary: written } = await report(
    { GITHUB_API_URL: github.url, INPUT_COMMENT: "false", EXPERIMENT_EXIT_CODE: "0" },
    [
      {
        ...result,
        name: "<img src=x>|*bold*",
        url: "javascript:alert(1)",
        scores: [{ ...result.scores[0], name: "a|b\nc" }],
      },
    ],
  );

  expect(written).toContain("**&lt;img src=x&gt;\\|\\*bold\\***: Check failed.");
  expect(written).toContain("| a\\|b c | 85% |");
  expect(written).not.toContain("javascript:");
});
