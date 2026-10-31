import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, statSync, writeFileSync } from "node:fs";
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
      minScore: 4,
      passed: true,
    },
    {
      name: "tone",
      kind: "up_down",
      value: 0.6,
      passCount: 3,
      failCount: 2,
      baselineValue: null,
      delta: null,
      minScore: null,
      passed: null,
    },
  ],
  url: "https://telemetry.dev/o/org/p/project/evals/experiments/6f9619ff-8b86-4011-b42d-00c04fc964ff",
};

const summary = `### telemetry.dev experiment

**[reply\\-v13\\-gpt\\-4o](https://telemetry.dev/o/org/p/project/evals/experiments/6f9619ff-8b86-4011-b42d-00c04fc964ff)**: Check failed. 118 of 120 runs completed, 2 failed.

| Score | Value | Change | Pass mark | Result |
| --- | --- | --- | --- | --- |
| correctness | 85% | ↓ 3 pts | 90% | Failed |
| helpfulness | 4.4 | ↑ 0.3 | 4.0 | Passed |
| tone | 60% | — | — | — |

The experiment script exited with code 1.`;

interface GithubRequest {
  method: string;
  url: string;
  body: unknown;
}

interface GithubComment {
  id: number;
  body: string;
  user: { login: string; type: "Bot" | "User" };
}

const BOT = { login: "github-actions[bot]", type: "Bot" } as const;
const HEAD_SHA = "0123456789abcdef0123456789abcdef01234567";

/**
 * The comment routes of the GitHub REST API for one pull request, with pages of the comments as
 * GitHub gives them. Without `user`, GET /user fails as it does for the GITHUB_TOKEN. `fail` answers
 * the requests of its method, or each request, with its status and body.
 */
async function fakeGithub(
  options: {
    comments?: GithubComment[];
    user?: { login: string };
    fail?: { method?: string; status: number; body?: unknown };
  } = {},
) {
  const requests: GithubRequest[] = [];
  const server = createServer(async (req: IncomingMessage, res) => {
    let raw = "";
    for await (const chunk of req) raw += chunk;
    requests.push({ method: req.method!, url: req.url!, body: raw ? JSON.parse(raw) : undefined });

    if (req.url === "/user") {
      res.writeHead(options.user ? 200 : 403, { "content-type": "application/json" });
      res.end(
        JSON.stringify(options.user ?? { message: "Resource not accessible by integration" }),
      );

      return;
    }

    const { fail } = options;

    if (fail && (fail.method === undefined || fail.method === req.method)) {
      res.writeHead(fail.status, {
        "content-type": "application/json",
        location: "https://github.com/login",
      });
      res.end(JSON.stringify(fail.body ?? {}));

      return;
    }

    if (req.method !== "GET") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ id: 9 }));

      return;
    }

    const url = new URL(req.url!, "http://github.test");
    const page = Number(url.searchParams.get("page"));
    const size = Number(url.searchParams.get("per_page"));
    const comments = options.comments ?? [];
    const last = Math.max(Math.ceil(comments.length / size), 1);
    const pageUrl = (number: number) => {
      url.searchParams.set("page", String(number));

      return url.href;
    };
    res.writeHead(200, {
      "content-type": "application/json",
      ...(page < last
        ? { link: `<${pageUrl(page + 1)}>; rel="next", <${pageUrl(last)}>; rel="last"` }
        : {}),
    });
    res.end(JSON.stringify(comments.slice((page - 1) * size, page * size)));
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
  writeFileSync(event, JSON.stringify({ pull_request: { number: 42, head: { sha: HEAD_SHA } } }));

  const child = spawn("node", [join(action, "report.mjs")], {
    env: {
      PATH: process.env.PATH,
      TELEMETRY_DEV_EXPERIMENT_RESULTS: results,
      GITHUB_STEP_SUMMARY: stepSummary,
      GITHUB_OUTPUT: output,
      GITHUB_EVENT_PATH: event,
      GITHUB_REPOSITORY: "acme/support",
      GITHUB_SERVER_URL: "https://github.com",
      GITHUB_RUN_ID: "4242",
      GITHUB_TOKEN: "ghs_test",
      DEFAULT_GITHUB_TOKEN: "true",
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

test("run.sh runs a .ts script with the tsx of the project, from a directory above", () => {
  const project = workdir();
  const evals = join(project, "evals");
  const output = join(project, "output");
  mkdirSync(join(project, "node_modules", ".bin"), { recursive: true });
  mkdirSync(evals);
  writeFileSync(output, "");
  // A stand-in for the tsx of the project that records the script it gets.
  writeFileSync(
    join(project, "node_modules", ".bin", "tsx"),
    `#!/bin/sh\necho "$1" > "${join(project, "ran")}"\n`,
    { mode: 0o755 },
  );

  const run = spawnSync("bash", [join(action, "run.sh")], {
    cwd: evals,
    encoding: "utf8",
    env: {
      PATH: process.env.PATH,
      RUNNER_TEMP: project,
      GITHUB_OUTPUT: output,
      INPUT_SCRIPT: "reply.eval.ts",
      TELEMETRY_DEV_API_KEY: "td_live_job",
    },
  });

  expect(run.status).toBe(0);
  expect(readFileSync(join(project, "ran"), "utf8")).toBe("reply.eval.ts\n");
  expect(outputs(output)["exit-code"]).toBe("0");
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
    { INPUT_COMMAND: "touch ran" },
    "Set the api-key input or the TELEMETRY_DEV_API_KEY variable.",
  ],
  [
    "a script of another type",
    { INPUT_SCRIPT: "eval.rb", TELEMETRY_DEV_API_KEY: "k" },
    "The script must end in .ts, .js, or .py. For another runner, give a command.",
  ],
  [
    "a .ts script in a project without tsx",
    { INPUT_SCRIPT: "reply.eval.ts", TELEMETRY_DEV_API_KEY: "k" },
    "No tsx in node_modules. Add tsx to the devDependencies of the project and install them, or give a command.",
  ],
])(
  "run.sh stops with an error for %s, which the report does not blame on the script",
  async (_, env, message) => {
    const { dir, run, outputs: written } = runScript(env);

    expect(existsSync(join(dir, "ran"))).toBe(false);
    expect(run.stdout).toContain(`::error title=telemetry.dev experiment::${message}`);
    expect(written["exit-code"]).toBe("2");
    expect(written.error).toBe(message);

    const { summary: reported } = await report(
      { ACTION_ERROR: written.error!, EXPERIMENT_EXIT_CODE: "2", INPUT_COMMENT: "false" },
      [],
    );

    expect(reported).toMatch(
      /^### telemetry\.dev experiment\n\nThe action stopped before it ran the script: [^\n]+\n$/,
    );
  },
);

/** The comment that the action writes for `report`, with the commit and the run of the job. */
function posted(report: string): string {
  return `${report}\n\nCommit [0123456](https://github.com/acme/support/pull/42/commits/${HEAD_SHA}). The [workflow run](https://github.com/acme/support/actions/runs/4242) has the log.`;
}

/** The marker of a step, as the key of the earlier versions of the action makes it. */
function marker(parts: string[]): string {
  return `<!-- telemetry-dev-experiment:${createHash("sha256").update(parts.join("\0")).digest("hex").slice(0, 12)} -->`;
}

test("report.mjs writes the summary, posts it on the pull request, and sets the outputs", async () => {
  const github = await fakeGithub();

  const {
    code,
    summary: written,
    outputs: set,
  } = await report(
    {
      GITHUB_API_URL: github.url,
      MATRIX: "null",
      INPUT_WORKING_DIRECTORY: ".",
      INPUT_COMMENT: "TRUE",
    },
    [result],
  );

  expect(code).toBe(0);
  expect(written).toBe(`${summary}\n`);
  expect(set).toEqual({ "check-status": "failed", url: result.url });
  expect(github.requests.map((request) => `${request.method} ${request.url}`)).toEqual([
    "GET /user",
    "GET /repos/acme/support/issues/42/comments?per_page=100&page=1",
    "POST /repos/acme/support/issues/42/comments",
  ]);
  // A job with no matrix in the default directory keeps the marker of its comment from before.
  expect(github.requests[2]!.body).toEqual({
    body: `${marker(["evals", "reply", "support-golden", "", ""])}\n${posted(summary)}`,
  });
});

test("report.mjs fails the job for a failed experiment before the last, when the script exits with 0", async () => {
  // The lines of an SDK that writes the CI mode.
  const failed = { ...result, ci: true, runsPerItem: 1 };
  const passed = { ...failed, name: "reply-short", checkStatus: "passed", itemsFailed: 0 };
  const stopped = { ...passed, name: "reply-stopped", status: "stopped", checkStatus: null };
  // A failed check and a stopped experiment of a script that runs with the CI mode off.
  const notInCi = { ...result, name: "reply-local", ci: false, runsPerItem: 3 };
  const stoppedNotInCi = { ...stopped, name: "reply-local-stopped", ci: false };

  const failedFirst = await report({ EXPERIMENT_EXIT_CODE: "0", INPUT_COMMENT: "false" }, [
    failed,
    stopped,
    passed,
  ]);
  // A line of an SDK from before the ci field, which version 1.0.0 did not fail the job for.
  const olderSdk = await report({ EXPERIMENT_EXIT_CODE: "0", INPUT_COMMENT: "false" }, [result]);
  const allPassed = await report({ EXPERIMENT_EXIT_CODE: "0", INPUT_COMMENT: "false" }, [passed]);
  const offCi = await report({ EXPERIMENT_EXIT_CODE: "0", INPUT_COMMENT: "false" }, [
    notInCi,
    stoppedNotInCi,
  ]);

  expect(failedFirst.code).toBe(1);
  expect(failedFirst.stdout).toContain(
    "::error title=telemetry.dev experiment::The check of the experiment reply-v13-gpt-4o failed.",
  );
  expect(failedFirst.stdout).toContain(
    "::error title=telemetry.dev experiment::The experiment reply-stopped did not complete. Its status is stopped.",
  );
  // check-status stays the one of the last experiment, as version 1 says.
  expect(failedFirst.outputs["check-status"]).toBe("passed");
  expect(failedFirst.summary).not.toContain("runs each");
  expect(failedFirst.summary).not.toContain("does not fail the job");
  expect(olderSdk.code).toBe(0);
  expect(olderSdk.stdout).not.toContain("::error");
  expect(olderSdk.summary).not.toContain("does not fail the job");
  expect(allPassed.code).toBe(0);
  expect(allPassed.outputs["check-status"]).toBe("passed");
  expect(offCi.code).toBe(0);
  expect(offCi.stdout).not.toContain("::error");
  expect(offCi.summary).toContain(
    "Check failed. 40 items, 3 runs each: 118 of 120 runs completed, 2 failed.\n\n" +
      "The CI mode of this experiment was off, thus it does not fail the job.",
  );
});

/** The body of the comment that the action posts for the result. */
async function commentBody(env: Record<string, string> = {}): Promise<string> {
  const github = await fakeGithub();
  await report({ GITHUB_API_URL: github.url, ...env }, [result]);

  return (github.requests.find((request) => request.method === "POST")!.body as { body: string })
    .body;
}

function changes(github: Awaited<ReturnType<typeof fakeGithub>>): string[] {
  return github.requests
    .filter((request) => request.method !== "GET")
    .map((request) => `${request.method} ${request.url}`);
}

test("report.mjs changes its comment from an earlier push, and a step with another dataset adds its own", async () => {
  const own = { id: 8, body: await commentBody(), user: BOT };

  const other = await fakeGithub({ comments: [own] });
  await report({ GITHUB_API_URL: other.url, INPUT_DATASET: "billing-golden" }, [result]);

  const again = await fakeGithub({
    comments: [{ id: 7, body: "Looks good", user: { login: "dana", type: "User" } }, own],
  });
  await report({ GITHUB_API_URL: again.url }, [result]);

  expect(changes(other)).toEqual(["POST /repos/acme/support/issues/42/comments"]);
  expect(changes(again)).toEqual(["PATCH /repos/acme/support/issues/comments/8"]);
});

test("report.mjs gives each matrix leg and working directory its own comment, which names the leg", async () => {
  const haiku = { MATRIX: '{\n  "model": "claude-haiku-4-5"\n}' };
  const own = { id: 8, body: await commentBody(haiku), user: BOT };

  const sonnet = await fakeGithub({ comments: [own] });
  await report({ GITHUB_API_URL: sonnet.url, MATRIX: '{"model":"claude-sonnet-4-6"}' }, [result]);
  const directory = await fakeGithub({ comments: [own] });
  await report({ GITHUB_API_URL: directory.url, ...haiku, INPUT_WORKING_DIRECTORY: "billing" }, [
    result,
  ]);
  const again = await fakeGithub({ comments: [own] });
  await report({ GITHUB_API_URL: again.url, ...haiku }, [result]);

  expect(own.body).toContain(
    "### telemetry.dev experiment\n\nMatrix: model\\=claude\\-haiku\\-4\\-5\n\n**",
  );
  expect(changes(sonnet)).toEqual(["POST /repos/acme/support/issues/42/comments"]);
  expect(changes(directory)).toEqual(["POST /repos/acme/support/issues/42/comments"]);
  expect(changes(again)).toEqual(["PATCH /repos/acme/support/issues/comments/8"]);
});

test.each([
  ["on the newest page", 1005, 10, [1, 2, 3, 4, 5, 11]],
  ["on an old page", 149, 1000, [1, 2]],
  ["on the last old page that it reads", 449, 700, [1, 2, 3, 4, 5]],
])(
  "report.mjs finds its comment %s of a pull request with more than 1,000 comments",
  async (_, before, after, pages) => {
    const own = { id: before + 1, body: await commentBody(), user: BOT };
    const github = await fakeGithub({
      comments: [...others(1, before), own, ...others(before + 2, after)],
    });

    await report({ GITHUB_API_URL: github.url }, [result]);

    expect(listedPages(github)).toEqual(pages);
    expect(changes(github)).toEqual([`PATCH /repos/acme/support/issues/comments/${before + 1}`]);
  },
);

test.each([
  ["12 pages", 1150, [1, 2, 3, 4, 5, 12, 11, 10, 9, 8]],
  ["7 pages", 650, [1, 2, 3, 4, 5, 7, 6]],
])(
  "report.mjs reads each page once on a pull request of %s without its comment, and adds the comment",
  async (_, count, pages) => {
    const github = await fakeGithub({ comments: others(1, count) });

    await report({ GITHUB_API_URL: github.url }, [result]);

    expect(listedPages(github)).toEqual(pages);
    expect(changes(github)).toEqual(["POST /repos/acme/support/issues/42/comments"]);
  },
);

/** Comments by a person, with the ids from `from`. */
function others(from: number, count: number) {
  return Array.from({ length: count }, (_, index) => ({
    id: from + index,
    body: "Looks good",
    user: { login: "dana", type: "User" as const },
  }));
}

/** The pages of comments that the action read, in order. */
function listedPages(github: Awaited<ReturnType<typeof fakeGithub>>): number[] {
  return github.requests
    .filter((request) => request.method === "GET" && request.url.includes("/comments?"))
    .map((request) => Number(new URL(request.url, "http://github.test").searchParams.get("page")));
}

test.each([
  ["a person pasted under the GITHUB_TOKEN", undefined, { login: "mallory", type: "User" }],
  ["another app wrote under the GITHUB_TOKEN", undefined, { login: "helper[bot]", type: "Bot" }],
  ["another bot wrote under a user token", { login: "dana" }, BOT],
] as const)(
  "report.mjs does not change a comment with its marker that %s",
  async (_, user, author) => {
    const github = await fakeGithub({
      user,
      comments: [{ id: 7, body: await commentBody(), user: author }],
    });

    await report({ GITHUB_API_URL: github.url }, [result]);

    expect(changes(github)).toEqual(["POST /repos/acme/support/issues/42/comments"]);
  },
);

test.each([
  ["the user of a user token", { login: "dana" }, { login: "dana", type: "User" }],
  ["an app, under the token of an app", undefined, { login: "helper[bot]", type: "Bot" }],
] as const)("report.mjs changes the comment of %s", async (_, user, author) => {
  const github = await fakeGithub({
    user,
    comments: [{ id: 7, body: await commentBody(), user: author }],
  });

  await report({ GITHUB_API_URL: github.url, DEFAULT_GITHUB_TOKEN: "false" }, [result]);

  expect(changes(github)).toEqual(["PATCH /repos/acme/support/issues/comments/7"]);
});

test.each([
  [
    "a token that cannot write",
    { status: 403, body: { message: "Resource not accessible by integration" } },
    "GitHub returned 403 for GET /repos/acme/support/issues/42/comments: Resource not accessible by integration. The token needs the pull-requests: write permission, and a pull request from a fork gets a token that cannot write.",
  ],
  [
    "a comment that is too long",
    {
      method: "POST",
      status: 422,
      body: {
        message: "Validation Failed",
        errors: [{ field: "body", message: "body is too long (maximum is 65536 characters)" }],
      },
    },
    "GitHub returned 422 for POST /repos/acme/support/issues/42/comments: Validation Failed. body is too long (maximum is 65536 characters).",
  ],
  [
    "a redirect",
    { method: "POST", status: 301 },
    "POST /repos/acme/support/issues/42/comments failed: unexpected redirect",
  ],
])("report.mjs warns with the reason and succeeds for %s", async (_, fail, reason) => {
  const github = await fakeGithub({ fail });

  const { code, stdout } = await report({ GITHUB_API_URL: github.url }, [result]);

  expect(code).toBe(0);
  expect(stdout).toContain(
    `::warning title=telemetry.dev experiment::The comment on the pull request failed: ${reason}\n`,
  );
});

test("report.mjs cuts a comment that is too long for GitHub, and the summary keeps all of it", async () => {
  const scores = Array.from({ length: 800 }, (_, index) => ({
    ...result.scores[0],
    name: `${"&".repeat(40)} ${index}`,
  }));
  const github = await fakeGithub();

  const { summary: written } = await report({ GITHUB_API_URL: github.url }, [
    { ...result, scores },
  ]);
  const body = (
    github.requests.find((request) => request.method === "POST")!.body as {
      body: string;
    }
  ).body;

  expect(written.length).toBeGreaterThan(65_536);
  expect(written).toContain(" 799 | 85% |");
  expect(body.length).toBeLessThanOrEqual(65_536);
  expect(body.startsWith(`${marker(["evals", "reply", "support-golden", "", ""])}\n`)).toBe(true);
  expect(body).toMatch(
    /\| 85% \| ↓ 3 pts \| 90% \| Failed \|\n\nThe comment is cut here\. The job summary of the workflow run has all of the result\.\n\nCommit \[0123456\]/,
  );
});

test("report.mjs reports a script that ended before it wrote a result, and comment: false posts nothing", async () => {
  const github = await fakeGithub();

  const { summary: written, outputs: set } = await report(
    { GITHUB_API_URL: github.url, INPUT_COMMENT: "False", EXPERIMENT_EXIT_CODE: "1" },
    [],
  );

  expect(github.requests).toEqual([]);

  expect(written).toBe(
    "### telemetry.dev experiment\n\nThe experiment script exited with code 1 before it wrote a result. The log of the step has the error.\n",
  );
  expect(set).toEqual({ "check-status": "none", url: "" });
});

test("report.mjs warns that it wrote no comment when the github-token input is empty", async () => {
  const github = await fakeGithub();

  const { code, stdout } = await report({ GITHUB_API_URL: github.url, GITHUB_TOKEN: "" }, [result]);

  expect(github.requests).toEqual([]);
  expect(code).toBe(0);
  expect(stdout).toContain(
    "::warning title=telemetry.dev experiment::The github-token input is empty, thus the action wrote no comment.",
  );
});

test("report.mjs warns for a comment input that is not true or false, and posts as version 1 did", async () => {
  const github = await fakeGithub();

  const { stdout } = await report({ GITHUB_API_URL: github.url, INPUT_COMMENT: "no" }, [result]);

  expect(stdout).toContain(
    "::warning title=telemetry.dev experiment::The comment input is not true or false, thus the action reads it as true.",
  );
  expect(changes(github)).toEqual(["POST /repos/acme/support/issues/42/comments"]);
});

test("report.mjs escapes names, so that they cannot break the table, add HTML, or mention someone", async () => {
  const github = await fakeGithub();
  const { summary: written } = await report(
    { GITHUB_API_URL: github.url, INPUT_COMMENT: "false", EXPERIMENT_EXIT_CODE: "0" },
    [
      {
        ...result,
        name: "<img src=x>|*bold* @octocat",
        url: "javascript:alert(1)",
        scores: [{ ...result.scores[0], name: "a\\|b\nc ~~s~~ fixes #1, GH-2 https://e.com/x" }],
      },
    ],
  );

  expect(written).toContain("**\\<img src\\=x\\>\\|\\*bold\\* \\@\u2060octocat**: Check failed.");
  expect(written).toContain(
    "| a\\\\\\|b c \\~\\~s\\~\\~ fixes \\#\u20601\\, GH\\-\u20602 https\\:\\/\\/e\\.com\\/x | 85% |",
  );
  expect(written).not.toContain("javascript:");
});
