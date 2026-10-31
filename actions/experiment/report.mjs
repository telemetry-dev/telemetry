// Writes the job summary of the experiment action, and the comment with the result on the pull
// request. runExperiment and run_experiment add one line of JSON to the results file for each
// experiment: the body of GET /v1/experiments/{id}.
import { createHash } from "node:crypto";
import { appendFileSync, readFileSync } from "node:fs";

const RATE_KINDS = new Set(["pass_fail", "up_down"]);
/**
 * Pages of 100 comments that the action reads to find its comment from an earlier push: the first
 * page, then the newest pages.
 */
const MAX_COMMENT_PAGES = 10;
/** The longest comment body that GitHub takes, in characters. */
const MAX_COMMENT_LENGTH = 65_536;
const REQUEST_TIMEOUT_MS = 15_000;
/** The login of the GITHUB_TOKEN, which cannot read GET /user. */
const ACTIONS_BOT = "github-actions[bot]";
const STATUS = { completed: "Completed", failed: "Failed", stopped: "Stopped" };

const env = process.env;
const exitCode = Number.parseInt(env.EXPERIMENT_EXIT_CODE ?? "", 10);
const results = readResults(env.TELEMETRY_DEV_EXPERIMENT_RESULTS);
const leg = matrixLeg(env.MATRIX);
const workingDirectory = env.INPUT_WORKING_DIRECTORY || ".";
// The comment of this job, matrix leg, and input, so that a new push changes it and a second step
// adds its own. A job with no matrix that runs in the default directory keeps the key of the
// earlier versions of the action, so that it still changes its comment from before an update.
const key = createHash("sha256")
  .update(
    [
      env.GITHUB_WORKFLOW,
      env.GITHUB_JOB,
      env.INPUT_DATASET,
      env.INPUT_SCRIPT,
      env.INPUT_COMMAND,
      ...(leg ? [JSON.stringify(leg)] : []),
      ...(workingDirectory === "." ? [] : [`working-directory=${workingDirectory}`]),
    ].join("\0"),
  )
  .digest("hex")
  .slice(0, 12);
const marker = `<!-- telemetry-dev-experiment:${key} -->`;
const report = render(results, exitCode);

if (env.GITHUB_STEP_SUMMARY) appendFileSync(env.GITHUB_STEP_SUMMARY, `${report}\n`);

const last = results.at(-1);

if (env.GITHUB_OUTPUT) {
  appendFileSync(
    env.GITHUB_OUTPUT,
    `check-status=${last?.checkStatus ?? "none"}\nurl=${last ? (httpsUrl(last.url) ?? "") : ""}\n`,
  );
}

// A script that goes on after a failed experiment exits with 0, and the last step of the action
// fails the job only for an exit code that is not 0.
const failures = exitCode === 0 ? results.flatMap(failure) : [];

for (const message of failures) command("error", message);

const commentInput = env.INPUT_COMMENT ?? "true";

// Version 1 posts the comment for each value but false, thus another value only gets a warning.
if (!["true", "True", "TRUE", "false", "False", "FALSE"].includes(commentInput)) {
  command("warning", "The comment input is not true or false, thus the action reads it as true.");
}

if (!["false", "False", "FALSE"].includes(commentInput)) {
  await comment(report).catch((error) => {
    command(
      "warning",
      `The comment on the pull request failed: ${error instanceof Error ? error.message : String(error)}`,
    );
  });
}

if (failures.length > 0) process.exitCode = 1;

function readResults(path) {
  if (!path) return [];

  let text;

  try {
    text = readFileSync(path, "utf8");
  } catch {
    return [];
  }

  return text
    .split("\n")
    .filter((line) => line.trim() !== "")
    .flatMap((line) => {
      try {
        const result = JSON.parse(line);

        return typeof result === "object" && result !== null && typeof result.name === "string"
          ? [result]
          : [];
      } catch {
        return [];
      }
    });
}

/** The values of the matrix of the job, from toJSON(matrix), or undefined with no matrix. */
function matrixLeg(json) {
  try {
    const matrix = JSON.parse(json ?? "null");

    return typeof matrix === "object" &&
      matrix !== null &&
      !Array.isArray(matrix) &&
      Object.keys(matrix).length > 0
      ? matrix
      : undefined;
  } catch {
    return undefined;
  }
}

function failure(experiment) {
  if (experiment.status !== "completed")
    return [
      `The experiment ${experiment.name} did not complete. Its status is ${experiment.status}.`,
    ];

  return experiment.checkStatus === "failed"
    ? [`The check of the experiment ${experiment.name} failed.`]
    : [];
}

function render(experiments, code) {
  const lines = ["### telemetry.dev experiment", ""];

  if (leg) {
    const values = Object.entries(leg).map(
      ([name, value]) =>
        `${text(name)}\\=${text(typeof value === "string" ? value : JSON.stringify(value))}`,
    );

    lines.push(`Matrix: ${values.join(", ")}`, "");
  }

  if (env.ACTION_ERROR) {
    lines.push(`The action stopped before it ran the script: ${text(env.ACTION_ERROR)}`);
  } else if (experiments.length === 0) {
    lines.push(
      `The experiment script exited with code ${Number.isNaN(code) ? "unknown" : code} before it wrote a result. The log of the step has the error.`,
    );
  }

  for (const experiment of experiments) {
    const url = httpsUrl(experiment.url);
    const name = url ? `[${text(experiment.name)}](${url})` : text(experiment.name);
    // The counts are of runs: with runsPerItem above 1, each item runs more than one time.
    const total = Number(experiment.itemsTotal) || 0;
    const failed = Number(experiment.itemsFailed) || 0;
    const completed = Math.max((Number(experiment.itemsDone) || 0) - failed, 0);

    lines.push(
      `**${name}**: ${outcome(experiment)}. ${completed} of ${total} runs completed${failed > 0 ? `, ${failed} failed` : ""}.`,
    );

    if (typeof experiment.error === "string" && experiment.error !== "") {
      lines.push("", `> ${text(experiment.error)}`);
    }

    const scores = Array.isArray(experiment.scores) ? experiment.scores : [];

    if (scores.length > 0) {
      lines.push(
        "",
        "| Score | Value | Change | Pass mark | Result |",
        "| --- | --- | --- | --- | --- |",
      );

      for (const score of scores) {
        lines.push(
          `| ${text(score.name)} | ${value(score.kind, score.value)} | ${change(score.kind, score.delta)} | ${value(score.kind, score.minScore)} | ${score.passed === true ? "Passed" : score.passed === false ? "Failed" : "—"} |`,
        );
      }
    }

    lines.push("");
  }

  if (code !== 0 && experiments.length > 0) {
    lines.push(
      `The experiment script exited with code ${Number.isNaN(code) ? "unknown" : code}.`,
      "",
    );
  }

  return lines.join("\n").trimEnd();
}

function outcome(experiment) {
  if (experiment.status !== "completed")
    return STATUS[experiment.status] ?? text(String(experiment.status));

  if (experiment.checkStatus === "failed") return "Check failed";

  return experiment.checkStatus === "passed" ? "Check passed" : "Completed";
}

/** A rate as a whole percent, and a mean with one decimal, as the experiment page shows them. */
function value(kind, number) {
  if (typeof number !== "number") return "—";

  return RATE_KINDS.has(kind) ? `${Math.round(number * 100)}%` : number.toFixed(1);
}

function change(kind, delta) {
  if (typeof delta !== "number") return "—";

  if (delta === 0) return "0";

  const size = RATE_KINDS.has(kind) ? `${Math.abs(delta)} pts` : Math.abs(delta).toFixed(1);

  return `${delta > 0 ? "↑" : "↓"} ${size}`;
}

/**
 * Text that cannot break the table, or add HTML, a link, a mention, or an issue reference to the
 * comment. GitHub finds mentions and issue references in the rendered text, thus a backslash does
 * not stop them, but a word joiner after the @, the #, or the GH- does.
 */
function text(value) {
  return String(value)
    .replace(/[\r\n]+/g, " ")
    .replace(/[@#]|\bGH-/gi, "$&\u2060")
    .replace(/[!-/:-@[-`{-~]/g, "\\$&");
}

function httpsUrl(url) {
  return typeof url === "string" && /^https:\/\/[^\s()<>]+$/.test(url) ? url : undefined;
}

/** A workflow command, which ends at the first new line of its message. */
function command(kind, message) {
  console.log(
    `::${kind} title=telemetry.dev experiment::${message.replace(/%/g, "%25").replace(/\r/g, "%0D").replace(/\n/g, "%0A")}`,
  );
}

async function comment(report) {
  const pull = pullRequest();

  if (pull === undefined || !env.GITHUB_REPOSITORY) return;

  if (!env.GITHUB_TOKEN) {
    command("warning", "The github-token input is empty, thus the action wrote no comment.");

    return;
  }

  const body = commentBody(report, pull);
  const api = (env.GITHUB_API_URL ?? "https://api.github.com").replace(/\/+$/, "");
  const issue = `${api}/repos/${env.GITHUB_REPOSITORY}/issues`;
  // Anyone can paste the marker into a comment, so the action changes only a comment that its token
  // wrote. A user token knows its login. The GITHUB_TOKEN and app tokens cannot read /user. The
  // GITHUB_TOKEN writes as github-actions[bot], and an app token as the bot of its app, whose login
  // the action cannot get.
  const login =
    (await github("GET", `${api}/user`).then(
      ({ body: user }) => (typeof user?.login === "string" ? user.login : undefined),
      () => undefined,
    )) ?? (env.DEFAULT_GITHUB_TOKEN === "true" ? ACTIONS_BOT : undefined);
  const ours = (item) =>
    login === undefined ? item.user?.type === "Bot" : item.user?.login === login;
  const find = (page) =>
    (Array.isArray(page.body) ? page.body : []).find(
      (item) => typeof item.body === "string" && item.body.startsWith(marker) && ours(item),
    );
  const list = (page) =>
    github("GET", `${issue}/${pull.number}/comments?per_page=100&page=${page}`);
  const first = await list(1);
  let existing = find(first);

  // The list starts with the oldest comments. On a long pull request the comment can be past the
  // pages that the action reads, and then the comment that it adds is on the newest page.
  for (
    let page = lastPage(first.link), read = 1;
    existing === undefined && page > 1 && read < MAX_COMMENT_PAGES;
    page--, read++
  ) {
    existing = find(await list(page));
  }

  if (existing) await github("PATCH", `${issue}/comments/${existing.id}`, { body });
  else await github("POST", `${issue}/${pull.number}/comments`, { body });
}

/**
 * The marker, the report, and the commit and the run of the result. The report is cut at a line
 * when the comment is too long for GitHub. The job summary keeps all of it.
 */
function commentBody(report, pull) {
  const server = (env.GITHUB_SERVER_URL ?? "https://github.com").replace(/\/+$/, "");
  const repository = `${server}/${env.GITHUB_REPOSITORY}`;
  const footer = [
    pull.headSha &&
      `Commit [${pull.headSha.slice(0, 7)}](${repository}/pull/${pull.number}/commits/${pull.headSha}).`,
    /^\d+$/.test(env.GITHUB_RUN_ID ?? "") &&
      `The [workflow run](${repository}/actions/runs/${env.GITHUB_RUN_ID}) has the log.`,
  ]
    .filter(Boolean)
    .join(" ");
  const body = `${marker}\n${report}\n\n${footer}`.trimEnd();

  if (body.length <= MAX_COMMENT_LENGTH) return body;

  const cut = "The comment is cut here. The job summary of the workflow run has all of the result.";
  let room = MAX_COMMENT_LENGTH - `${marker}\n\n\n${cut}\n\n${footer}`.length;
  const kept = [];

  for (const line of report.split("\n")) {
    room -= line.length + 1;

    if (room < 0) break;

    kept.push(line);
  }

  return `${marker}\n${kept.join("\n")}\n\n${cut}\n\n${footer}`.trimEnd();
}

/** The number of the last page in the Link header of a list, or 1 when the list has one page. */
function lastPage(link) {
  const match = /<[^>]*[?&]page=(\d+)[^>]*>;\s*rel="last"/.exec(link ?? "");

  return match ? Number(match[1]) : 1;
}

function pullRequest() {
  if (!env.GITHUB_EVENT_PATH) return undefined;

  try {
    const pull = JSON.parse(readFileSync(env.GITHUB_EVENT_PATH, "utf8")).pull_request;
    const sha = pull?.head?.sha;

    return Number.isSafeInteger(pull?.number) && pull.number > 0
      ? {
          number: pull.number,
          headSha: typeof sha === "string" && /^[0-9a-f]{40,64}$/.test(sha) ? sha : undefined,
        }
      : undefined;
  } catch {
    return undefined;
  }
}

/** The JSON body of the answer of GitHub, and its Link header. */
async function github(method, url, body) {
  const call = `${method} ${new URL(url).pathname}`;
  const headers = {
    accept: "application/vnd.github+json",
    authorization: `Bearer ${env.GITHUB_TOKEN}`,
    "x-github-api-version": "2022-11-28",
  };
  let res;
  let raw;

  try {
    res = await fetch(url, {
      method,
      ...(body
        ? {
            headers: { ...headers, "content-type": "application/json" },
            body: JSON.stringify(body),
          }
        : { headers }),
      redirect: "error",
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    // The timeout also stops a body that does not end.
    raw = await res.text();
  } catch (error) {
    // fetch puts the reason of a network error or a redirect in the cause. The cause of a refused
    // connection to a host with two addresses has an empty message and a code.
    const reason =
      error?.name === "TimeoutError"
        ? `GitHub did not answer in ${REQUEST_TIMEOUT_MS / 1000} s`
        : error?.cause?.message || error?.cause?.code || error?.message || String(error);

    throw new Error(`${call} failed: ${reason}`);
  }

  let answer;

  try {
    answer = JSON.parse(raw);
  } catch {
    answer = undefined;
  }

  if (!res.ok) {
    const reasons = [
      answer?.message,
      ...(Array.isArray(answer?.errors) ? answer.errors.map((error) => error?.message) : []),
    ].filter((reason) => typeof reason === "string" && reason !== "");
    const hint =
      res.status === 403
        ? " The token needs the pull-requests: write permission, and a pull request from a fork gets a token that cannot write."
        : "";

    throw new Error(
      `GitHub returned ${res.status} for ${call}${reasons.length > 0 ? `: ${reasons.join(". ")}` : ""}.${hint}`,
    );
  }

  return { body: answer, link: res.headers.get("link") };
}
