// Writes the job summary of the experiment action, and the comment with the result on the pull
// request. runExperiment and run_experiment add one line of JSON to the results file for each
// experiment: the body of GET /v1/experiments/{id}.
import { createHash } from "node:crypto";
import { appendFileSync, readFileSync } from "node:fs";

const RATE_KINDS = new Set(["pass_fail", "up_down"]);
/** Pages of 100 comments that the action reads to find its comment from an earlier push. */
const MAX_COMMENT_PAGES = 10;
const STATUS = { completed: "Completed", failed: "Failed", stopped: "Stopped" };

const env = process.env;
const exitCode = Number.parseInt(env.EXPERIMENT_EXIT_CODE ?? "", 10);
const results = readResults(env.TELEMETRY_DEV_EXPERIMENT_RESULTS);
// The comment of this job and input, so that a new push changes it and a second step adds its own.
const key = createHash("sha256")
  .update(
    [
      env.GITHUB_WORKFLOW,
      env.GITHUB_JOB,
      env.INPUT_DATASET,
      env.INPUT_SCRIPT,
      env.INPUT_COMMAND,
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

if (env.INPUT_COMMENT !== "false") {
  await comment(`${marker}\n${report}`).catch((error) => {
    // A pull request from a fork gets a token that cannot write comments.
    console.log(
      `::warning title=telemetry.dev experiment::The comment on the pull request failed: ${error.message}`,
    );
  });
}

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

function render(experiments, code) {
  const lines = ["### telemetry.dev experiment", ""];

  if (experiments.length === 0) {
    lines.push(
      `The experiment script exited with code ${Number.isNaN(code) ? "unknown" : code} before it wrote a result. The log of the step has the error.`,
    );
  }

  for (const experiment of experiments) {
    const url = httpsUrl(experiment.url);
    const name = url ? `[${text(experiment.name)}](${url})` : text(experiment.name);
    const total = Number(experiment.itemsTotal) || 0;
    const failed = Number(experiment.itemsFailed) || 0;
    const ran = Math.max((Number(experiment.itemsDone) || 0) - failed, 0);

    lines.push(
      `**${name}**: ${outcome(experiment)}. ${ran} of ${total} items ran${failed > 0 ? `, ${failed} failed` : ""}.`,
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

function value(kind, number) {
  if (typeof number !== "number") return "—";

  return RATE_KINDS.has(kind)
    ? `${Math.round(number * 100)}%`
    : String(Math.round(number * 100) / 100);
}

function change(kind, delta) {
  if (typeof delta !== "number") return "—";

  if (delta === 0) return "0";

  const size = RATE_KINDS.has(kind) ? `${Math.abs(delta)} pts` : String(Math.abs(delta));

  return `${delta > 0 ? "↑" : "↓"} ${size}`;
}

/** Text that cannot break the table or add HTML to the comment. */
function text(value) {
  return String(value)
    .replace(/[\r\n]+/g, " ")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\|/g, "\\|")
    .replace(/([[\]()*_`])/g, "\\$1");
}

function httpsUrl(url) {
  return typeof url === "string" && /^https:\/\/[^\s()<>]+$/.test(url) ? url : undefined;
}

async function comment(body) {
  const pull = pullRequestNumber();

  if (pull === undefined || !env.GITHUB_TOKEN || !env.GITHUB_REPOSITORY) return;

  const api = (env.GITHUB_API_URL ?? "https://api.github.com").replace(/\/+$/, "");
  const issue = `${api}/repos/${env.GITHUB_REPOSITORY}/issues`;
  // Anyone can paste the marker into a comment, so the action changes only a comment that its token
  // wrote. A user token knows its login. The GITHUB_TOKEN and app tokens cannot read /user, and
  // they write as a bot.
  const login = await github("GET", `${api}/user`).then(
    (user) => (typeof user?.login === "string" ? user.login : undefined),
    () => undefined,
  );
  const ours = (item) =>
    login === undefined ? item.user?.type === "Bot" : item.user?.login === login;
  let existing;

  for (let page = 1; page <= MAX_COMMENT_PAGES && existing === undefined; page++) {
    const comments = await github("GET", `${issue}/${pull}/comments?per_page=100&page=${page}`);

    existing = comments.find(
      (item) => typeof item.body === "string" && item.body.startsWith(marker) && ours(item),
    );

    if (comments.length < 100) break;
  }

  if (existing) await github("PATCH", `${issue}/comments/${existing.id}`, { body });
  else await github("POST", `${issue}/${pull}/comments`, { body });
}

function pullRequestNumber() {
  if (!env.GITHUB_EVENT_PATH) return undefined;

  try {
    const number = JSON.parse(readFileSync(env.GITHUB_EVENT_PATH, "utf8")).pull_request?.number;

    return Number.isSafeInteger(number) && number > 0 ? number : undefined;
  } catch {
    return undefined;
  }
}

async function github(method, url, body) {
  const headers = {
    accept: "application/vnd.github+json",
    authorization: `Bearer ${env.GITHUB_TOKEN}`,
    "x-github-api-version": "2022-11-28",
  };
  const res = await fetch(url, {
    method,
    ...(body
      ? { headers: { ...headers, "content-type": "application/json" }, body: JSON.stringify(body) }
      : { headers }),
    redirect: "error",
    signal: AbortSignal.timeout(15_000),
  });

  if (!res.ok)
    throw new Error(`GitHub returned ${res.status} for ${method} ${new URL(url).pathname}`);

  return res.json();
}
