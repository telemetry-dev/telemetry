import { execFile, spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readdir, readFile, rm, stat, utimes, writeFile } from "node:fs/promises";
import { type AddressInfo, createServer } from "node:net";
import { hostname, tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

import { afterAll, beforeAll, beforeEach, expect, test, vi } from "vitest";

import { type Session, sessionFile, updateSession } from "../src/session.ts";
import {
  beforeRefresh,
  calls,
  EMAIL,
  PARALLEL_TRACE_ID,
  PAYLOADS,
  PROJECTS,
  QUIET_TRACE_ID,
  QUIET_TRACE_URL,
  REFRESH_TOKEN,
  serverState,
  startFixture,
  TOKEN,
  TRACE_ID,
  traces,
} from "./fixture.ts";

const CLI = fileURLToPath(new URL("../src/cli.ts", import.meta.url));

let fixture: Awaited<ReturnType<typeof startFixture>>;
let home: string;

beforeAll(async () => {
  fixture = await startFixture();
});

afterAll(async () => {
  await fixture.close();
});

const sessionPath = () => join(home, sessionFile(fixture.url));

const signIn = (
  project?: (typeof PROJECTS)[number],
  scope = "openid email offline_access telemetry:read telemetry:payloads",
) =>
  writeFile(
    sessionPath(),
    JSON.stringify({
      server: fixture.url,
      client: { client_id: "fixture-client" },
      tokens: { access_token: TOKEN, token_type: "Bearer", scope },
      email: "ada@acme.dev",
      ...(project ? { project: { id: project.id, name: "support-bot" } } : {}),
    }),
    { mode: 0o600 },
  );

beforeEach(async () => {
  home = await mkdtemp(join(tmpdir(), "telemetry-cli-"));
  calls.length = 0;
  delete beforeRefresh.run;
  Object.assign(serverState, {
    failProjects: false,
    removedFromWorkspace: false,
    errorCodes: true,
    traceUrl: QUIET_TRACE_URL,
  });

  return () => rm(home, { recursive: true, force: true });
});

const runWith = (env: Record<string, string>, ...args: string[]) =>
  new Promise<{ code: number; stdout: string; stderr: string }>((resolve) => {
    execFile(
      process.execPath,
      [CLI, ...args],
      {
        env: {
          PATH: process.env.PATH,
          HOME: home,
          TELEMETRY_DEV_URL: fixture.url,
          TELEMETRY_DEV_CONFIG_DIR: home,
          NO_COLOR: "1",
          ...env,
        },
      },
      (error, stdout, stderr) => resolve({ code: error ? Number(error.code) : 0, stdout, stderr }),
    );
  });

const run = (...args: string[]) => runWith({}, ...args);

/** The address of a port that was just closed, which refuses connections. */
const closedServer = () =>
  new Promise<string>((resolve) => {
    const server = createServer().listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      server.close(() => resolve(`http://127.0.0.1:${port}`));
    });
  });

const CONTROL = new RegExp(String.raw`[\u0000-\u0008\u000b-\u001f\u007f-\u009f]`);

test("human output never passes terminal escapes from captured text, and --json keeps the data", async () => {
  await signIn(PROJECTS[0]);

  const human = await run("traces");
  expect(human.code).toBe(0);
  expect(human.stdout).toContain("triage classify ticket");
  expect(human.stdout).not.toMatch(CONTROL);

  // Project names are written by any workspace member, and `use` echoes the one chosen.
  for (const result of [await run("projects"), await run("use", "eval-runner")]) {
    expect(result.code).toBe(0);
    expect(result.stdout + result.stderr).toContain("eval-runner");
    expect(result.stdout + result.stderr).not.toMatch(CONTROL);
  }

  const json = await run("traces", "--json");
  expect(json.code).toBe(0);
  expect(json.stdout).not.toContain("\u001b");
  expect(JSON.parse(json.stdout).traces.map((trace: { name: string }) => trace.name)).toEqual(
    traces().traces.map((trace) => trace.name),
  );
});

test("--project matches a name in any case, and an unknown name is a usage error that lists the projects", async () => {
  await signIn();

  const matched = await run("traces", "--project", "EVAL-RUNNER", "--errors", "--json");
  expect(matched.code).toBe(0);
  expect(calls.find((call) => call.tool === "search_traces")?.args).toMatchObject({
    project_id: PROJECTS[1]!.id,
    status: "error",
  });

  const unknown = await run("traces", "-p", "billing");
  expect(unknown.code).toBe(2);
  expect(unknown.stderr).toContain("support-bot, eval-runner");
});

test("a workspace with several projects and no default asks for one instead of guessing", async () => {
  await signIn();

  const result = await run("usage");
  expect(result.code).toBe(2);
  expect(result.stderr).toContain("telemetry use");
  expect(calls.some((call) => call.tool === "get_usage_summary")).toBe(false);
});

test("commands exit 4 when signed out, and status reports it", async () => {
  const traced = await run("traces");
  expect(traced.code).toBe(4);
  expect(traced.stderr).toContain("telemetry login");

  const status = await run("status", "--json");
  expect(status.code).toBe(4);
  expect(JSON.parse(status.stdout)).toMatchObject({ signed_in: false });
});

test("payload --field prints the whole text across pages", async () => {
  await signIn(PROJECTS[0]);

  const result = await run(
    "payload",
    "4bf92f3577b34da6a3ce929d0e0e4736",
    "f6",
    "--field",
    "output",
  );
  expect(result.code).toBe(0);
  expect(result.stdout).toBe(PAYLOADS.output);
  expect(calls.filter((call) => call.tool === "get_span_payload").length).toBeGreaterThan(1);
});

test("trace leads with where the run failed and which calls repeated, counting parallel calls once", async () => {
  await signIn(PROJECTS[0]);

  const result = await run("trace", PARALLEL_TRACE_ID);
  expect(result.code).toBe(0);

  const lines = result.stdout.split("\n");
  // Three 1 s searches started 50 ms apart cover 100 ms to 1,200 ms: 1.1 s of the 3 s
  // trace, where adding them up would say 3 s and 100%.
  expect(lines).toContain("  execute_tool web_search  ×3  1.10s  37%");
  // The request that failed, under the tool that made it, under the agent.
  expect(lines).toContain(
    "  invoke_agent researcher › execute_tool fetch_page › GET /page  TimeoutError: Request timed out after 700 ms",
  );
  expect(lines.indexOf("Errors (1)")).toBeGreaterThan(0);
  expect(lines.indexOf("Repeated (1)")).toBeGreaterThan(lines.indexOf("Errors (1)"));
  expect(lines.findIndex((line) => line.startsWith("SPAN"))).toBeGreaterThan(
    lines.indexOf("Repeated (1)"),
  );
  // Only the request where the failure began is listed and offered, not the tool and the
  // agent it passed through.
  expect(result.stderr.match(/telemetry payload .*/g)).toEqual([
    `telemetry payload ${PARALLEL_TRACE_ID} t2 --field=error`,
  ]);

  // Two chat calls that don't overlap cover 1.18 s + 1.42 s of the 4.21 s trace. Measuring
  // from the first start to the last end would say 2.64 s.
  const sequential = await run("trace", TRACE_ID);
  expect(sequential.stdout.split("\n")).toContain("  chat gpt-5  ×2  2.60s  62%");

  // A sign-in without the payloads scope isn't pointed at a command that would refuse it.
  await signIn(PROJECTS[0], "openid email offline_access telemetry:read");
  const unscoped = await run("trace", PARALLEL_TRACE_ID);
  expect(unscoped.code).toBe(0);
  expect(unscoped.stderr).not.toContain("telemetry payload");
});

test("a trace with no failures and no repeated calls goes straight to the span tree", async () => {
  await signIn(PROJECTS[0]);

  const result = await run("trace", QUIET_TRACE_ID);
  expect(result.code).toBe(0);
  expect(result.stdout).toMatch(/^SPAN /m);
  expect(result.stdout).not.toMatch(/^(Errors|Repeated) /m);
  expect(result.stderr).toBe("");
});

test("trace links its dashboard page only when the server sends a web address", async () => {
  await signIn(PROJECTS[0]);

  const lines = (await run("trace", QUIET_TRACE_ID)).stdout.split("\n");
  const ids = lines.findIndex((line) => line.startsWith("  trace "));
  expect(lines[ids + 1]).toBe(`  View: ${QUIET_TRACE_URL}`);

  // A server from before trace links sends no url; the rest aren't web addresses. Each still
  // shows the trace, just without the line.
  for (const url of [undefined, "javascript:alert(1)", "file:///etc/passwd", "not a url"]) {
    serverState.traceUrl = url;
    const result = await run("trace", QUIET_TRACE_ID);
    expect(result.code).toBe(0);
    expect(result.stdout).toMatch(/^SPAN /m);
    expect(result.stdout).not.toContain("View:");
  }

  // An escape sequence in the link can't reach the terminal; the URL parser encodes it.
  serverState.traceUrl = "https://telemetry.dev/traces/\u001b]8;;https://evil.example\u0007x";
  const hostile = await run("trace", QUIET_TRACE_ID);
  expect(hostile.stdout).not.toContain("\u001b");
  expect(hostile.stdout.split("\n")).toContain(
    "  View: https://telemetry.dev/traces/%1B]8;;https://evil.example%07x",
  );

  // --json passes the server's text through as sent; only the terminal line is normalized.
  const json = await run("trace", QUIET_TRACE_ID, "--json");
  expect(JSON.parse(json.stdout).trace.url).toBe(serverState.traceUrl);
});

/** The words a POSIX shell makes of a pasted command line. */
const shellWords = (command: string) =>
  new Promise<string[]>((resolve, reject) => {
    execFile("sh", ["-c", `printf '%s\\n' ${command}`], (error, stdout) =>
      error ? reject(error) : resolve(stdout.trimEnd().split("\n")),
    );
  });

test("suggested next commands run as pasted, in the same project, and never print with --json", async () => {
  await signIn();

  // The newest trace succeeded, so the suggestion names the failed one after it.
  const listed = await run("traces", "-p", "support-bot");
  expect(listed.code).toBe(0);
  const inspect = listed.stderr.match(/Inspect the newest failure: (.*)/)?.[1];
  expect(await shellWords(inspect!)).toEqual([
    "telemetry",
    "trace",
    TRACE_ID,
    "--project=support-bot",
  ]);

  const sessions = await run("sessions", "--project=support-bot");
  expect(sessions.code).toBe(0);
  const words = await shellWords(sessions.stderr.match(/telemetry traces .*/)![0]);
  expect(words).toEqual([
    "telemetry",
    "traces",
    "--session=Ada's thread 42",
    "--project=support-bot",
  ]);

  // Run as pasted, it asks for that session in that project.
  calls.length = 0;
  expect((await run(...words.slice(1))).code).toBe(0);
  expect(calls.find((call) => call.tool === "search_traces")?.args).toMatchObject({
    project_id: PROJECTS[0]!.id,
    session_id: "Ada's thread 42",
  });

  for (const args of [["traces"], ["sessions"], ["trace", PARALLEL_TRACE_ID]]) {
    const result = await run(...args, "-p", "support-bot", "--json");
    expect(result.code).toBe(0);
    expect(result.stderr).toBe("");
  }
});

test("use saves the default project in a file only its owner can read", async () => {
  await signIn();

  const result = await run("use", "eval-runner");
  expect(result.code).toBe(0);

  const path = sessionPath();
  expect((await stat(path)).mode & 0o777).toBe(0o600);
  expect(JSON.parse(await readFile(path, "utf8"))).toMatchObject({
    project: { id: PROJECTS[1]!.id, name: "eval-runner" },
    tokens: { access_token: TOKEN },
  });
});

test("a mistyped command suggests the closest one and exits 2", async () => {
  const result = await run("tarces");
  expect(result.code).toBe(2);
  expect(result.stderr).toContain("telemetry traces");
});

test("input that isn't valid exits 2 before calling telemetry.dev, even from a script", async () => {
  await signIn(PROJECTS[0]);

  const invalid = [
    await run("--bogus"),
    await run("constructor"),
    await run("help", "constructor"),
    await run("usage", "--range", "constructor"),
    await run("logs", "--level", "__proto__"),
    await run("traces", "checkout", "failed"),
    await run("login", "--finish", ""),
    await runWith({ TELEMETRY_DEV_URL: "telemetry.dev" }, "status"),
  ];

  expect(invalid.map((result) => result.code)).toEqual(invalid.map(() => 2));
  expect(calls).toEqual([]);
});

test("with --json, an error is one JSON line on stderr with a code to branch on, and stdout stays empty", async () => {
  const errorOf = async (result: { code: number; stdout: string; stderr: string }) => {
    expect(result.stdout).toBe("");
    expect(result.stderr.trim().split("\n")).toHaveLength(1);

    return { exit: result.code, ...JSON.parse(result.stderr).error };
  };

  expect(await errorOf(await run("traces", "--json"))).toMatchObject({
    exit: 4,
    code: "not_signed_in",
    hint: "telemetry login",
    retryable: false,
    exit_code: 4,
  });

  await signIn(PROJECTS[0]);

  expect(await errorOf(await run("traces", "--limit", "99", "--json"))).toMatchObject({
    exit: 2,
    code: "invalid_arguments",
    exit_code: 2,
  });
  expect(await errorOf(await run("traces", "--project", "nope", "--json"))).toMatchObject({
    exit: 2,
    code: "project_not_found",
    hint: "telemetry projects",
  });
  expect(await errorOf(await run("trace", "0000", "--json"))).toMatchObject({
    exit: 1,
    code: "trace_not_found",
    message: "Trace not found or outside retention",
  });

  const closed = await closedServer();
  await writeFile(
    join(home, sessionFile(closed)),
    JSON.stringify({ server: closed, tokens: { access_token: TOKEN, token_type: "Bearer" } }),
  );
  expect(
    await errorOf(
      await runWith(
        { TELEMETRY_DEV_URL: closed },
        "traces",
        "--project",
        PROJECTS[0]!.id,
        "--json",
      ),
    ),
  ).toMatchObject({ exit: 1, code: "unreachable", retryable: true });

  // After `--`, --json is a value, so the error stays the line a person reads.
  const value = await run("use", "--", "--json");
  expect(value.code).toBe(2);
  expect(value.stderr).toMatch(/^error No project "--json"/);

  // The same failures without --json keep the line a person reads.
  const human = await run("trace", "0000");
  expect(human.code).toBe(1);
  expect(human.stderr).toBe("error Trace not found or outside retention\n");
});

test("the code telemetry.dev gives a tool error becomes the CLI's code, exit code, and hint", async () => {
  const errorOf = async (...args: string[]) => {
    const result = await run(...args, "--json");
    expect(result.stdout).toBe("");

    return { exit: result.code, ...JSON.parse(result.stderr).error };
  };

  // A default project or a project id goes to telemetry.dev without a lookup first.
  const deleted = "0f0e0d0c-0b0a-4908-8706-050403020100";
  await signIn({ ...PROJECTS[0]!, id: deleted });
  expect(await errorOf("traces")).toMatchObject({
    exit: 2,
    code: "project_not_found",
    hint: "telemetry projects",
  });
  expect(await errorOf("traces", "--project", deleted)).toMatchObject({
    code: "project_not_found",
  });
  // People read the CLI's own words, not the text written for an agent.
  expect((await run("traces", "--project", deleted)).stderr).toMatch(
    /^error That project isn't in this workspace/,
  );

  await signIn(PROJECTS[0]);
  expect(await errorOf("traces", "--cursor", "bm90LWEtY3Vyc29y")).toMatchObject({
    exit: 2,
    code: "invalid_cursor",
    retryable: false,
  });

  serverState.failProjects = true;
  expect(await errorOf("projects")).toMatchObject({
    exit: 1,
    code: "server_error",
    message: "telemetry.dev could not complete this request. Try again shortly.",
    retryable: true,
  });

  serverState.removedFromWorkspace = true;
  expect(await errorOf("traces")).toMatchObject({
    exit: 4,
    code: "access_denied",
    hint: "telemetry login",
  });

  // A server from before the codes, and a code this CLI doesn't know, stay tool_error.
  serverState.errorCodes = false;
  expect(await errorOf("traces")).toMatchObject({
    exit: 1,
    code: "tool_error",
    message: "Not a member of this workspace",
    retryable: false,
  });

  const { toolError } = await import("../src/errors.ts");
  expect(
    toolError("Plan limit reached", { "dev.telemetry/error": { code: "plan_limit" } }),
  ).toMatchObject({
    code: "tool_error",
    message: "Plan limit reached",
  });
  expect(
    toolError("…", { "dev.telemetry/error": { code: "payloads_not_allowed", retryable: false } }),
  ).toMatchObject({ code: "payloads_not_allowed", exitCode: 4, hint: "telemetry login" });
  expect(
    toolError("Span not found", {
      "dev.telemetry/error": { code: "span_not_found", retryable: false },
    }),
  ).toMatchObject({ code: "span_not_found", exitCode: 1, message: "Span not found" });
});

test("a request that times out inside fetch is a retryable timeout, not a failed request", async () => {
  const { Api } = await import("../src/api.ts");

  for (const code of [
    "UND_ERR_CONNECT_TIMEOUT",
    "UND_ERR_HEADERS_TIMEOUT",
    "UND_ERR_BODY_TIMEOUT",
  ]) {
    vi.stubGlobal("fetch", () =>
      Promise.reject(
        new TypeError("fetch failed", { cause: Object.assign(new Error(""), { code }) }),
      ),
    );

    const api = new Api({
      server: fixture.url,
      tokens: { access_token: TOKEN, token_type: "Bearer" },
    });

    try {
      await expect(api.call("list_projects")).rejects.toMatchObject({
        code: "timeout",
        retryable: true,
        exitCode: 1,
      });
    } finally {
      vi.unstubAllGlobals();
      await api.close();
    }
  }
});

const savedSession = async () => JSON.parse(await readFile(sessionPath(), "utf8"));

test("a shell without a terminal signs in with --no-browser, then --finish, keeping the old sign-in until then and revoking it after", async () => {
  await writeFile(
    sessionPath(),
    JSON.stringify({
      server: fixture.url,
      client: { client_id: "fixture-client" },
      tokens: { access_token: TOKEN, token_type: "Bearer", refresh_token: "previous-refresh" },
    }),
    { mode: 0o600 },
  );

  const started = await run("login", "--no-browser");
  expect(started.code).toBe(0);
  expect(started.stderr).toContain("telemetry login --finish");

  const link = started.stderr.match(/http:\/\/127\.0\.0\.1:\d+\/authorize\?\S+/)?.[0];
  expect(link).toBeDefined();
  expect(await savedSession()).toMatchObject({ tokens: { access_token: TOKEN }, pending: {} });

  // The browser, after the person selects Allow.
  const approved = await fetch(link!, { redirect: "manual" });
  const address = approved.headers.get("location")!;

  // A code the server refuses, such as an expired one, ends with a reason, not a stack.
  const refusedAddress = new URL(address);
  refusedAddress.searchParams.set("code", "expired");
  const refused = await run("login", "--finish", refusedAddress.href);
  expect(refused.code).toBe(4);
  expect(refused.stderr).toContain("telemetry login");
  expect(refused.stderr).not.toContain("    at ");
  expect(calls.some((call) => call.tool === "oauth:revoke")).toBe(false);
  expect(await savedSession()).toMatchObject({ tokens: { access_token: TOKEN }, pending: {} });

  const finished = await run("login", "--finish", address);
  expect(finished.code).toBe(0);
  expect(finished.stderr).toContain("eval-runner");
  expect(finished.stderr).not.toMatch(CONTROL);

  const session = await savedSession();
  expect(session).toMatchObject({ email: EMAIL, tokens: { refresh_token: REFRESH_TOKEN } });
  expect(session.pending).toBeUndefined();

  // The sign-in it replaced no longer works at the server.
  expect(calls.filter((call) => call.tool === "oauth:revoke").map((call) => call.args)).toEqual([
    expect.objectContaining({ token: "previous-refresh", client_id: "fixture-client" }),
  ]);

  // The fixture rejects a code without the verifier its challenge came from.
  expect(calls.some((call) => call.tool === "oauth:authorization_code")).toBe(true);

  const again = await run("login", "--finish", address);
  expect(again.code).toBe(4);
});

test("login --json prints the link without opening a browser or waiting, and --finish --json prints the sign-in", async () => {
  // A browser that could open: an `open` and `xdg-open` that record being run.
  const bin = await mkdtemp(join(tmpdir(), "telemetry-cli-bin-"));
  const opened = join(bin, "opened");
  for (const name of ["open", "xdg-open"]) {
    await writeFile(join(bin, name), `#!/bin/sh\ntouch '${opened}'\n`, { mode: 0o755 });
  }
  const browser = { PATH: `${bin}:${process.env.PATH}`, DISPLAY: ":0" };

  try {
    const started = await runWith(browser, "login", "--json");
    expect(started).toMatchObject({ code: 0, stderr: "" });

    const pending = JSON.parse(started.stdout);
    expect(pending).toEqual({
      status: "pending",
      authorization_url: expect.stringMatching(/^http:\/\/127\.0\.0\.1:\d+\/authorize\?/),
      next: "telemetry login --finish '<address>'",
    });

    const approved = await fetch(pending.authorization_url, { redirect: "manual" });
    const finished = await run("login", "--finish", approved.headers.get("location")!, "--json");
    expect(finished).toMatchObject({ code: 0, stderr: "" });

    const status = JSON.parse((await run("status", "--json")).stdout);
    expect(status).toMatchObject({ signed_in: true, email: EMAIL });
    expect(JSON.parse(finished.stdout)).toEqual({ status: "signed_in", ...status });

    const again = await run("login", "--finish", approved.headers.get("location")!, "--json");
    expect(again).toMatchObject({ code: 4, stdout: "" });
    expect(JSON.parse(again.stderr).error.code).toBe("no_pending_sign_in");

    expect(JSON.parse((await run("logout", "--json")).stdout)).toEqual({
      was_signed_in: true,
      revoked: true,
    });
    expect(JSON.parse((await run("logout", "--json")).stdout)).toEqual({
      was_signed_in: false,
      revoked: null,
    });

    // `open` runs detached, so by now, several commands later, it would have left its mark.
    await expect(stat(opened)).rejects.toThrow();
  } finally {
    await rm(bin, { recursive: true });
  }
});

test("login --json returns the link at once even when stdin is a terminal", async () => {
  const { login } = await import("../src/auth.ts");
  const { loadSession } = await import("../src/session.ts");

  vi.stubEnv("TELEMETRY_DEV_CONFIG_DIR", home);
  const isTTY = Object.getOwnPropertyDescriptor(process.stdin, "isTTY");
  Object.defineProperty(process.stdin, "isTTY", { value: true, configurable: true });

  try {
    const result = await login({
      session: await loadSession(fixture.url),
      browser: false,
      defer: true,
      onAuthorizationUrl: () => {},
      onWaiting: () => {
        throw new Error("login waited for a person");
      },
    });

    expect(result).toEqual({ status: "pending" });
    expect((await loadSession(fixture.url)).pending).toBeDefined();
  } finally {
    if (isTTY) Object.defineProperty(process.stdin, "isTTY", isTTY);
    else delete (process.stdin as { isTTY?: boolean }).isTTY;
    vi.unstubAllEnvs();
  }
});

test("a sign-in whose project list fails still signs in, without a default project", async () => {
  serverState.failProjects = true;

  for (const format of [[], ["--json"]]) {
    const started = await run("login", "--json");
    const approved = await fetch(JSON.parse(started.stdout).authorization_url, {
      redirect: "manual",
    });
    const finished = await run("login", "--finish", approved.headers.get("location")!, ...format);

    expect(finished.code).toBe(0);
    if (format.length) expect(JSON.parse(finished.stdout)).toMatchObject({ project: null });
    else expect(finished.stderr).toContain("Couldn't load your projects");
  }

  expect(JSON.parse((await run("status", "--json")).stdout)).toMatchObject({
    signed_in: true,
    project: null,
  });
});

test("a sign-in that can't reach the server says so, like any other command", async () => {
  const result = await runWith({ TELEMETRY_DEV_URL: await closedServer() }, "login", "--json");
  expect(result).toMatchObject({ code: 1, stdout: "" });
  expect(JSON.parse(result.stderr).error).toMatchObject({ code: "unreachable", retryable: true });
});

test("an expired access token is refreshed and the new one saved", async () => {
  await writeFile(
    sessionPath(),
    JSON.stringify({
      server: fixture.url,
      client: { client_id: "fixture-client" },
      tokens: { access_token: "expired", token_type: "Bearer", refresh_token: REFRESH_TOKEN },
    }),
    { mode: 0o600 },
  );

  const result = await run("projects", "--json");
  expect(result.code).toBe(0);
  expect(JSON.parse(result.stdout).projects).toHaveLength(2);
  expect(calls.some((call) => call.tool === "oauth:refresh_token")).toBe(true);
  expect(await savedSession()).toMatchObject({
    tokens: { access_token: TOKEN, refresh_token: REFRESH_TOKEN },
  });
});

test("a refresh that loses a race with another command uses that command's sign-in and keeps its changes", async () => {
  await writeFile(
    sessionPath(),
    JSON.stringify({
      server: fixture.url,
      client: { client_id: "fixture-client" },
      tokens: { access_token: "expired", token_type: "Bearer", refresh_token: "rotated-away" },
    }),
    { mode: 0o600 },
  );

  // Another command refreshes first and sets a default project, so the server has
  // already rotated this command's refresh token when it arrives.
  beforeRefresh.run = async (refreshToken) => {
    if (refreshToken !== "rotated-away") return;

    await writeFile(
      sessionPath(),
      JSON.stringify({
        server: fixture.url,
        client: { client_id: "fixture-client" },
        tokens: { access_token: "expired", token_type: "Bearer", refresh_token: REFRESH_TOKEN },
        project: { id: PROJECTS[0]!.id, name: "support-bot" },
      }),
    );
  };

  const result = await run("projects", "--json");
  expect(result.code).toBe(0);
  expect(await savedSession()).toMatchObject({
    tokens: { access_token: TOKEN, refresh_token: REFRESH_TOKEN },
    project: { id: PROJECTS[0]!.id },
  });
});

test("when the server refuses one command's refresh before another's is saved, both keep the new sign-in", async () => {
  await writeFile(
    sessionPath(),
    JSON.stringify({
      server: fixture.url,
      client: { client_id: "fixture-client" },
      tokens: { access_token: "expired", token_type: "Bearer", refresh_token: "rotating" },
    }),
    { mode: 0o600 },
  );

  // The server rotates "rotating" for the first command but answers slowly, and refuses
  // the second command's refresh with the same token at once.
  let first = true;
  beforeRefresh.run = async (refreshToken) => {
    if (refreshToken !== "rotating") return undefined;

    if (!first) return "refuse";
    first = false;
    await new Promise((resolve) => setTimeout(resolve, 1500));

    return "accept";
  };

  const winner = run("projects", "--json");
  await new Promise((resolve) => setTimeout(resolve, 500));
  const loser = run("projects", "--json");

  expect((await winner).code).toBe(0);
  expect((await loser).code).toBe(0);
  expect((await savedSession()).tokens).toMatchObject({
    access_token: TOKEN,
    refresh_token: REFRESH_TOKEN,
  });
  expect((await run("projects", "--json")).code).toBe(0);
});

test("logout revokes the refresh token and keeps the client registration and other servers' sign-ins", async () => {
  const production = JSON.stringify({
    server: "https://telemetry.dev",
    tokens: { access_token: "x" },
  });
  await writeFile(join(home, "session.json"), production);
  await writeFile(
    sessionPath(),
    JSON.stringify({
      server: fixture.url,
      client: { client_id: "fixture-client" },
      tokens: { access_token: TOKEN, token_type: "Bearer", refresh_token: REFRESH_TOKEN },
    }),
    { mode: 0o600 },
  );

  const result = await run("logout");
  expect(result.code).toBe(0);
  expect(calls.find((call) => call.tool === "oauth:revoke")?.args).toMatchObject({
    token: REFRESH_TOKEN,
  });

  const session = await savedSession();
  expect(session.client).toEqual({ client_id: "fixture-client" });
  expect(session.tokens).toBeUndefined();
  expect(await readFile(join(home, "session.json"), "utf8")).toBe(production);
});

const lockPath = () => `${sessionPath()}.lock`;
const markerFor = (path: string, token: string) =>
  `${path}.${createHash("sha256").update(token).digest("hex").slice(0, 16)}`;
const holdLock = async (path: string, token: string, pid: number, host = hostname()) =>
  writeFile(path, JSON.stringify({ token, pid, host }));
const backdate = async (path: string) => {
  const minuteAgo = new Date(Date.now() - 60_000);
  await utimes(path, minuteAgo, minuteAgo);
};
const exitedPid = () => spawnSync(process.execPath, ["-e", ""]).pid;

const record = (write: number) =>
  updateSession(
    fixture.url,
    (saved) =>
      ({
        ...saved,
        writes: [...((saved as { writes?: number[] }).writes ?? []), write],
      }) as Session,
  );
const recorded = async () =>
  ((await savedSession().catch(() => ({}))) as { writes?: number[] }).writes?.toSorted(
    (a, b) => a - b,
  ) ?? [];

// updateSession reads the config directory from the environment on each call.
const inConfigDir = async (run: () => Promise<void>) => {
  const configured = process.env.TELEMETRY_DEV_CONFIG_DIR;
  process.env.TELEMETRY_DEV_CONFIG_DIR = home;

  try {
    await run();
  } finally {
    if (configured === undefined) delete process.env.TELEMETRY_DEV_CONFIG_DIR;
    else process.env.TELEMETRY_DEV_CONFIG_DIR = configured;
  }
};

test("the sign-in lock waits for a running holder however long it holds, and for a command already removing an abandoned lock", async () => {
  const running = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"]);

  try {
    await inConfigDir(async () => {
      await holdLock(lockPath(), "running", running.pid!);
      await backdate(lockPath());

      const waiting = Promise.all([0, 1, 2, 3].map(record));
      await sleep(500);
      expect(await recorded()).toEqual([]);

      await rm(lockPath());
      await waiting;
      expect(await recorded()).toEqual([0, 1, 2, 3]);

      // Another machine's lock is judged by its age, and this one is fresh.
      await holdLock(lockPath(), "other-machine", process.ppid, "another-machine");
      const behindOtherMachine = record(5);
      await sleep(500);
      expect(await recorded()).toEqual([0, 1, 2, 3]);

      await rm(lockPath());
      await behindOtherMachine;
      expect(await recorded()).toEqual([0, 1, 2, 3, 5]);

      await holdLock(lockPath(), "abandoned", exitedPid());
      const marker = markerFor(lockPath(), "abandoned");
      await holdLock(marker, "breaking", running.pid!);

      const behindBreaker = record(4);
      await sleep(500);
      expect(await recorded()).toEqual([0, 1, 2, 3, 5]);

      await rm(marker);
      await behindBreaker;
      expect(await recorded()).toEqual([0, 1, 2, 3, 4, 5]);
    });
  } finally {
    running.kill();
  }
});

test("the sign-in lock is taken over at once, without lost writes, when its holder or a command breaking it has exited", async () => {
  await inConfigDir(async () => {
    const takeOver = async (writes: number[]) => {
      const started = Date.now();
      await Promise.all(writes.map(record));
      expect(Date.now() - started).toBeLessThan(2000);
    };

    await holdLock(lockPath(), "exited", exitedPid());
    await takeOver([0, 1, 2, 3, 4, 5, 6, 7]);

    await holdLock(lockPath(), "exited-again", exitedPid());
    await holdLock(markerFor(lockPath(), "exited-again"), "exited-breaker", exitedPid());
    await takeOver([8]);

    // Left by an earlier process that had this one's pid.
    await holdLock(lockPath(), "same-pid", process.pid);
    await takeOver([9]);

    // Another machine's lock can only be judged by its age.
    await holdLock(lockPath(), "other-machine", process.ppid, "another-machine");
    await backdate(lockPath());
    await takeOver([10]);

    expect(await recorded()).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(await readdir(home)).toEqual([sessionFile(fixture.url)]);
  });
});
