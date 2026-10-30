import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { type AddressInfo, createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterAll, beforeAll, beforeEach, expect, test, vi } from "vitest";

import { sessionFile } from "../src/session.ts";
import {
  beforeRefresh,
  calls,
  EMAIL,
  PAYLOADS,
  PROJECTS,
  REFRESH_TOKEN,
  startFixture,
  TOKEN,
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

const signIn = (project?: (typeof PROJECTS)[number]) =>
  writeFile(
    sessionPath(),
    JSON.stringify({
      server: fixture.url,
      client: { client_id: "fixture-client" },
      tokens: {
        access_token: TOKEN,
        token_type: "Bearer",
        scope: "openid email offline_access telemetry:read telemetry:payloads",
      },
      email: "ada@acme.dev",
      ...(project ? { project: { id: project.id, name: "support-bot" } } : {}),
    }),
    { mode: 0o600 },
  );

beforeEach(async () => {
  home = await mkdtemp(join(tmpdir(), "telemetry-cli-"));
  calls.length = 0;
  delete beforeRefresh.run;

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
    code: "tool_error",
    message: "Trace not found.",
  });

  // A port that was just closed refuses the connection.
  const closed = await new Promise<string>((resolve) => {
    const server = createServer().listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      server.close(() => resolve(`http://127.0.0.1:${port}`));
    });
  });
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
  expect(human.stderr).toBe("error Trace not found.\n");
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
