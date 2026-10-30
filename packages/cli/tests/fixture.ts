import { createHash, randomBytes } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";

import { createMcpHandler, McpServer } from "@modelcontextprotocol/server";
import { z } from "zod";

export const TOKEN = "fixture-access-token";

export const PROJECTS = [
  {
    id: "7d1c3f0e-2b4a-4c8e-9f61-0a5b2c3d4e5f",
    name: "support-bot",
    retention_days: 30,
    created_at: "2026-08-14T09:12:00.000Z",
  },
  {
    id: "c2a9e4b1-6f3d-4a7e-8b20-9d1e5f6a7b8c",
    // Any workspace member can name a project; this one writes the clipboard (OSC 52).
    name: "eval-runner\u001b]52;c;aGk=\u0007",
    retention_days: 90,
    created_at: "2026-09-02T13:09:32.549Z",
  },
];

const minutesAgo = (minutes: number) => new Date(Date.now() - minutes * 60_000).toISOString();

export const TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736";
export const QUIET_TRACE_ID = "9a0b7c6d5e4f30211a2b3c4d5e6f7081";
export const PARALLEL_TRACE_ID = "3e8a1c5b7d9f40e2a6b4c8d0e2f4a6b8";

export const traces = () => ({
  traces: [
    // The newest trace succeeded, so a suggestion to inspect a failure has to look past it.
    {
      id: QUIET_TRACE_ID,
      name: "support.answer",
      status: "success",
      error_type: null,
      started_at: minutesAgo(1),
      duration_ms: 1870,
      environment: "production",
      service_name: "support-api",
      session_id: "conv_8f2a91",
      user_id: "user_4821",
      models: ["gpt-5"],
      span_count: 3,
      error_span_count: 0,
      input_tokens: 1934,
      output_tokens: 211,
      cost_usd: 0.00718,
    },
    {
      id: TRACE_ID,
      name: "support.answer",
      status: "error",
      error_type: "RateLimitError",
      started_at: minutesAgo(2),
      duration_ms: 4210,
      environment: "production",
      service_name: "support-api",
      session_id: "conv_8f2a91",
      user_id: "user_4821",
      models: ["gpt-5", "gpt-5-mini"],
      span_count: 7,
      error_span_count: 1,
      input_tokens: 2841,
      output_tokens: 373,
      cost_usd: 0.01234,
    },
    {
      id: "1f2e3d4c5b6a79880a9b8c7d6e5f4a3b",
      // Captured text can carry terminal escapes; the CLI must never pass them through.
      name: "triage \u001b]52;c;ZXZpbA==\u0007classify\u001b[2J ticket",
      status: "success",
      error_type: null,
      started_at: minutesAgo(64),
      duration_ms: 640,
      environment: "production",
      service_name: "triage-worker",
      session_id: null,
      user_id: null,
      models: ["claude-haiku-5-5"],
      span_count: 2,
      error_span_count: 0,
      input_tokens: 812,
      output_tokens: 44,
      cost_usd: 0.00051,
    },
  ],
  next_cursor: "eyJzdGFydGVkQXQiOiIyMDI2LTEwLTI4In0",
});

const span = (
  id: string,
  parent: string | null,
  name: string,
  offsetMs: number,
  durationMs: number,
  extra: Record<string, unknown> = {},
) => ({
  id,
  parent_span_id: parent,
  name,
  operation: null,
  kind: "internal",
  agent_name: null,
  tool_name: null,
  provider: null,
  model: null,
  status: "ok",
  error_type: null,
  error: null,
  finish_reason: null,
  started_at: new Date(Date.parse(minutesAgo(2)) + offsetMs).toISOString(),
  duration_ms: durationMs,
  input_tokens: null,
  output_tokens: null,
  cost_usd: null,
  ...extra,
});

export const traceDetail = () => ({
  trace: {
    id: TRACE_ID,
    name: "support.answer",
    status: "error",
    error_type: "RateLimitError",
    started_at: minutesAgo(2),
    duration_ms: 4210,
    environment: "production",
    service_name: "support-api",
    framework: "ai-sdk",
    session_id: "conv_8f2a91",
    user_id: "user_4821",
    input_tokens: 2841,
    output_tokens: 373,
    cost_usd: 0.01234,
  },
  span_count: 7,
  log_count: 2,
  spans: [
    span("a1", null, "invoke_agent support-bot", 0, 4210, {
      agent_name: "support-bot",
      operation: "invoke_agent",
    }),
    span("b2", "a1", "chat gpt-5-mini", 40, 610, {
      model: "gpt-5-mini",
      provider: "openai",
      input_tokens: 912,
      output_tokens: 38,
      cost_usd: 0.0011,
    }),
    span("c3", "a1", "execute_tool search_docs", 680, 820, { tool_name: "search_docs" }),
    span("d4", "c3", "GET /v1/search", 720, 690),
    span("e5", "a1", "chat gpt-5", 1540, 1180, {
      model: "gpt-5",
      provider: "openai",
      input_tokens: 1929,
      output_tokens: 335,
      cost_usd: 0.0112,
    }),
    span("f6", "a1", "chat gpt-5", 2760, 1420, {
      model: "gpt-5",
      provider: "openai",
      status: "error",
      error_type: "RateLimitError",
      error:
        "429 Too Many Requests: Rate limit reached for gpt-5 in organization org-acme on tokens per min (TPM).",
    }),
    // A parent outside this page still shows, as a root.
    span("g7", "zz-missing", "flush_telemetry", 4180, 25),
  ],
  logs: [
    {
      timestamp: minutesAgo(2),
      level: "warn",
      span_id: "e5",
      message: "Retrying chat completion after 429 (attempt 2 of 3)",
    },
    {
      timestamp: minutesAgo(2),
      level: "error",
      span_id: "f6",
      message: "Giving up: rate limit persisted after 3 attempts",
    },
  ],
  next_cursor: null,
});

/** A run that succeeded with each operation once: nothing to summarize above the tree. */
const quietTraceDetail = () => ({
  trace: {
    id: QUIET_TRACE_ID,
    name: "support.answer",
    status: "success",
    error_type: null,
    started_at: minutesAgo(1),
    duration_ms: 1870,
    environment: "production",
    service_name: "support-api",
    framework: "ai-sdk",
    session_id: "conv_8f2a91",
    user_id: "user_4821",
    input_tokens: 1934,
    output_tokens: 211,
    cost_usd: 0.00718,
  },
  span_count: 3,
  log_count: 0,
  spans: [
    span("q1", null, "invoke_agent support-bot", 0, 1870, {
      agent_name: "support-bot",
      operation: "invoke_agent",
    }),
    span("q2", "q1", "execute_tool lookup_order", 30, 240, { tool_name: "lookup_order" }),
    span("q3", "q1", "chat gpt-5", 290, 1560, {
      model: "gpt-5",
      provider: "openai",
      input_tokens: 1934,
      output_tokens: 211,
      cost_usd: 0.00718,
    }),
  ],
  logs: [],
  next_cursor: null,
});

/**
 * An agent that runs three 1 s searches at once, so they cover 1.1 s of a 3 s trace, then
 * fails two levels down, in the request a tool made. The failure is marked on the tool
 * and the agent too, as instrumentation does when an exception passes through them.
 */
const parallelTraceDetail = () => ({
  trace: {
    id: PARALLEL_TRACE_ID,
    name: "research.report",
    status: "error",
    error_type: "TimeoutError",
    started_at: minutesAgo(2),
    duration_ms: 3000,
    environment: "production",
    service_name: "research-api",
    framework: "ai-sdk",
    session_id: null,
    user_id: null,
    input_tokens: null,
    output_tokens: null,
    cost_usd: null,
  },
  span_count: 6,
  log_count: 0,
  spans: [
    span("r1", null, "invoke_agent researcher", 0, 3000, {
      agent_name: "researcher",
      operation: "invoke_agent",
      status: "error",
      error_type: "TimeoutError",
      error: "TimeoutError: tool fetch_page failed",
    }),
    span("s1", "r1", "execute_tool web_search", 100, 1000, { tool_name: "web_search" }),
    span("s2", "r1", "execute_tool web_search", 150, 1000, { tool_name: "web_search" }),
    span("s3", "r1", "execute_tool web_search", 200, 1000, { tool_name: "web_search" }),
    span("t1", "r1", "execute_tool fetch_page", 1300, 800, {
      tool_name: "fetch_page",
      status: "error",
      error_type: "TimeoutError",
      error: "TimeoutError: Request timed out after 700 ms",
    }),
    span("t2", "t1", "GET /page", 1350, 700, {
      status: "error",
      error_type: "TimeoutError",
      error: "Request timed out after 700 ms\n    at fetchPage (tools.ts:41:11)",
    }),
  ],
  logs: [],
  next_cursor: null,
});

const TRACE_DETAILS: Record<string, () => unknown> = {
  [TRACE_ID]: traceDetail,
  [QUIET_TRACE_ID]: quietTraceDetail,
  [PARALLEL_TRACE_ID]: parallelTraceDetail,
};

export const sessions = () => ({
  sessions: [
    {
      // Session ids come from the user's app and can hold spaces and quotes.
      id: "Ada's thread 42",
      last_active_at: minutesAgo(1),
      user_id: "user_4821",
      agent_name: "support-bot",
      models: ["gpt-5"],
      trace_count: 2,
      error_trace_count: 0,
      input_tokens: 3120,
      output_tokens: 402,
      cost_usd: 0.0114,
    },
    {
      id: "conv_8f2a91",
      last_active_at: minutesAgo(2),
      user_id: "user_4821",
      agent_name: "support-bot",
      models: ["gpt-5", "gpt-5-mini"],
      trace_count: 6,
      error_trace_count: 1,
      input_tokens: 11240,
      output_tokens: 1388,
      cost_usd: 0.0482,
    },
    {
      id: "conv_77c0de",
      last_active_at: minutesAgo(41),
      user_id: "user_0193",
      agent_name: "support-bot",
      models: ["gpt-5"],
      trace_count: 3,
      error_trace_count: 0,
      input_tokens: 4410,
      output_tokens: 602,
      cost_usd: 0.0165,
    },
  ],
  next_cursor: null,
});

export const logs = () => ({
  range_days: 7,
  counts: {
    debugCount: 1204,
    infoCount: 8810,
    warnCount: 96,
    errorCount: 18,
    totalCount: 10128,
    traceCount: 4410,
  },
  has_more: true,
  logs: [
    {
      timestamp: minutesAgo(2),
      level: "error",
      message: "Giving up: rate limit persisted after 3 attempts",
      service_name: "support-api",
      environment: "production",
      trace_id: TRACE_ID,
      span_id: "f6",
    },
    {
      timestamp: minutesAgo(2),
      level: "warn",
      message: "Retrying chat completion after 429 (attempt 2 of 3)",
      service_name: "support-api",
      environment: "production",
      trace_id: TRACE_ID,
      span_id: "e5",
    },
    {
      timestamp: minutesAgo(18),
      level: "info",
      message: "Indexed 42 new help-center articles",
      service_name: "triage-worker",
      environment: "production",
      trace_id: null,
      span_id: null,
    },
  ],
});

const totals = (scale: number) => ({
  totalRequests: Math.round(12480 * scale),
  errorCount: Math.round(132 * scale * 1.3),
  errorRate: 1.06 * 1.3,
  p95DurationMs: 2400 * scale,
  avgDurationMs: 910,
  totalInputTokens: Math.round(7_100_000 * scale),
  totalOutputTokens: Math.round(1_100_000 * scale),
  totalCostUsd: 41.2 * scale,
  activeUsers: Math.round(312 * scale),
});

export const usage = () => ({
  range_days: 7,
  since: minutesAgo(7 * 24 * 60),
  data_incomplete: false,
  current: { ...totals(1), errorCount: 132, errorRate: 1.06 },
  previous: totals(0.86),
  group_by: "model",
  breakdown: [
    {
      category: "gpt-5",
      errorCount: 98,
      errorRate: 1.2,
      p95DurationMs: 2810,
      totalCostUsd: 33.9,
      totalTokens: 5_900_000,
      totalRequests: 8120,
    },
    {
      category: "gpt-5-mini",
      errorCount: 21,
      errorRate: 0.6,
      p95DurationMs: 940,
      totalCostUsd: 4.1,
      totalTokens: 1_700_000,
      totalRequests: 3410,
    },
    {
      category: "claude-haiku-5-5",
      errorCount: 0,
      errorRate: 0,
      p95DurationMs: 610,
      totalCostUsd: 3.2,
      totalTokens: 600_000,
      totalRequests: 950,
    },
  ],
  breakdown_limit: 6,
});

export const PAYLOADS: Record<string, string> = {
  input: JSON.stringify([
    { role: "system", content: "You are the Acme support assistant." },
    { role: "user", content: "My invoice shows two charges for October. Can you refund one?" },
  ]),
  output:
    "I found both charges. I've refunded the duplicate; it will reach your card in 5-10 days.",
};

const loose = z.looseObject({});

const result = (value: unknown) => ({
  content: [{ type: "text" as const, text: JSON.stringify(value) }],
});

const server = () => {
  const mcp = new McpServer({ name: "telemetry.dev fixture", version: "1.0.0" });
  const tool = (name: string, handler: (args: Record<string, unknown>) => unknown) =>
    mcp.registerTool(name, { inputSchema: loose }, (args: Record<string, unknown>) => {
      calls.push({ tool: name, args });
      const value = handler(args);

      return typeof value === "string"
        ? { isError: true, content: [{ type: "text" as const, text: value }] }
        : result(value);
    });

  tool("list_projects", () =>
    failProjects.value
      ? "telemetry.dev could not complete this request. Try again shortly."
      : { organization_id: "org_fixture", projects: PROJECTS },
  );
  tool("search_traces", () => traces());
  tool("search_sessions", () => sessions());
  tool("get_trace", (args) =>
    Object.hasOwn(TRACE_DETAILS, String(args.trace_id))
      ? TRACE_DETAILS[String(args.trace_id)]!()
      : "Trace not found.",
  );
  tool("list_logs", () => logs());
  tool("get_usage_summary", () => usage());
  tool("get_span_payload", (args) => {
    const text = PAYLOADS[String(args.field)] ?? "";
    const offset = Number(args.offset ?? 0);
    // Small pages, so the CLI has to follow next_offset.
    const end = Math.min(text.length, offset + 40);

    return {
      field: args.field,
      total_characters: text.length,
      stored_text_truncated: false,
      offset,
      next_offset: end < text.length ? end : null,
      text: text.slice(offset, end),
    };
  });

  return mcp;
};

export const calls: { tool: string; args: Record<string, unknown> }[] = [];

/** Makes list_projects answer with a tool error, as a server failure would. */
export const failProjects = { value: false };

/**
 * Runs before the token endpoint answers a refresh, to act as another CLI process or a
 * server that rotates tokens. "accept" or "refuse" overrides the usual check.
 */
export const beforeRefresh: {
  run?: (refreshToken: string) => Promise<"accept" | "refuse" | undefined>;
} = {};

const handler = createMcpHandler(() => server(), { legacy: "stateless", responseMode: "json" });

export const REFRESH_TOKEN = "fixture-refresh-token";
export const EMAIL = "ada@acme.dev";

const unsignedJwt = (claims: object) =>
  `e30.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.`;

/** Codes this authorization server issued, with the PKCE challenge each must answer. */
const codes = new Map<string, { challenge: string; redirectUri: string }>();

const json = (response: ServerResponse, status: number, value: unknown, headers = {}) =>
  response
    .writeHead(status, { "Content-Type": "application/json", ...headers })
    .end(JSON.stringify(value));

const readBody = async (request: IncomingMessage) => {
  const chunks: Buffer[] = [];

  for await (const chunk of request) chunks.push(chunk as Buffer);

  return Buffer.concat(chunks);
};

/**
 * A stand-in for telemetry.dev: the MCP server, which accepts one bearer token, and a
 * minimal OAuth 2.1 authorization server with registration, PKCE, refresh, and revocation.
 * GET /authorize approves at once, the way a person selecting Allow would.
 */
export const startFixture = async (port = 0) => {
  let origin = "";

  const http = createServer(async (request, response) => {
    const url = new URL(request.url ?? "/", origin);

    if (url.pathname === "/.well-known/oauth-protected-resource/mcp") {
      return json(response, 200, { resource: `${origin}/mcp`, authorization_servers: [origin] });
    }

    if (url.pathname === "/.well-known/oauth-authorization-server") {
      return json(response, 200, {
        issuer: origin,
        authorization_endpoint: `${origin}/authorize`,
        token_endpoint: `${origin}/token`,
        registration_endpoint: `${origin}/register`,
        revocation_endpoint: `${origin}/revoke`,
        response_types_supported: ["code"],
        grant_types_supported: ["authorization_code", "refresh_token"],
        code_challenge_methods_supported: ["S256"],
        token_endpoint_auth_methods_supported: ["none"],
        userinfo_endpoint: `${origin}/userinfo`,
        authorization_response_iss_parameter_supported: true,
      });
    }

    if (url.pathname === "/register") {
      const metadata = JSON.parse((await readBody(request)).toString()) as object;
      calls.push({ tool: "oauth:register", args: metadata as Record<string, unknown> });

      return json(response, 201, {
        ...metadata,
        client_id: "fixture-client",
        client_id_issued_at: 0,
      });
    }

    if (url.pathname === "/authorize") {
      const code = randomBytes(8).toString("hex");
      const redirectUri = url.searchParams.get("redirect_uri") ?? "";
      codes.set(code, { challenge: url.searchParams.get("code_challenge") ?? "", redirectUri });

      const back = new URL(redirectUri);
      back.searchParams.set("code", code);
      back.searchParams.set("state", url.searchParams.get("state") ?? "");
      back.searchParams.set("iss", origin);

      return response.writeHead(302, { Location: back.href }).end();
    }

    if (url.pathname === "/token") {
      const form = new URLSearchParams((await readBody(request)).toString());
      calls.push({ tool: `oauth:${form.get("grant_type")}`, args: Object.fromEntries(form) });

      const verdict =
        form.get("grant_type") === "refresh_token"
          ? await beforeRefresh.run?.(form.get("refresh_token") ?? "")
          : undefined;

      if (
        form.get("grant_type") === "refresh_token" &&
        (verdict === "refuse" ||
          (verdict !== "accept" && form.get("refresh_token") !== REFRESH_TOKEN))
      ) {
        return json(response, 400, { error: "invalid_grant" });
      }

      if (form.get("grant_type") === "authorization_code") {
        const issued = codes.get(form.get("code") ?? "");
        const answer = createHash("sha256")
          .update(form.get("code_verifier") ?? "")
          .digest("base64url");
        codes.delete(form.get("code") ?? "");

        if (
          !issued ||
          issued.challenge !== answer ||
          issued.redirectUri !== form.get("redirect_uri")
        ) {
          return json(response, 400, { error: "invalid_grant" });
        }
      }

      return json(response, 200, {
        access_token: TOKEN,
        token_type: "Bearer",
        expires_in: 3600,
        refresh_token: REFRESH_TOKEN,
        scope: "openid email offline_access telemetry:read telemetry:payloads",
        // Like telemetry.dev, the ID token names only the subject; the email is in userinfo.
        id_token: unsignedJwt({ sub: "user_1" }),
      });
    }

    if (url.pathname === "/userinfo") {
      return request.headers.authorization === `Bearer ${TOKEN}`
        ? json(response, 200, { sub: "user_1", email: EMAIL, email_verified: true })
        : json(response, 401, { error: "invalid_token" });
    }

    if (url.pathname === "/revoke") {
      calls.push({
        tool: "oauth:revoke",
        args: Object.fromEntries(new URLSearchParams((await readBody(request)).toString())),
      });

      return response.writeHead(200).end();
    }

    if (request.headers.authorization !== `Bearer ${TOKEN}`) {
      return json(
        response,
        401,
        { error: "invalid_token" },
        {
          "WWW-Authenticate": `Bearer resource_metadata="${origin}/.well-known/oauth-protected-resource/mcp"`,
        },
      );
    }

    const body = await readBody(request);

    const answer = await handler.fetch(
      new Request(`${origin}${request.url}`, {
        method: request.method,
        headers: request.headers as Record<string, string>,
        body: body.length > 0 ? body : null,
      }),
    );

    response.writeHead(answer.status, Object.fromEntries(answer.headers));
    response.end(Buffer.from(await answer.arrayBuffer()));
  });

  await new Promise<void>((resolve) => http.listen(port, "127.0.0.1", resolve));
  const address = http.address();
  origin = `http://127.0.0.1:${typeof address === "object" && address ? address.port : port}`;

  return {
    url: origin,
    close: () => new Promise<void>((resolve) => http.close(() => resolve())),
  };
};
