import { BIN } from "./invocation.ts";

/** Exit codes, documented in the README. */
export const EXIT = {
  ok: 0,
  failure: 1,
  usage: 2,
  auth: 4,
  interrupted: 130,
} as const;

/** What scripts and agents branch on. Listed in the README; only ever added to. */
export type ErrorCode =
  | "failed"
  | "internal"
  | "invalid_arguments"
  | "unknown_command"
  | "project_not_found"
  | "project_ambiguous"
  | "project_required"
  | "no_projects"
  | "not_signed_in"
  | "sign_in_expired"
  | "sign_in_failed"
  | "no_pending_sign_in"
  | "payloads_not_allowed"
  | "access_denied"
  | "invalid_cursor"
  | "trace_not_found"
  | "span_not_found"
  | "unreachable"
  | "timeout"
  | "server_error"
  | "tool_error"
  | "result_too_large"
  | "server_unsupported"
  | "request_failed";

type ErrorDetails = {
  readonly exitCode?: number;
  readonly code?: ErrorCode;
  /** A command to run next. `telemetry login` needs a person at a browser. */
  readonly hint?: string;
  /** Whether running the same command again may succeed. */
  readonly retryable?: boolean;
};

/** A failure with a message written for the person running the command. */
export class CliError extends Error {
  readonly exitCode: number;
  readonly code: ErrorCode;
  readonly hint: string | undefined;
  readonly retryable: boolean;

  constructor(message: string, details: number | ErrorDetails = {}) {
    super(message);
    const {
      exitCode = EXIT.failure,
      code,
      hint,
      retryable = false,
    } = typeof details === "number" ? { exitCode: details } : details;
    this.exitCode = exitCode;
    this.code = code ?? (exitCode === EXIT.usage ? "invalid_arguments" : "failed");
    this.hint = hint;
    this.retryable = retryable;
  }
}

export const usageError = (message: string, details: Omit<ErrorDetails, "exitCode"> = {}) =>
  new CliError(message, { ...details, exitCode: EXIT.usage });

export const payloadsNotAllowed = () =>
  new CliError(
    `This sign-in can't read captured prompts and outputs. Run ${BIN} login, check "Also read captured prompts, outputs, and tool arguments", and select Allow.`,
    { exitCode: EXIT.auth, code: "payloads_not_allowed", hint: `${BIN} login` },
  );

/**
 * The CliError for a tool error, from the code telemetry.dev puts in the result's
 * `_meta`. A server without codes, or a code this CLI doesn't know, gives `tool_error`.
 * The result's text is written for agents, so codes a person acts on get the CLI's own.
 */
export const toolError = (text: string, meta: unknown): CliError => {
  const error = (meta as Record<string, unknown> | undefined)?.["dev.telemetry/error"] as
    | { code?: unknown; retryable?: unknown }
    | undefined;
  const retryable = error?.retryable === true;

  switch (error?.code) {
    case "access_denied":
      return new CliError(
        `Your account no longer has access to this workspace. Run ${BIN} login and choose a workspace you belong to.`,
        { exitCode: EXIT.auth, code: "access_denied", hint: `${BIN} login` },
      );
    case "payloads_not_allowed":
      return payloadsNotAllowed();
    case "project_not_found":
      return usageError(
        `That project isn't in this workspace; it may have been deleted. Run ${BIN} projects to list the projects, and ${BIN} use <name> to change the default.`,
        { code: "project_not_found", hint: `${BIN} projects` },
      );
    case "invalid_cursor":
      return usageError(
        "That cursor isn't from an earlier page of this command with these options. Run it again without --cursor.",
        { code: "invalid_cursor" },
      );
    // The server checks what the CLI can't: a window that starts after it ends once a
    // duration is resolved, or one that ends before the project's retention.
    case "invalid_arguments":
      return usageError(text || "telemetry.dev didn't accept these options.", {
        code: "invalid_arguments",
      });
    case "trace_not_found":
    case "span_not_found":
      return new CliError(text, { code: error.code });
    case "server_error":
      return new CliError(text, { code: "server_error", retryable });
    default:
      return new CliError(text || "telemetry.dev could not complete this request.", {
        code: "tool_error",
        retryable,
      });
  }
};

export const notSignedIn = () =>
  new CliError(`You're not signed in. Run ${BIN} login first.`, {
    exitCode: EXIT.auth,
    code: "not_signed_in",
    hint: `${BIN} login`,
  });

// Each is a refresh or authorization the server refused for good.
type AuthErrors = readonly (abstract new (...args: never[]) => Error)[];

const NETWORK_CODES = new Set([
  "ECONNREFUSED",
  "ECONNRESET",
  "EAI_AGAIN",
  "EHOSTUNREACH",
  "ENETUNREACH",
  "ENOTFOUND",
  "ETIMEDOUT",
  "UND_ERR_SOCKET",
]);

// undici's own timeouts, raised by fetch before any of the CLI's timers fire.
const TIMEOUT_CODES = new Set([
  "UND_ERR_CONNECT_TIMEOUT",
  "UND_ERR_HEADERS_TIMEOUT",
  "UND_ERR_BODY_TIMEOUT",
]);

const PROXY_VARIABLES = ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"];

/**
 * Node's fetch ignores proxy variables unless NODE_USE_ENV_PROXY=1 or --use-env-proxy
 * was set at startup, which Node.js 22.21 and 24.0 added (Node.js docs, "Enterprise
 * Network Configuration"), so behind a proxy the CLI connects directly and fails.
 */
const proxyNote = () => {
  if (!PROXY_VARIABLES.some((name) => process.env[name])) return "";

  const [major = 0, minor = 0] = process.versions.node.split(".").map(Number);
  const supported = major >= 24 || (major === 22 && minor >= 21);
  const enabled =
    process.env.NODE_USE_ENV_PROXY === "1" ||
    process.execArgv.includes("--use-env-proxy") ||
    (process.env.NODE_OPTIONS ?? "").split(/\s+/).includes("--use-env-proxy");

  if (!supported) {
    return ` A proxy is set, but Node.js ${process.versions.node} ignores it. Node.js 22.21+ or 24+ uses it with NODE_USE_ENV_PROXY=1 set.`;
  }

  return enabled ? "" : " A proxy is set, but Node.js uses it only with NODE_USE_ENV_PROXY=1 set.";
};

/**
 * A failed request to `server` as a CliError: a refused sign-in, an unreachable or slow
 * server, or anything else, named by the server's host.
 */
export const translateError = (error: unknown, authErrors: AuthErrors, server: string): never => {
  if (error instanceof CliError) throw error;

  if (authErrors.some((type) => error instanceof type)) {
    throw new CliError(`Your sign-in has expired or was disconnected. Run ${BIN} login again.`, {
      exitCode: EXIT.auth,
      code: "sign_in_expired",
      hint: `${BIN} login`,
    });
  }

  // The network error sits at the bottom of a chain: the SDK wraps fetch's
  // "fetch failed", which wraps the socket error that has the code.
  const chain: unknown[] = [error];

  for (let link = error; link instanceof Error && link.cause !== undefined && chain.length < 8; ) {
    link = link.cause;
    chain.push(link);
  }

  const codes = chain
    .map((link) => (link as { code?: unknown } | null)?.code)
    .filter((code): code is string => typeof code === "string");
  const network = codes.find((code) => NETWORK_CODES.has(code));
  // Node's AggregateError for a failed connection has an empty message.
  const message =
    chain.findLast((link): link is Error => link instanceof Error && link.message !== "")
      ?.message ?? String(error);

  const host = new URL(server).host;

  if (network) {
    throw new CliError(
      `Couldn't reach ${host} (${network}). Check your connection and try again.${proxyNote()}`,
      { code: "unreachable", retryable: true },
    );
  }

  if (
    codes.some((code) => code === "REQUEST_TIMEOUT" || TIMEOUT_CODES.has(code)) ||
    chain.some(
      (link) =>
        link instanceof Error && (link.name === "TimeoutError" || link.name === "AbortError"),
    )
  ) {
    throw new CliError(`${host} took too long to answer. Try again shortly.${proxyNote()}`, {
      code: "timeout",
      retryable: true,
    });
  }

  throw new CliError(`${host} request failed: ${message}`, { code: "request_failed" });
};
