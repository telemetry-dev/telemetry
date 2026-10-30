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
  | "unreachable"
  | "timeout"
  | "tool_error"
  | "result_too_large"
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

export const notSignedIn = () =>
  new CliError("You're not signed in. Run `telemetry login` first.", {
    exitCode: EXIT.auth,
    code: "not_signed_in",
    hint: "telemetry login",
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

export const translateError = (error: unknown, authErrors: AuthErrors): never => {
  if (error instanceof CliError) throw error;

  if (authErrors.some((type) => error instanceof type)) {
    throw new CliError(
      "Your sign-in has expired or was disconnected. Run `telemetry login` again.",
      { exitCode: EXIT.auth, code: "sign_in_expired", hint: "telemetry login" },
    );
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

  if (network) {
    throw new CliError(
      `Couldn't reach telemetry.dev (${network}). Check your connection and try again.`,
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
    throw new CliError("telemetry.dev took too long to answer. Try again shortly.", {
      code: "timeout",
      retryable: true,
    });
  }

  throw new CliError(`telemetry.dev request failed: ${message}`, { code: "request_failed" });
};
