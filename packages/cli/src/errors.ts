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
