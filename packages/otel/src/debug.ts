import type { LogLevel, SdkLogLevel } from "./config.ts";

const ORDER = { debug: 0, info: 1, warn: 2, error: 3, silent: 4 } satisfies Record<
  SdkLogLevel,
  number
>;

let currentLevel: SdkLogLevel = "warn";

export function setLogLevel(level: SdkLogLevel): void {
  currentLevel = level;
}

function emit(level: LogLevel, args: unknown[]): void {
  if (ORDER[level] < ORDER[currentLevel]) return;
  // eslint-disable-next-line no-console
  console[level]("[telemetry.dev]", ...args);
}

export const diag = {
  debug: (...args: unknown[]) => emit("debug", args),
  info: (...args: unknown[]) => emit("info", args),
  warn: (...args: unknown[]) => emit("warn", args),
  error: (...args: unknown[]) => emit("error", args),
};

/** Fail-open guard: SDK internals report through onError + diagnostics, never into user code. */
export function reportError(onError: ((error: Error) => void) | undefined, cause: unknown): void {
  try {
    onError?.(cause instanceof Error ? cause : new Error(String(cause)));
  } catch {
    // onError itself must never propagate
  }

  diag.error(cause);
}
