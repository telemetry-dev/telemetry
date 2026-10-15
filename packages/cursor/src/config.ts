import { lstatSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { init, shutdown, type ClientOverrides, type TelemetryOptions } from "@telemetry-dev/sdk";

type JsonValue = string | number | boolean | null | undefined | JsonValue[] | JsonRecord;

interface JsonRecord {
  [key: string]: JsonValue;
}

/**
 * SDK options accepted by @telemetry-dev/cursor. `registerGlobal` is excluded:
 * the daemon owns its process, but every span is still created through the
 * SDK's own tracer with explicit parenting.
 */
export type TelemetryDevCursorOptions = Omit<TelemetryOptions, "registerGlobal" | "sdkName"> & {
  /** Override for ~/.cursor/chats when resolving chat titles (tests). */
  chatsDir?: string;
};

export type { ClientOverrides };

/** Config file written by `telemetry-dev-cursor install`; env vars win over it. */
export function configPath(): string {
  return join(homedir(), ".cursor", "telemetry-dev.json");
}

/**
 * Reads ~/.cursor/telemetry-dev.json. Cursor launched from the macOS dock does
 * not inherit shell env vars, so the api key usually has to live in this file.
 * Keys with a set env var are omitted: the SDK falls back to env for absent
 * options, which keeps the documented env-over-file precedence. Never throws:
 * a file that exists but cannot be read as a JSON object disables telemetry,
 * because it may be the file that turned content capture off.
 */
export function fileConfig(): TelemetryDevCursorOptions {
  let record: JsonRecord | undefined;

  try {
    record = readConfigFile();
  } catch {
    return { enabled: false };
  }

  if (!record) return {};

  const pick = (key: string, envVar: string): string | undefined =>
    process.env[envVar] === undefined ? readString(record[key]) : undefined;

  // Any value other than `true` counts as false, so a mistyped "false" turns
  // content capture off instead of silently leaving it on.
  const bool = (key: string): boolean | undefined =>
    record[key] === undefined ? undefined : record[key] === true;

  return {
    apiKey: pick("apiKey", "TELEMETRY_DEV_API_KEY"),
    baseUrl: pick("baseUrl", "TELEMETRY_DEV_BASE_URL"),
    environment: pick("environment", "TELEMETRY_DEV_ENVIRONMENT"),
    serviceName: pick("serviceName", "OTEL_SERVICE_NAME"),
    enabled: bool("enabled"),
    captureInput: bool("captureInput"),
    captureOutput: bool("captureOutput"),
  };
}

/**
 * Returns the parsed config file, or undefined when it does not exist. Throws
 * when it cannot be read or is not a JSON object; a symlink whose target is
 * gone counts as unreadable, not absent.
 */
export function readConfigFile(): JsonRecord | undefined {
  const path = configPath();
  let text: string;

  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    const missing = error instanceof Error && "code" in error && error.code === "ENOENT";

    if (missing && !lstatSync(path, { throwIfNoEntry: false })) return undefined;
    throw error;
  }

  const record = asRecord(JSON.parse(text));

  if (!record || Array.isArray(record)) throw new Error("not a JSON object");

  return record;
}

let initialized = false;

/** Initializes the telemetry.dev SDK exactly once per process for this integration. */
export function ensureInit(
  options: TelemetryDevCursorOptions = {},
  overrides?: ClientOverrides,
): void {
  if (initialized) return;
  initialized = true;
  const { chatsDir: _chatsDir, ...sdkOptions } = options;
  init(
    {
      ...sdkOptions,
      serviceName: options.serviceName ?? process.env.OTEL_SERVICE_NAME ?? "cursor",
      sdkName: "@telemetry-dev/cursor",
      registerGlobal: false,
    },
    overrides,
  );
}

/** Test seam for unit tests that need a fresh SDK singleton. */
export async function resetForTesting(): Promise<void> {
  initialized = false;
  await shutdown();
}

function asRecord<T>(value: T): JsonRecord | undefined {
  return value !== null && !(value instanceof Function) && Object(value) === value
    ? (value as JsonRecord)
    : undefined;
}

function readString<T>(value: T): string | undefined {
  return typeof value === "string" ? value : undefined;
}
