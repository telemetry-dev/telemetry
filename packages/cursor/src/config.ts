import { readFileSync } from "node:fs";
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
 * options, which keeps the documented env-over-file precedence.
 */
export function fileConfig(): TelemetryDevCursorOptions {
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(configPath(), "utf8"));
  } catch {
    // A missing or malformed file must never crash the daemon.
    return {};
  }
  const record = asRecord(parsed);
  if (!record) return {};
  const pick = (key: string, envVar: string): string | undefined =>
    process.env[envVar] === undefined ? readString(record[key]) : undefined;
  return {
    apiKey: pick("apiKey", "TELEMETRY_DEV_API_KEY"),
    baseUrl: pick("baseUrl", "TELEMETRY_DEV_BASE_URL"),
    environment: pick("environment", "TELEMETRY_DEV_ENVIRONMENT"),
    serviceName: pick("serviceName", "OTEL_SERVICE_NAME"),
  };
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
  const raw: unknown = value;
  return String(raw) === raw ? raw : undefined;
}
