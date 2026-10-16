import type { Exception } from "@opentelemetry/api";
import { jsonAttr } from "@telemetry-dev/otel";

import type { ResolvedConfig, TelemetryDevOptions } from "./config.ts";

export type JsonValue =
  | string
  | number
  | boolean
  | bigint
  | null
  | undefined
  | JsonValue[]
  | { [key: string]: JsonValue };

// Helpers shared by the two package entries (`.` → ai@7, `./v6` → ai@6). They live in their own
// module because the entries must not import each other.

export function readId(value: JsonValue): string | null {
  if (typeof value === "string") {
    return value.length > 0 ? value : null;
  }

  if (typeof value === "number" || typeof value === "bigint") {
    return String(value);
  }

  return null;
}

// The Vercel AI Gateway reports its provider id as the bare "gateway". Surface its product name so
// traces read "Vercel AI Gateway" instead of an opaque slug; every other provider passes through.
// Applied at both provider read sites so the chain root and its model steps never disagree.
export function providerLabel(provider: string): string {
  return provider === "gateway" ? "Vercel AI Gateway" : provider;
}

/**
 * Masks and JSON-stringifies one piece of captured content. Undefined means write nothing: the
 * value is absent or unserializable, the mask dropped it, or the mask threw.
 */
export function contentAttr(
  config: Pick<ResolvedConfig, "mask" | "onError">,
  key: string,
  value: unknown,
): string | undefined {
  if (value === undefined || !config.mask) return jsonAttr(value);
  let masked: unknown;

  try {
    masked = config.mask(value, { key });
  } catch (error) {
    reportTelemetryError(config.onError, error);

    return undefined;
  }

  return jsonAttr(masked);
}

/**
 * What a failed wrapper call records as its span status message and exception. Without a mask
 * that is the error itself; with one it is the masked message, or `fallback` when the mask omits
 * it or throws, and no stack trace, which repeats the message.
 */
export function failureRecord(
  config: Pick<ResolvedConfig, "mask" | "onError">,
  error: unknown,
  fallback: string,
): { message: string; exception: Exception } {
  if (!config.mask) {
    let message = fallback;

    try {
      message = unknownErrorMessage(error);
    } catch {
      // An error whose string conversion throws is still recorded, with the fallback text.
    }

    return { message, exception: error instanceof Error ? error : message };
  }

  let name: string | undefined;
  let detail: string | undefined;

  try {
    if (error instanceof Error) {
      name = typeof error.name === "string" && error.name.length > 0 ? error.name : undefined;
      detail = error.message;
    } else {
      detail = unknownErrorMessage(error);
    }
  } catch {
    detail = undefined;
  }

  const message =
    (detail ? contentAttr(config, "exception.message", detail) : undefined) || fallback;

  return { message, exception: { name: name ?? "Error", message } };
}

/** Reports to onError without throwing, including for thrown values String() cannot convert. */
export function reportTelemetryError(
  onError: TelemetryDevOptions["onError"],
  error: unknown,
): void {
  let reported: Error | string;

  try {
    reported = error instanceof Error ? error : String(error);
  } catch {
    reported = new Error("A value that cannot be converted to a string was thrown");
  }

  try {
    onError?.(reported);
  } catch {
    // A throwing onError must not stop the caller from ending its spans.
  }
}

export function unknownErrorMessage(value: unknown): string {
  // oxlint-disable-next-line typescript/no-base-to-string -- error payloads must not be JSON-emitted.
  return String(value);
}

export const SEVERITY_INFO = 9;
export const SEVERITY_WARN = 13;
export const SEVERITY_ERROR = 17;

/**
 * Loose supertype of every telemetry event the hooks receive: each hook re-narrows to the
 * concrete `ai` major's event shape internally. Keeping the parameter this wide (`object`, so
 * TS's weak-type check never trips) makes the integration objects assignable to the ai@6
 * `TelemetryIntegration` and ai@7 `Telemetry` interfaces (function parameters are contravariant),
 * without the emitted types ever importing from `ai` — v6 installs lack `Telemetry` and v7
 * installs lack `TelemetryIntegration`.
 */
export type TelemetryDevEvent = object;
