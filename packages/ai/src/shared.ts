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
