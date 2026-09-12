import type { Attributes } from "@opentelemetry/api";

// Pinned by docs/sdk-conformance.md ("instrumentation scopes @telemetry-dev/sdk and
// telemetry_dev") and asserted by the ingest e2e (C11). Lives here so the metrics
// pipeline and BYO processor emit the identical scope; do NOT rename to the package name.
export const SCOPE_NAME = "@telemetry-dev/sdk";
export const SCOPE_VERSION = "0.0.0";

export function omitUndefined(attributes: Attributes): Attributes {
  const out: Attributes = {};

  for (const key of Object.keys(attributes)) {
    const value = attributes[key];

    if (value !== undefined) {
      out[key] = value;
    }
  }

  return out;
}

// Stringify structured content (messages / tool args) for the gen_ai.* string attributes the
// ingest parses back into JSON. Returns undefined so omitUndefined drops absent content.
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export function jsonAttr<T>(value: T): string | undefined {
  if (value === undefined) return undefined;

  if (typeof value === "string") return value;

  try {
    return JSON.stringify(value);
  } catch {
    return undefined;
  }
}
