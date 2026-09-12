import { jsonAttr, reportError } from "@telemetry-dev/otel";

import type { MaskFn } from "./config.ts";

// Byte-identical with the Python SDK so truncated payloads are self-describing across languages.
export const TRUNCATION_MARKER = "...[truncated]";

export interface CaptureConfig {
  mask?: MaskFn;
  maxAttributeLength: number;
  onError?: (cause: Error) => void;
}

export function truncate(value: string, maxLength: number): string {
  if (value.length <= maxLength) return value;

  // Total stays within maxLength so the provider's attributeValueLengthLimit backstop
  // (set to the same cap) never slices the marker off.
  return value.slice(0, Math.max(maxLength - TRUNCATION_MARKER.length, 0)) + TRUNCATION_MARKER;
}

/** The single content funnel: mask → JSON stringify → truncate. */
export function prepareCaptureValue(
  key: string,
  value: Parameters<MaskFn>[0],
  cfg: CaptureConfig,
): string | undefined {
  let masked = value;

  if (cfg.mask) {
    try {
      masked = cfg.mask(value, { key });
    } catch (error) {
      reportError(cfg.onError, error instanceof Error ? error : new Error(String(error)));

      return undefined;
    }
  }

  const serialized = jsonAttr(masked);

  if (serialized === undefined) return undefined;

  return truncate(serialized, cfg.maxAttributeLength);
}
