import { jsonAttr, reportError } from "@telemetry-dev/otel";

import type { MaskFn } from "./config.ts";

// Byte-identical with the Python SDK so truncated payloads are self-describing across languages.
export const TRUNCATION_MARKER = "...[truncated]";

export interface CaptureConfig {
  mask?: MaskFn;
  maxAttributeLength: number;
  onError?: (cause: Error) => void;
}

export interface BoundedCaptureOptions {
  maxBytes?: number;
  maxDepth?: number;
  maxItems?: number;
  skip?: (key: string, value: unknown, parent: unknown, path: readonly string[]) => boolean;
}

export interface BoundedCaptureResult {
  value: unknown;
  truncated: boolean;
}

export interface BoundedCaptureDetails extends BoundedCaptureResult {
  bytes: number;
  items: number;
}

interface CapturedObject {
  [key: string]: CapturedValue;
}

type CapturedValue = null | boolean | number | string | CapturedValue[] | CapturedObject;

const jsonStringByteLength = (value: string, limit: number) => {
  let bytes = 2;

  for (let index = 0; index < value.length && bytes <= limit; index += 1) {
    const code = value.charCodeAt(index);

    if (
      code === 0x22 ||
      code === 0x5c ||
      code === 0x08 ||
      code === 0x09 ||
      code === 0x0a ||
      code === 0x0c ||
      code === 0x0d
    ) {
      bytes += 2;
    } else if (code < 0x20) {
      bytes += 6;
    } else if (code < 0x80) {
      bytes += 1;
    } else if (code < 0x800) {
      bytes += 2;
    } else if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);

      if (next >= 0xdc00 && next <= 0xdfff) {
        bytes += 4;
        index += 1;
      } else {
        bytes += 6;
      }
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      bytes += 6;
    } else {
      bytes += 3;
    }
  }

  return bytes;
};

const captureLimit = (name: string, value: number | undefined, fallback: number) => {
  if (value === undefined) return fallback;

  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative safe integer`);
  }

  return value;
};

export function boundedCaptureDetails(
  input: unknown,
  options: BoundedCaptureOptions = {},
): BoundedCaptureDetails {
  const maxBytes = captureLimit("maxBytes", options.maxBytes, 48 * 1024);
  const maxDepth = captureLimit("maxDepth", options.maxDepth, 32);
  const maxItems = captureLimit("maxItems", options.maxItems, 1_000);
  const ancestors = new WeakSet<object>();
  let bytes = 0;
  let items = 0;
  let truncated = false;

  const visit = (
    value: unknown,
    depth: number,
    path: readonly string[],
  ): CapturedValue | undefined => {
    if (items >= maxItems) {
      truncated = true;

      return undefined;
    }

    items += 1;

    if (
      value === null ||
      typeof value === "boolean" ||
      typeof value === "number" ||
      typeof value === "string"
    ) {
      const length =
        typeof value === "string"
          ? jsonStringByteLength(value, maxBytes - bytes)
          : (JSON.stringify(value)?.length ?? 0);

      if (bytes + length > maxBytes) {
        truncated = true;

        return undefined;
      }

      bytes += length;

      return value;
    }

    if (typeof value !== "object") return undefined;

    if (depth >= maxDepth || ancestors.has(value) || bytes + 2 > maxBytes) {
      truncated = true;

      return undefined;
    }

    ancestors.add(value);
    bytes += 2;

    try {
      if (Array.isArray(value)) {
        const result: CapturedValue[] = [];

        for (const item of value) {
          if (items >= maxItems) {
            truncated = true;
            break;
          }

          const previousBytes = bytes;

          if (result.length > 0) bytes += 1;
          const converted = visit(item, depth + 1, path);

          if (converted === undefined) {
            bytes = previousBytes;
          } else {
            result.push(converted);
          }
        }

        return result;
      }

      const result = {} as CapturedObject;
      let emitted = false;

      for (const key in value) {
        if (items >= maxItems) {
          truncated = true;
          break;
        }

        if (!Object.hasOwn(value, key)) {
          items += 1;
          continue;
        }

        const keyBytes = jsonStringByteLength(key, maxBytes - bytes) + 1 + (emitted ? 1 : 0);

        if (bytes + keyBytes > maxBytes) {
          items += 1;
          truncated = true;
          continue;
        }

        let item: unknown;

        try {
          const descriptor = Object.getOwnPropertyDescriptor(value, key);
          item = descriptor?.get ? descriptor.get.call(value) : descriptor?.value;

          if (options.skip?.(key, item, value, path)) {
            items += 1;
            continue;
          }
        } catch {
          items += 1;
          truncated = true;
          continue;
        }

        const previousBytes = bytes;
        bytes += keyBytes;
        const converted = visit(item, depth + 1, [...path, key]);

        if (converted === undefined) {
          bytes = previousBytes;
        } else {
          if (key === "__proto__") {
            Object.defineProperty(result, key, {
              configurable: true,
              enumerable: true,
              value: converted,
              writable: true,
            });
          } else {
            result[key] = converted;
          }

          emitted = true;
        }
      }

      return result;
    } catch {
      truncated = true;

      return undefined;
    } finally {
      ancestors.delete(value);
    }
  };

  return { value: visit(input, 0, []), truncated, bytes, items };
}

export function boundedCapture(
  input: unknown,
  options: BoundedCaptureOptions = {},
): BoundedCaptureResult {
  const { value, truncated } = boundedCaptureDetails(input, options);

  return { value, truncated };
}

export function truncate(value: string, maxLength: number): string {
  if (value.length <= maxLength) return value;

  const limit = Math.max(maxLength, 0);
  const marker = TRUNCATION_MARKER.slice(0, limit);

  return value.slice(0, limit - marker.length) + marker;
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
