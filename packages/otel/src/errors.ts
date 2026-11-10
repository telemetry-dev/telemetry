// Error classes that provider SDKs throw for any failed request, whose name says nothing about the
// cause: openai's and Anthropic's APIError (a stream's error event), @google/genai's ApiError, and
// the AI SDK's APICallError and StreamProviderError. Plain Error is here for errors that carry a
// code, such as the failures that the integrations build from a response body with `code`,
// `status`, and `type` fields. The Python SDK keeps the same rule in `_error_type`; change both
// together.
const GENERIC_ERROR_NAMES = new Set([
  "Error",
  "APIError",
  "ApiError",
  "AI_APICallError",
  "AI_StreamProviderError",
]);

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : undefined;
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

// Some providers send numeric codes, such as Z.ai's "1302" for a rate limit, and openai's Python SDK
// turns OpenRouter's numeric code into a string. A number names no cause the way a class name does.
function errorCode(value: unknown): string | undefined {
  const code = text(value);

  return code === undefined || /^\d+$/.test(code) ? undefined : code;
}

function httpStatus(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value >= 400 && value <= 599
    ? value
    : undefined;
}

function className(error: Error): string {
  // Many SDK error classes (e.g. openai's BadRequestError) never override the inherited "Error"
  // name, so the constructor name is the specific one (Python parity: type(error).__name__).
  return error.name !== "Error" && error.name.length > 0
    ? error.name
    : (error.constructor?.name ?? "Error") || "Error";
}

/**
 * The `error.type` of a failure: the provider's error code when the error carries one, so that a
 * rate limit, an exhausted quota, an overflowing context, and an overloaded provider read apart.
 *
 * - An error with an HTTP status and a code that is not only digits (openai's APIError:
 *   `insufficient_quota`, `context_length_exceeded`) gives the code.
 * - A generic provider error class (see GENERIC_ERROR_NAMES) gives its code, else its HTTP status
 *   (`429`), else the provider's error type (Anthropic's `overloaded_error` from a stream).
 * - The AI SDK's RetryError gives the type of the last error it retried.
 * - Any other error gives its class name; a value that is not an Error gives `fallback`.
 */
export function errorTypeOf(error: unknown, fallback = "Error"): string {
  if (!(error instanceof Error)) return fallback;

  try {
    const fields = error as Error & Record<string, unknown>;

    if (fields.name === "AI_RetryError" && fields.lastError instanceof Error) {
      return errorTypeOf(fields.lastError, fallback);
    }

    const name = className(error);
    const data = record(record(fields.data)?.error);
    const code = errorCode(fields.code) ?? errorCode(data?.code);
    const status = httpStatus(fields.status) ?? httpStatus(fields.statusCode);

    if (code !== undefined && status !== undefined) return code;
    if (!GENERIC_ERROR_NAMES.has(name)) return name;
    if (code !== undefined) return code;
    if (status !== undefined) return String(status);

    return text(fields.type) ?? text(data?.type) ?? name;
  } catch {
    // A getter on a third-party error can throw; recording the error must not.
    return fallback;
  }
}
