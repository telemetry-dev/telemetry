import { reportError } from "@telemetry-dev/otel";

import { type ApiOptions, isRecord, request, resolveApi } from "./api.ts";

export interface ScoreOptions extends ApiOptions {
  /** The score name. `user_feedback` is a thumbs up (1) or down (0) from a user of your app. */
  name: string;
  /** The trace that the score is on. Pass `traceId`, `sessionId`, or both. */
  traceId?: string;
  /** A span of the trace, for a score on one span. */
  spanId?: string;
  sessionId?: string;
  /** Pass `value`, `label`, or both. */
  value?: number;
  label?: string;
  comment?: string;
}

/**
 * Why score failed. `code` is the code of the API error (`insufficient_scope`, `invalid_api_key`,
 * …), or `missing_api_key`, `invalid_score`, `timeout`, `network_error`, or `invalid_response`. A
 * response with a different status and no API error gives `http_<status>`.
 */
export class ScoreError extends Error {
  override readonly name = "ScoreError";

  constructor(
    readonly code: string,
    message: string,
    readonly status?: number,
    options?: { cause?: unknown },
  ) {
    super(message, options);
  }
}

export const USER_FEEDBACK = "user_feedback";
export const MAX_SCORE_NAME_LENGTH = 64;
export const MAX_LABEL_LENGTH = 64;
export const MAX_REASON_LENGTH = 4_000;
const MAX_SESSION_ID_LENGTH = 1_024;
const TRACE_ID = /^(?!0+$)[0-9a-f]{32}$/;
const SPAN_ID = /^(?!0+$)[0-9a-f]{16}$/;
const TIMEOUT_MS = 10_000;

/**
 * Records a score on a trace, a span, or a session, for example the feedback of a user on a
 * reply. Like the telemetry functions, score never rejects: it resolves with the id of the score,
 * or with undefined after it reports the error through `onError`. It makes one request, with no
 * retry, so that a call in a request handler ends in 10 seconds at most. The trace id of a span
 * from a client that is not initialized is all zeros, and score reports it as `invalid_score`.
 */
export async function score(options: ScoreOptions): Promise<string | undefined> {
  // A caller without types can pass no options, and score must not reject then either.
  const fields: Partial<ScoreOptions> = isRecord(options) ? options : {};
  const api = resolveApi(fields);

  try {
    const name = typeof fields.name === "string" ? fields.name.trim() : fields.name;
    const label = typeof fields.label === "string" ? fields.label.trim() : fields.label;
    const problem = checkScore({ ...fields, name, label });

    if (problem) throw new ScoreError("invalid_score", problem);

    if (!api.apiKey) {
      throw new ScoreError(
        "missing_api_key",
        "No API key: pass apiKey or set TELEMETRY_DEV_API_KEY",
      );
    }

    const body = await request({ ...api, apiKey: api.apiKey }, ScoreError, "POST", "/v1/scores", {
      body: {
        traceId: fields.traceId,
        spanId: fields.spanId,
        sessionId: fields.sessionId,
        name,
        value: fields.value,
        label,
        comment: fields.comment,
      },
      timeoutMs: TIMEOUT_MS,
      idempotent: false,
      attempts: 1,
    });

    if (!isRecord(body) || typeof body.id !== "string") {
      throw new ScoreError("invalid_response", "telemetry.dev returned a score it cannot read");
    }

    return body.id;
  } catch (error) {
    reportError(api.onError, error);

    return undefined;
  }
}

/**
 * Why the server would reject the value and the label of a score, or undefined. A pass/fail score
 * has the label `pass` with the value 1, or `fail` with 0.
 */
export function checkValue(
  name: string,
  value: number | undefined,
  label: string | undefined,
): string | undefined {
  if (value !== undefined && !(typeof value === "number" && Number.isFinite(value))) {
    return "value must be a finite number";
  }

  if (
    label !== undefined &&
    (typeof label !== "string" || label.length === 0 || label.length > MAX_LABEL_LENGTH)
  ) {
    return `label must be a string of 1 to ${MAX_LABEL_LENGTH} characters`;
  }

  if (value === undefined && label === undefined) return "Pass a value or a label";

  if (name === USER_FEEDBACK && !(label === undefined && (value === 0 || value === 1))) {
    return `${USER_FEEDBACK} takes the value 1 for up or 0 for down, and no label`;
  }

  if (label === "pass" && value !== undefined && value !== 1) {
    return "The label pass goes with the value 1";
  }

  if (label === "fail" && value !== undefined && value !== 0) {
    return "The label fail goes with the value 0";
  }

  return undefined;
}

function checkScore(fields: Partial<ScoreOptions>): string | undefined {
  const { name, traceId, spanId, sessionId, value, label, comment } = fields;

  if (typeof name !== "string" || name.length === 0 || name.length > MAX_SCORE_NAME_LENGTH) {
    return `name must be a string of 1 to ${MAX_SCORE_NAME_LENGTH} characters`;
  }

  if (traceId === undefined && sessionId === undefined) return "Pass traceId or sessionId";

  if (traceId !== undefined && !(typeof traceId === "string" && TRACE_ID.test(traceId))) {
    return "traceId must be 32 lowercase hex digits and not all zeros";
  }

  if (spanId !== undefined && !(typeof spanId === "string" && SPAN_ID.test(spanId))) {
    return "spanId must be 16 lowercase hex digits and not all zeros";
  }

  if (spanId !== undefined && traceId === undefined) return "Pass the traceId of the span";

  if (
    sessionId !== undefined &&
    (typeof sessionId !== "string" ||
      sessionId.length === 0 ||
      sessionId.length > MAX_SESSION_ID_LENGTH)
  ) {
    return `sessionId must be a string of 1 to ${MAX_SESSION_ID_LENGTH} characters`;
  }

  if (
    comment !== undefined &&
    (typeof comment !== "string" || comment.length > MAX_REASON_LENGTH)
  ) {
    return `comment must be a string of at most ${MAX_REASON_LENGTH} characters`;
  }

  return checkValue(name, value, label);
}
