import {
  type Attributes,
  type AttributeValue,
  type Context,
  createContextKey,
} from "@opentelemetry/api";
import {
  activeContext,
  DEFAULT_BASE_URL,
  jsonAttr,
  omitUndefined,
  reportError,
  resolveEnv,
  sessionIdOf,
  sha256,
  withContext,
} from "@telemetry-dev/otel";

import type { FieldCaptureConfig } from "./attrs.ts";
import { prepareCaptureValue } from "./capture.ts";
import { currentClient } from "./client.ts";

/** A message of a chat prompt. A placeholder stands for a list of messages that the app passes. */
export type PromptMessage =
  | { readonly role: "system" | "user" | "assistant"; readonly content: string }
  | { readonly role: "placeholder"; readonly name: string };

/** A message that `compile()` makes from a prompt message. */
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface PromptModel {
  readonly provider: string;
  readonly name: string;
}

/** The parameters that are set in the version. A parameter that is not set is absent. */
export interface PromptParameters {
  readonly temperature?: number;
  readonly topP?: number;
  readonly maxTokens?: number;
  readonly frequencyPenalty?: number;
  readonly presencePenalty?: number;
  readonly stop?: readonly string[];
  readonly seed?: number;
  readonly toolChoice?: "auto" | "none" | "required" | { readonly tool: string };
  readonly parallelToolCalls?: boolean;
}

export interface PromptTool {
  readonly name: string;
  readonly description: string;
  /** JSON schema of the tool arguments. */
  readonly parameters: Readonly<Record<string, unknown>>;
  readonly strict: boolean;
}

export type PromptResponseFormat =
  | { readonly type: "text" }
  | { readonly type: "json_object" }
  | {
      readonly type: "json_schema";
      readonly name: string;
      readonly schema: Readonly<Record<string, unknown>>;
      readonly strict: boolean;
    };

/**
 * Values for `{{name}}` variables. A missing value, null, or undefined becomes an empty text, and a
 * value that is not a string becomes JSON. A placeholder message takes a list of messages.
 */
export type PromptVariables<M = ChatMessage> = Readonly<
  Record<string, string | number | boolean | null | undefined | readonly M[]>
>;

interface PromptBase {
  readonly name: string;
  /** The version that the app got; null for the fallback. */
  readonly version: number | null;
  /** The label that the prompt was fetched by; null when it was fetched by version. */
  readonly label: string | null;
  /** Null for the fallback. */
  readonly model: PromptModel | null;
  readonly parameters: PromptParameters;
  readonly tools: readonly PromptTool[];
  readonly responseFormat: PromptResponseFormat;
  /** True when getPrompt could not fetch the prompt and returned the `fallback` option. */
  readonly isFallback: boolean;
}

export interface ChatPrompt extends PromptBase {
  readonly type: "chat";
  readonly messages: readonly PromptMessage[];
  readonly text: null;
  /**
   * Fills in the variables and replaces each placeholder message with the messages in its
   * variable. `withPrompt` records the variables of the last call before it starts or inside it.
   * It tags each message object that it makes with this prompt and these variables, so that a
   * model call outside `withPrompt` that sends the message records them; see `promptAttributesFor`.
   */
  compile<M = ChatMessage>(variables?: PromptVariables<M>): (ChatMessage | M)[];
}

export interface TextPrompt extends PromptBase {
  readonly type: "text";
  readonly messages: readonly [];
  readonly text: string;
  /**
   * Fills in the variables. `withPrompt` records the variables of the last call before it starts
   * or inside it. A string cannot carry a tag, so a model call outside `withPrompt` does not
   * record a text prompt.
   */
  compile(variables?: PromptVariables): string;
}

export type Prompt = ChatPrompt | TextPrompt;

interface GetPromptOptionsBase {
  /** Fetch the version that this label points to. Default "production" when `version` is not set. */
  label?: string;
  /** Fetch this version number. Do not set it together with `label`. */
  version?: number;
  /** Seconds before a call fetches a kept prompt again, in the background. Default 60. */
  cacheTtlSeconds?: number;
  /** API key with the prompts:read scope. Falls back to the initialized client, then `TELEMETRY_DEV_API_KEY`. */
  apiKey?: string;
  /** Falls back to the initialized client, then `TELEMETRY_DEV_BASE_URL`, then `https://api.telemetry.dev`. */
  baseUrl?: string;
  /** Falls back to the initialized client's `fetch`, then `globalThis.fetch`. */
  fetch?: typeof fetch;
}

export interface GetChatPromptOptions extends GetPromptOptionsBase {
  /** Default "chat". getPrompt rejects a prompt of the other type. */
  type?: "chat";
  /** Messages to use when the prompt cannot be fetched and no copy of it is kept. */
  fallback?: readonly PromptMessage[];
}

export interface GetTextPromptOptions extends GetPromptOptionsBase {
  type: "text";
  /** Text to use when the prompt cannot be fetched and no copy of it is kept. */
  fallback?: string;
}

export type GetPromptOptions = GetChatPromptOptions | GetTextPromptOptions;

/**
 * Why getPrompt failed. `code` is the code of the API error (`prompt_not_found`,
 * `label_not_found`, `insufficient_scope`, `invalid_api_key`, …), or `missing_api_key`,
 * `timeout`, `network_error`, `invalid_response`, or `type_mismatch`. A response with a different
 * status and no API error gives `http_<status>`, for example `http_302` for a redirect: getPrompt
 * does not go to the location of a redirect.
 */
export class PromptError extends Error {
  override readonly name = "PromptError";

  constructor(
    readonly code: string,
    message: string,
    readonly status?: number,
    options?: { cause?: unknown },
  ) {
    super(message, options);
  }
}

interface PromptContent {
  readonly messages: readonly PromptMessage[];
  readonly text: string | null;
  readonly model: PromptModel;
  readonly parameters: PromptParameters;
  readonly tools: readonly PromptTool[];
  readonly responseFormat: PromptResponseFormat;
}

interface PromptResponse extends PromptContent {
  readonly name: string;
  readonly type: "chat" | "text";
  readonly label: string | null;
  readonly version: number;
  readonly rollout: (PromptContent & { readonly percent: number; readonly version: number }) | null;
}

interface CacheEntry {
  response?: PromptResponse;
  /** Why the last fetch failed, when there was no response to keep. */
  error?: unknown;
  /** When the last fetch ended, in `performance.now()` time, so that a clock change does not move it. */
  checkedAt: number;
  /** No fetch before this `performance.now()` time: the Retry-After of the last 429 or 503 response. */
  retryAt: number;
  pending?: Promise<PromptResponse>;
}

/** The prompt of a `withPrompt` call and the variables that its spans record. */
interface PromptScope {
  readonly prompt: Prompt;
  variables: object | undefined;
}

/** The `compile()` call that made a chat message. */
interface CompileTag {
  readonly prompt: Prompt;
  readonly variables: PromptVariables<unknown>;
}

const FETCH_TIMEOUT_MS = 10_000;
/** The longest Retry-After that getPrompt honors, so that a bad header cannot stop fetches. */
const MAX_RETRY_AFTER_MS = 3_600_000;
/** How long a failed fetch with no copy answers calls, so that they do not each fetch again. */
const ERROR_TTL_MS = 10_000;
const MAX_CACHE_ENTRIES = 1_000;
/** IMF-fixdate, the HTTP date that servers send: `Sun, 06 Nov 1994 08:49:37 GMT`. */
const HTTP_DATE = /^[A-Z][a-z]{2}, \d{2} [A-Z][a-z]{2} \d{4} \d{2}:\d{2}:\d{2} GMT$/;
const VARIABLE = /\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}/g;
const MODEL_CALL_OPERATIONS = new Set<unknown>(["chat", "text_completion", "generate_content"]);
const PROMPT_KEY = createContextKey("telemetry.dev prompt");
// Not enumerable, so JSON.stringify, spreads, and structuredClone leave the tag out of a copy.
const COMPILED_BY = Symbol("telemetry.dev compiled by");
/** How many messages promptAttributesFor looks at, from the end. */
const MAX_TAG_LOOKUP = 1_000;

const cache = new Map<string, CacheEntry>();
const compiledVariables = new WeakMap<Prompt, object>();
/** The Retry-After of the response that failed with this error, in milliseconds. */
const retryAfter = new WeakMap<PromptError, number>();

/**
 * Fetches a prompt version from telemetry.dev and keeps it. When the copy is older than
 * `cacheTtlSeconds`, a call returns it at once and fetches a new one in the background. Only a call
 * with no copy waits for a fetch, at most 10 seconds. Calls for the same prompt share one fetch.
 * When a fetch fails, getPrompt returns the last copy that it kept; with no copy it returns
 * `fallback`. With neither it rejects with a `PromptError`: unlike the telemetry functions,
 * getPrompt cannot fail open, because the app cannot run without the prompt. With no copy, the
 * calls of the next 10 seconds (or the TTL, if it is shorter) fail the same way without a fetch.
 *
 * A label with a gradual rollout gives each session one version: the choice uses the session id
 * of `propagateAttributes`, or a random choice per call without one.
 */
export function getPrompt(name: string, options: GetTextPromptOptions): Promise<TextPrompt>;
export function getPrompt(name: string, options?: GetChatPromptOptions): Promise<ChatPrompt>;
export async function getPrompt(name: string, options: GetPromptOptions = {}): Promise<Prompt> {
  const type = options.type ?? "chat";
  checkOptions(name, options, type);

  const core = currentClient().core;
  const env = resolveEnv();
  const apiKey = options.apiKey ?? core?.config.apiKey ?? env.TELEMETRY_DEV_API_KEY;

  // An empty value counts as unset, as in the Python SDK.
  const baseUrl = (
    options.baseUrl ||
    core?.config.baseUrl ||
    env.TELEMETRY_DEV_BASE_URL ||
    DEFAULT_BASE_URL
  ).replace(/\/+$/, "");

  const label = options.version === undefined ? (options.label ?? "production") : null;
  const onError = core?.config.onError;
  let response: PromptResponse;

  try {
    if (!apiKey) {
      throw new PromptError(
        "missing_api_key",
        "No API key: pass apiKey or set TELEMETRY_DEV_API_KEY",
      );
    }

    const url = new URL(`${baseUrl}/v1/prompts`);
    url.searchParams.set("name", name);

    if (label === null) url.searchParams.set("version", String(options.version));
    else url.searchParams.set("label", label);

    response = await load(
      [
        baseUrl,
        keyFingerprint(apiKey),
        name,
        label === null ? `version:${options.version}` : `label:${label}`,
      ].join("\0"),
      (options.cacheTtlSeconds ?? 60) * 1000,
      () =>
        fetchPrompt(
          url,
          {
            authorization: `Bearer ${apiKey}`,
            "x-telemetry-dev-sdk": core?.config.sdkName ?? "@telemetry-dev/sdk",
          },
          options.fetch ?? core?.config.fetchImpl ?? globalThis.fetch,
        ),
      onError,
    );
  } catch (error) {
    if (options.fallback === undefined) throw error;
    reportError(onError, error);

    return createPrompt({
      name,
      type,
      label,
      version: null,
      isFallback: true,
      content: {
        messages: type === "chat" ? (options.fallback as readonly PromptMessage[]) : [],
        text: type === "text" ? (options.fallback as string) : null,
        model: null,
        parameters: {},
        tools: [],
        responseFormat: { type: "text" },
      },
    });
  }

  if (response.type !== type) {
    throw new PromptError(
      "type_mismatch",
      `${name} is a ${response.type} prompt; pass type: "${response.type}"`,
    );
  }

  const chosen = response.rollout && inRollout(response) ? response.rollout : response;

  return createPrompt({
    name: response.name,
    type,
    label: response.label,
    version: chosen.version,
    isFallback: false,
    content: chosen,
  });
}

/**
 * Runs `fn` with `prompt` as the prompt of each model call (generation span) that starts inside
 * it: the spans get `td.prompt.name`, `td.prompt.version`, `td.prompt.label`, and, when input
 * capture is on, `td.prompt.variables`. The variables are those of the last `compile()` of this
 * prompt object before withPrompt starts, or of the last `compile()` inside `fn`, so a compile in
 * another task does not change them once withPrompt runs. Concurrent tasks in `fn` share the
 * variables, so call withPrompt in each task that compiles its own.
 */
export function withPrompt<T>(prompt: Prompt, fn: () => T): T {
  const scope: PromptScope = { prompt, variables: compiledVariables.get(prompt) };

  return withContext(activeContext().setValue(PROMPT_KEY, scope), fn);
}

/** The td.prompt.* attributes for a span with this operation that starts in `ctx`. */
export function promptAttributes(
  ctx: Context,
  operation: unknown,
  capture: FieldCaptureConfig,
): Attributes | undefined {
  if (!MODEL_CALL_OPERATIONS.has(operation)) return undefined;

  const scope = (ctx.getValue(PROMPT_KEY) ?? activeContext().getValue(PROMPT_KEY)) as
    | PromptScope
    | undefined;

  if (!scope) return undefined;

  return attributesOf(scope.prompt, scope.variables, capture);
}

/**
 * The td.prompt.* attributes for a model call outside `withPrompt` that sends `messages` (a list
 * of messages, or one message): those of the newest message, of the last 1000, that `compile()`
 * of a chat prompt returned. The tag is on the message object, so a copy of it gives nothing. Gives
 * undefined inside `withPrompt`, whose prompt the span gets. `td.prompt.variables` needs input
 * capture on in the client, and the span drops it when its own input capture is off. It never
 * throws.
 */
export function promptAttributesFor(messages: unknown): Record<string, AttributeValue> | undefined {
  const config = currentClient().core?.config;

  try {
    if (activeContext().getValue(PROMPT_KEY) !== undefined) return undefined;

    const tag = newestTag(messages);

    if (!tag) return undefined;

    return attributesOf(tag.prompt, tag.variables, config) as Record<string, AttributeValue>;
  } catch (error) {
    reportError(config?.onError, error);

    return undefined;
  }
}

function attributesOf(
  prompt: Prompt,
  variables: object | undefined,
  capture: FieldCaptureConfig | undefined,
): Attributes {
  return omitUndefined({
    "td.prompt.name": prompt.name,
    "td.prompt.version": prompt.version ?? undefined,
    "td.prompt.label": prompt.label ?? undefined,
    "td.prompt.variables":
      variables !== undefined && capture?.captureInput
        ? prepareCaptureValue("td.prompt.variables", variables, capture)
        : undefined,
  });
}

function newestTag(messages: unknown): CompileTag | undefined {
  if (!Array.isArray(messages)) return tagOf(messages);
  const stop = Math.max(0, messages.length - MAX_TAG_LOOKUP);

  for (let index = messages.length - 1; index >= stop; index -= 1) {
    const tag = tagOf(messages[index]);

    if (tag) return tag;
  }

  return undefined;
}

function tagOf(message: unknown): CompileTag | undefined {
  return typeof message === "object" && message !== null
    ? (message as { [COMPILED_BY]?: CompileTag })[COMPILED_BY]
    : undefined;
}

function checkOptions(name: string, options: GetPromptOptions, type: string): void {
  if (typeof name !== "string" || name.length === 0) {
    throw new TypeError("name must be a non-empty string");
  }

  if (type !== "chat" && type !== "text") throw new TypeError('type must be "chat" or "text"');

  if (options.label !== undefined && options.version !== undefined) {
    throw new TypeError("Pass label or version, not both");
  }

  if (
    options.version !== undefined &&
    (!Number.isSafeInteger(options.version) || options.version < 1)
  ) {
    throw new RangeError("version must be a positive integer");
  }

  const ttl = options.cacheTtlSeconds;

  if (ttl !== undefined && !(Number.isFinite(ttl) && ttl >= 0)) {
    throw new RangeError("cacheTtlSeconds must be a non-negative number");
  }

  if (
    options.fallback !== undefined &&
    (type === "chat" ? !Array.isArray(options.fallback) : typeof options.fallback !== "string")
  ) {
    throw new TypeError(
      type === "chat" ? "fallback must be a list of messages" : "fallback must be a string",
    );
  }
}

/**
 * Returns the kept copy of a prompt, and fetches a new copy in the background when it is older
 * than the TTL. Only a call with no copy waits for a fetch. A failed fetch with no copy fails the
 * calls of the next ERROR_TTL_MS (or the TTL, if it is shorter) without a fetch.
 */
async function load(
  key: string,
  ttlMs: number,
  fetchResponse: () => Promise<PromptResponse>,
  onError: ((cause: Error) => void) | undefined,
): Promise<PromptResponse> {
  let entry = cache.get(key);

  if (!entry) {
    if (cache.size >= MAX_CACHE_ENTRIES) cache.delete(cache.keys().next().value!);
    entry = { checkedAt: Number.NEGATIVE_INFINITY, retryAt: 0 };
    cache.set(key, entry);
  }

  const fresh =
    performance.now() <
    Math.max(
      entry.checkedAt + (entry.response ? ttlMs : Math.min(ttlMs, ERROR_TTL_MS)),
      entry.retryAt,
    );

  if (entry.response) {
    if (!fresh) entry.pending ??= refresh(entry, fetchResponse, onError);

    return entry.response;
  }

  if (fresh) throw entry.error;

  return (entry.pending ??= refresh(entry, fetchResponse, onError));
}

function refresh(
  entry: CacheEntry,
  fetchResponse: () => Promise<PromptResponse>,
  onError: ((cause: Error) => void) | undefined,
): Promise<PromptResponse> {
  return fetchResponse()
    .then(
      (response) => {
        entry.response = response;
        entry.retryAt = 0;

        return response;
      },
      (error: unknown) => {
        const delay = error instanceof PromptError ? retryAfter.get(error) : undefined;
        entry.retryAt = delay === undefined ? 0 : performance.now() + delay;

        if (!entry.response) {
          entry.error = error;
          throw error;
        }

        reportError(onError, error);

        return entry.response;
      },
    )
    .finally(() => {
      // A failed fetch counts too, so that calls do not each start a fetch while it fails.
      entry.checkedAt = performance.now();
      entry.pending = undefined;
    });
}

async function fetchPrompt(
  url: URL,
  headers: Record<string, string>,
  fetchImpl: typeof fetch,
): Promise<PromptResponse> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    const res = await fetchImpl(url, {
      method: "GET",
      headers,
      signal: controller.signal,
      // A redirect to another host must not get the API key, so a 3xx fails below like any status
      // that is not OK. Cloudflare Workers reject redirect: "error".
      redirect: "manual",
    });

    const body: unknown = await res.json().catch(() => undefined);

    if (!res.ok) {
      const error = isRecord(body) && isRecord(body.error) ? body.error : {};

      const failure = new PromptError(
        typeof error.code === "string" ? error.code : `http_${res.status}`,
        typeof error.message === "string" ? error.message : `telemetry.dev returned ${res.status}`,
        res.status,
      );

      const delay = retryAfterMs(res);
      if (delay !== undefined) retryAfter.set(failure, delay);

      throw failure;
    }

    const response = parseResponse(body);

    if (!response) {
      throw new PromptError("invalid_response", "telemetry.dev returned a prompt it cannot read");
    }

    return response;
  } catch (error) {
    // The timeout can also cut the body, which then reads as an invalid response.
    if (controller.signal.aborted) {
      throw new PromptError("timeout", `Fetching the prompt took more than ${FETCH_TIMEOUT_MS} ms`);
    }

    if (error instanceof PromptError) throw error;

    throw new PromptError("network_error", "Could not reach telemetry.dev", undefined, {
      cause: error,
    });
  } finally {
    clearTimeout(timer);
  }
}

function createPrompt(input: {
  name: string;
  type: "chat" | "text";
  label: string | null;
  version: number | null;
  isFallback: boolean;
  content: Omit<PromptContent, "model"> & { readonly model: PromptModel | null };
}): Prompt {
  const { content, type } = input;

  const prompt: Prompt = Object.freeze({
    name: input.name,
    type,
    version: input.version,
    label: input.label,
    messages: content.messages,
    text: content.text,
    model: content.model,
    parameters: content.parameters,
    tools: content.tools,
    responseFormat: content.responseFormat,
    isFallback: input.isFallback,
    compile(variables: PromptVariables<unknown> = {}) {
      compiledVariables.set(prompt, variables);
      const scope = activeContext().getValue(PROMPT_KEY) as PromptScope | undefined;

      if (scope?.prompt === prompt) scope.variables = variables;

      return type === "chat"
        ? compileMessages(content.messages, Object.freeze({ prompt, variables }))
        : fill(content.text ?? "", variables);
    },
  }) as Prompt;

  return prompt;
}

function compileMessages(messages: readonly PromptMessage[], tag: CompileTag): unknown[] {
  const { variables } = tag;

  return messages.flatMap((message) => {
    if (message.role !== "placeholder") {
      const compiled = { role: message.role, content: fill(message.content, variables) };
      Object.defineProperty(compiled, COMPILED_BY, { value: tag });

      return [compiled];
    }

    const value = Object.hasOwn(variables, message.name) ? variables[message.name] : undefined;

    if (value == null) return [];

    if (!Array.isArray(value)) {
      throw new TypeError(`The ${message.name} variable must be a list of messages`);
    }

    return value;
  });
}

function fill(text: string, variables: PromptVariables<unknown>): string {
  return text.replace(VARIABLE, (_tag, name: string) => {
    const value = Object.hasOwn(variables, name) ? variables[name] : undefined;

    if (value == null) return "";

    return typeof value === "string" ? value : (jsonAttr(value) ?? "");
  });
}

// One session keeps one version: FNV-1a 32-bit of "<name>:<label>:<session id>", mod 100.
function inRollout(response: PromptResponse): boolean {
  const sessionId = sessionIdOf(activeContext()) ?? currentClient().core?.config.processSessionId;

  const bucket =
    sessionId === undefined
      ? Math.floor(Math.random() * 100)
      : fnv1a(`${response.name}:${response.label}:${sessionId}`) % 100;

  return bucket < response.rollout!.percent;
}

function fnv1a(text: string): number {
  let hash = 2166136261;

  for (const byte of new TextEncoder().encode(text)) {
    hash = Math.imul(hash ^ byte, 16777619) >>> 0;
  }

  return hash;
}

function keyFingerprint(apiKey: string): string {
  const digest = sha256(new TextEncoder().encode(apiKey)).subarray(0, 16);

  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** The Retry-After of a 429 or 503 response, as seconds or an IMF-fixdate HTTP date, capped. */
function retryAfterMs(res: Response): number | undefined {
  if (res.status !== 429 && res.status !== 503) return undefined;

  const value = res.headers.get("retry-after")?.trim();

  if (!value) return undefined;

  let ms: number;

  if (/^\d+$/.test(value)) ms = Number(value) * 1000;
  else if (HTTP_DATE.test(value)) ms = Date.parse(value) - Date.now();
  else return undefined;

  return ms > 0 ? Math.min(ms, MAX_RETRY_AFTER_MS) : undefined;
}

function parseResponse(body: unknown): PromptResponse | undefined {
  if (!isRecord(body)) return undefined;
  const { name, type, label, version, rollout } = body;

  if (
    typeof name !== "string" ||
    (type !== "chat" && type !== "text") ||
    (label !== null && typeof label !== "string") ||
    !isVersion(version)
  ) {
    return undefined;
  }

  const content = parseContent(body, type);

  if (!content) return undefined;

  if (rollout == null) return deepFreeze({ name, type, label, version, ...content, rollout: null });

  if (
    !isRecord(rollout) ||
    !isVersion(rollout.version) ||
    !Number.isInteger(rollout.percent) ||
    (rollout.percent as number) < 0 ||
    (rollout.percent as number) > 100
  ) {
    return undefined;
  }

  const rolloutContent = parseContent(rollout, type);

  if (!rolloutContent) return undefined;

  return deepFreeze({
    name,
    type,
    label,
    version,
    ...content,
    rollout: { percent: rollout.percent as number, version: rollout.version, ...rolloutContent },
  });
}

function parseContent(
  value: Record<string, unknown>,
  type: "chat" | "text",
): PromptContent | undefined {
  const { messages, text, model, parameters, tools, responseFormat } = value;

  const validBody =
    type === "chat"
      ? Array.isArray(messages) && messages.every(isPromptMessage)
      : typeof text === "string";

  if (
    !validBody ||
    !isRecord(model) ||
    typeof model.provider !== "string" ||
    typeof model.name !== "string" ||
    !isRecord(parameters) ||
    !Array.isArray(tools) ||
    !tools.every((tool) => isRecord(tool) && typeof tool.name === "string") ||
    !isRecord(responseFormat) ||
    typeof responseFormat.type !== "string"
  ) {
    return undefined;
  }

  return {
    messages: type === "chat" ? (messages as PromptMessage[]) : [],
    text: type === "text" ? (text as string) : null,
    model: model as unknown as PromptModel,
    parameters: parameters as PromptParameters,
    tools: tools as PromptTool[],
    responseFormat: responseFormat as unknown as PromptResponseFormat,
  };
}

function isPromptMessage(value: unknown): value is PromptMessage {
  if (!isRecord(value)) return false;

  return value.role === "placeholder"
    ? typeof value.name === "string"
    : (value.role === "system" || value.role === "user" || value.role === "assistant") &&
        typeof value.content === "string";
}

function isVersion(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 1;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Prompt objects share the cached response, so an app cannot change the cache through them.
function deepFreeze<T>(value: T): T {
  if (typeof value === "object" && value !== null && !Object.isFrozen(value)) {
    Object.freeze(value);

    for (const item of Object.values(value)) deepFreeze(item);
  }

  return value;
}
