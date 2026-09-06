/**
 * Google GenAI instrumentation for telemetry.dev.
 *
 * TypeScript has no global instrumentation entry point: `@google/genai` implements
 * `generateContent`, `generateContentStream`, and `embedContent` as arrow-function
 * instance fields on `Models`, so prototype patching cannot intercept them. Use
 * `wrapGoogleGenAI(client)` on each client you want traced.
 */

import type { AsyncLocalStorage } from "node:async_hooks";
import { startSpan, type SpanFields, type SpanHandle } from "@telemetry-dev/sdk";

type AttributeValue = NonNullable<SpanFields["attributes"]>[string];
type AlsConstructor = new <T>() => AsyncLocalStorage<T>;
type AsyncHooksModule = { AsyncLocalStorage?: AlsConstructor };

function loadAls(): AlsConstructor | undefined {
  const globals = globalThis as { AsyncLocalStorage?: AlsConstructor };
  if (globals.AsyncLocalStorage) return globals.AsyncLocalStorage;
  const proc = globalThis.process as { getBuiltinModule?: (id: string) => unknown } | undefined;
  if (typeof proc?.getBuiltinModule !== "function") return undefined;
  try {
    const mod = proc.getBuiltinModule("node:async_hooks") as AsyncHooksModule | undefined;
    return mod?.AsyncLocalStorage;
  } catch {
    return undefined;
  }
}

const WRAPPED = Symbol("telemetry.dev.google-genai.wrapped");
const ORIGINAL = Symbol("telemetry.dev.google-genai.original");
const wrappedClients = new WeakSet<object>();
const usageCollectorModels = new WeakSet<object>();
const AlsCtor = loadAls();
const afcUsageStore = AlsCtor
  ? new AlsCtor<{ usage?: SpanFields["usage"]; toolUsePromptTokens?: number }>()
  : undefined;

type UnknownRecord = Record<string, unknown>;

type WrappedFunction = ((...args: unknown[]) => unknown) & {
  [WRAPPED]?: true;
  [ORIGINAL]?: (...args: unknown[]) => unknown;
};

interface RequestMapping {
  name: string;
  fields: SpanFields & { type: "generation" | "embedding" };
}

export interface GoogleGenAIClientLike {
  models: object;
  vertexai?: boolean;
}

const BUILTIN_TOOL_KEYS = [
  "googleSearch",
  "googleSearchRetrieval",
  "codeExecution",
  "urlContext",
  "computerUse",
  "fileSearch",
  "retrieval",
  "googleMaps",
  "enterpriseWebSearch",
  "parallelAiSearch",
  "mcpServers",
] as const;

function asRecord(value: unknown): UnknownRecord | undefined {
  return value !== null && typeof value === "object" ? (value as UnknownRecord) : undefined;
}

function asArray(value: unknown): unknown[] | undefined {
  return Array.isArray(value) ? value : undefined;
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function readNumber(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

function isThenable(value: unknown): value is PromiseLike<unknown> {
  if (value === null || typeof value !== "object" || !("then" in value)) return false;
  return typeof value.then === "function";
}

function compactUsage(usage: SpanFields["usage"]): SpanFields["usage"] {
  if (!usage) return undefined;
  return Object.values(usage).some((value) => value !== undefined) ? usage : undefined;
}

function jsonStringify(value: unknown): string | undefined {
  try {
    return JSON.stringify(value);
  } catch {
    return undefined;
  }
}

function setJsonAttribute(
  attributes: Record<string, AttributeValue>,
  key: string,
  value: unknown,
): void {
  const serialized = jsonStringify(value);
  if (serialized !== undefined) attributes[key] = serialized;
}

function stopSequences(value: unknown): string[] | undefined {
  const arr = asArray(value);
  if (!arr) return undefined;
  const strings = arr.filter((item): item is string => typeof item === "string");
  return strings.length > 0 ? strings : undefined;
}

function outputTypeFromConfig(config: UnknownRecord | undefined): string | undefined {
  if (!config) return undefined;
  const mime = readString(config.responseMimeType);
  if (
    mime === "application/json" ||
    config.responseSchema !== undefined ||
    config.responseJsonSchema !== undefined
  ) {
    return "json";
  }
  if (mime === "text/plain") return "text";
  return undefined;
}

function mapToolDefinitions(tools: unknown): unknown[] | undefined {
  const toolArray = asArray(tools);
  if (!toolArray) return undefined;
  const definitions: unknown[] = [];
  for (const tool of toolArray) {
    const record = asRecord(tool);
    if (!record) continue;
    if (typeof record.callTool === "function") {
      const name = readString(record.name);
      if (name) definitions.push({ type: "function", name });
      continue;
    }
    for (const declaration of asArray(record.functionDeclarations) ?? []) {
      const fn = asRecord(declaration);
      if (!fn) continue;
      const definition = { type: "function", name: fn.name };
      if (fn.description !== undefined) Object.assign(definition, { description: fn.description });
      if (fn.parameters !== undefined) Object.assign(definition, { parameters: fn.parameters });
      definitions.push(definition);
    }
    for (const key of BUILTIN_TOOL_KEYS) {
      if (record[key] !== undefined) definitions.push({ type: key });
    }
  }
  return definitions.length > 0 ? definitions : undefined;
}

function requestAttributesFromConfig(config: UnknownRecord | undefined) {
  const attributes: Record<string, AttributeValue> = {};
  if (!config) return attributes;
  const toolDefs = mapToolDefinitions(config.tools);
  if (toolDefs) {
    const serialized = jsonStringify(toolDefs);
    if (serialized !== undefined) attributes["gen_ai.tool.definitions"] = serialized;
  }
  const candidateCount = readNumber(config.candidateCount);
  if (candidateCount !== undefined) attributes["gen_ai.request.choice.count"] = candidateCount;
  if (config.toolConfig !== undefined)
    setJsonAttribute(attributes, "google_genai.request.tool_config", config.toolConfig);
  if (config.safetySettings !== undefined) {
    setJsonAttribute(attributes, "google_genai.request.safety_settings", config.safetySettings);
  }
  if (config.thinkingConfig !== undefined) {
    setJsonAttribute(attributes, "google_genai.request.thinking_config", config.thinkingConfig);
  }
  if (config.labels !== undefined)
    setJsonAttribute(attributes, "google_genai.request.labels", config.labels);
  const cachedContent = readString(config.cachedContent);
  if (cachedContent !== undefined)
    attributes["google_genai.request.cached_content"] = cachedContent;
  if (config.responseModalities !== undefined) {
    setJsonAttribute(
      attributes,
      "google_genai.request.response_modalities",
      config.responseModalities,
    );
  }
  return attributes;
}

function usageFromMetadata(usage: unknown): SpanFields["usage"] {
  const metadata = asRecord(usage);
  if (!metadata) return undefined;
  return compactUsage({
    inputTokens: readNumber(metadata.promptTokenCount),
    outputTokens: readNumber(metadata.candidatesTokenCount),
    totalTokens: readNumber(metadata.totalTokenCount),
    cacheReadInputTokens: readNumber(metadata.cachedContentTokenCount),
    reasoningOutputTokens: readNumber(metadata.thoughtsTokenCount),
  });
}

function candidateOutput(response: UnknownRecord): unknown[] | undefined {
  const candidates = asArray(response.candidates);
  if (!candidates || candidates.length === 0) return undefined;
  const output = candidates
    .map((candidate) => {
      const record = asRecord(candidate);
      const content = asRecord(record?.content);
      if (!content) return undefined;
      return {
        role: readString(content.role) ?? "model",
        parts: asArray(content.parts) ?? [],
      };
    })
    .filter((entry) => entry !== undefined);
  return output.length > 0 ? output : undefined;
}

function responseAttributes(response: UnknownRecord) {
  const attributes: Record<string, AttributeValue> = {};
  const candidates = asArray(response.candidates) ?? [];
  const promptFeedback = asRecord(response.promptFeedback);
  const blockReason = readString(promptFeedback?.blockReason);
  if (blockReason) attributes["google_genai.response.block_reason"] = blockReason;
  const blockReasonMessage = readString(promptFeedback?.blockReasonMessage);
  if (blockReasonMessage)
    attributes["google_genai.response.block_reason_message"] = blockReasonMessage;
  const promptSafetyRatings = asArray(promptFeedback?.safetyRatings);
  if (promptSafetyRatings && promptSafetyRatings.length > 0) {
    setJsonAttribute(
      attributes,
      "google_genai.response.prompt_safety_ratings",
      promptSafetyRatings,
    );
  }

  const finishReasons = candidates
    .map((candidate) => readString(asRecord(candidate)?.finishReason))
    .filter((reason): reason is string => reason !== undefined);
  if (finishReasons.length > 1) attributes["gen_ai.response.finish_reasons"] = finishReasons;

  const usageMetadata = asRecord(response.usageMetadata);
  const toolUsePromptTokens = readNumber(usageMetadata?.toolUsePromptTokenCount);
  if (toolUsePromptTokens !== undefined) {
    attributes["google_genai.usage.tool_use_prompt_tokens"] = toolUsePromptTokens;
  }

  const safetyEntries = candidates
    .map((candidate, index) => {
      const record = asRecord(candidate);
      const ratings = asArray(record?.safetyRatings);
      if (!ratings || ratings.length === 0) return undefined;
      return { candidateIndex: readNumber(record?.index) ?? index, ratings };
    })
    .filter((entry) => entry !== undefined);
  if (safetyEntries.length > 0) {
    setJsonAttribute(attributes, "google_genai.response.safety_ratings", safetyEntries);
  }

  const firstCandidate = asRecord(candidates[0]);
  if (firstCandidate?.groundingMetadata !== undefined) {
    setJsonAttribute(
      attributes,
      "google_genai.response.grounding_metadata",
      firstCandidate.groundingMetadata,
    );
  }
  if (firstCandidate?.urlContextMetadata !== undefined) {
    setJsonAttribute(
      attributes,
      "google_genai.response.url_context_metadata",
      firstCandidate.urlContextMetadata,
    );
  }

  const afcHistory = asArray(response.automaticFunctionCallingHistory);
  if (afcHistory && afcHistory.length > 0)
    attributes["google_genai.automatic_function_calling"] = true;

  return attributes;
}

function generateRequestFields(params: UnknownRecord): RequestMapping {
  const model = readString(params.model);
  const config = asRecord(params.config);
  const attributes = requestAttributesFromConfig(config);
  const fields: RequestMapping["fields"] = {
    type: "generation",
    model,
    input: normalizeContentInput(params.contents),
    systemInstructions: config?.systemInstruction,
    temperature: readNumber(config?.temperature),
    topP: readNumber(config?.topP),
    topK: readNumber(config?.topK),
    maxTokens: readNumber(config?.maxOutputTokens),
    stopSequences: stopSequences(config?.stopSequences),
    seed: readNumber(config?.seed),
    frequencyPenalty: readNumber(config?.frequencyPenalty),
    presencePenalty: readNumber(config?.presencePenalty),
    outputType: outputTypeFromConfig(config),
  };
  if (Object.keys(attributes).length > 0) fields.attributes = attributes;
  return { name: `chat ${model ?? "unknown"}`, fields };
}

function generateResponseFields(response: unknown): SpanFields {
  const record = asRecord(response) ?? {};
  const candidates = asArray(record.candidates) ?? [];
  const finishReasons = candidates
    .map((candidate) => readString(asRecord(candidate)?.finishReason))
    .filter((reason): reason is string => reason !== undefined);
  const attributes = responseAttributes(record);
  const afcHistory = asArray(record.automaticFunctionCallingHistory);
  const fields: SpanFields = {
    responseModel: readString(record.modelVersion),
    responseId: readString(record.responseId),
    finishReason: finishReasons[0],
    output: candidateOutput(record),
    usage: usageFromMetadata(record.usageMetadata),
  };
  if (Object.keys(attributes).length > 0) fields.attributes = attributes;
  if (afcHistory && afcHistory.length > 0) fields.input = afcHistory;
  if (readString(asRecord(record.promptFeedback)?.blockReason)) {
    delete fields.output;
    delete fields.finishReason;
  }
  return fields;
}

function embedRequestFields(params: UnknownRecord): RequestMapping {
  const model = readString(params.model);
  const config = asRecord(params.config);
  const attributes: Record<string, AttributeValue> = {};
  const taskType = readString(config?.taskType);
  if (taskType) attributes["google_genai.request.task_type"] = taskType;
  const outputDimensionality = readNumber(config?.outputDimensionality);
  if (outputDimensionality !== undefined) {
    attributes["google_genai.request.output_dimensionality"] = outputDimensionality;
  }
  const fields: RequestMapping["fields"] = { type: "embedding", model, input: params.contents };
  if (Object.keys(attributes).length > 0) fields.attributes = attributes;
  return { name: `embeddings ${model ?? "unknown"}`, fields };
}

function embedResponseFields(response: unknown): SpanFields {
  const record = asRecord(response) ?? {};
  const embeddings = asArray(record.embeddings) ?? [];
  const attributes: Record<string, AttributeValue> = {};
  if (embeddings.length > 0)
    attributes["google_genai.response.embedding_count"] = embeddings.length;
  const firstValues = asArray(asRecord(embeddings[0])?.values);
  if (firstValues) attributes["google_genai.response.embedding_dimensions"] = firstValues.length;
  let inputTokens = 0;
  let sawTokens = false;
  for (const embedding of embeddings) {
    const tokenCount = readNumber(asRecord(asRecord(embedding)?.statistics)?.tokenCount);
    if (tokenCount !== undefined) {
      sawTokens = true;
      inputTokens += tokenCount;
    }
  }
  const metadata = asRecord(record.metadata);
  const billableCharacters = readNumber(metadata?.billableCharacterCount);
  if (billableCharacters !== undefined) {
    attributes["google_genai.usage.billable_characters"] = billableCharacters;
  }
  const fields: SpanFields = { usage: sawTokens ? compactUsage({ inputTokens }) : undefined };
  if (Object.keys(attributes).length > 0) fields.attributes = attributes;
  return fields;
}

function isWrapped(fn: unknown): fn is WrappedFunction {
  return typeof fn === "function" && (fn as WrappedFunction)[WRAPPED] === true;
}

function markWrapped<T extends WrappedFunction>(
  fn: T,
  original: (...args: unknown[]) => unknown,
): T {
  Object.defineProperty(fn, WRAPPED, { value: true });
  Object.defineProperty(fn, ORIGINAL, { value: original });
  return fn;
}

function endOnce(span: SpanHandle): (fields?: SpanFields) => void {
  let ended = false;
  return (fields?: SpanFields) => {
    if (ended) return;
    ended = true;
    span.end(fields);
  };
}

interface RequestFallback {
  namePrefix: string;
  type: "generation" | "embedding";
}

function safeRequestMapping(
  params: UnknownRecord,
  mapRequest: (params: UnknownRecord) => RequestMapping,
  fallback: RequestFallback,
): RequestMapping {
  try {
    return mapRequest(params);
  } catch {
    let model: string | undefined;
    try {
      model = readString(params.model);
    } catch {
      model = undefined;
    }
    return {
      name: `${fallback.namePrefix} ${model ?? "unknown"}`,
      fields: { type: fallback.type, model },
    };
  }
}

function safeResponseFields(
  mapResponse: (response: unknown) => SpanFields,
  response: unknown,
): SpanFields {
  try {
    return mapResponse(response);
  } catch {
    return {};
  }
}

function shouldAggregateAfcUsage(params: UnknownRecord): boolean {
  try {
    const config = asRecord(params.config);
    const tools = asArray(config?.tools) ?? [];
    if (!tools.some((tool) => typeof asRecord(tool)?.callTool === "function")) return false;
    const automaticFunctionCalling = asRecord(config?.automaticFunctionCalling);
    return (
      automaticFunctionCalling?.disable !== true &&
      readNumber(automaticFunctionCalling?.maximumRemoteCalls) !== 0
    );
  } catch {
    return false;
  }
}

function installInternalUsageCollector(models: UnknownRecord | undefined): void {
  if (!models || usageCollectorModels.has(models)) return;
  const original = models.generateContentInternal;
  if (typeof original !== "function") return;
  models.generateContentInternal = function (this: unknown, ...args: unknown[]) {
    const result = original.apply(this, args);
    const recordUsage = (response: unknown) => {
      try {
        const store = afcUsageStore?.getStore();
        const usageMetadata = asRecord(response)?.usageMetadata;
        if (store) {
          store.usage = sumUsage(store.usage, usageFromMetadata(asRecord(usageMetadata)));
          const toolUsePromptTokens = readNumber(asRecord(usageMetadata)?.toolUsePromptTokenCount);
          if (toolUsePromptTokens !== undefined) {
            store.toolUsePromptTokens =
              (readNumber(store.toolUsePromptTokens) ?? 0) + toolUsePromptTokens;
          }
        }
      } catch {
        // Telemetry-only AFC usage collection must fail open.
      }
      return response;
    };
    return isThenable(result) ? result.then(recordUsage) : recordUsage(result);
  };
  usageCollectorModels.add(models);
}

function wrapUnary(
  original: (...args: unknown[]) => unknown,
  mapRequest: (params: UnknownRecord) => RequestMapping,
  mapResponse: (response: unknown) => SpanFields,
  provider: string,
  fallback: RequestFallback,
  models?: UnknownRecord,
): WrappedFunction {
  if (isWrapped(original)) return original;
  const wrapped = function (this: unknown, ...args: unknown[]) {
    const params = asRecord(args[0]) ?? {};
    const request = safeRequestMapping(params, mapRequest, fallback);
    const span = startSpan(request.name, { ...request.fields, provider });
    const end = endOnce(span);
    const usageStore = afcUsageStore;
    const internalUsage: { usage?: SpanFields["usage"]; toolUsePromptTokens?: number } | undefined =
      usageStore && shouldAggregateAfcUsage(params) ? {} : undefined;
    if (internalUsage) installInternalUsageCollector(models);
    const finishResponse = (response: unknown) => {
      const fields = safeResponseFields(mapResponse, response);
      if (internalUsage?.usage) fields.usage = internalUsage.usage;
      const toolUsePromptTokens = readNumber(internalUsage?.toolUsePromptTokens);
      if (toolUsePromptTokens !== undefined) {
        fields.attributes = {
          ...fields.attributes,
          "google_genai.usage.tool_use_prompt_tokens": toolUsePromptTokens,
        };
      }
      end(fields);
      return response;
    };
    try {
      const result =
        usageStore && internalUsage
          ? usageStore.run(internalUsage, () => original.call(this, ...args))
          : original.call(this, ...args);
      if (isThenable(result)) {
        return result.then(
          (response) => finishResponse(response),
          (error) => {
            end({ error });
            throw error;
          },
        );
      }
      return finishResponse(result);
    } catch (error) {
      end({ error });
      throw error;
    }
  };
  return markWrapped(wrapped as WrappedFunction, original);
}

interface CandidateAggregate {
  role: string;
  parts: UnknownRecord[];
  finishReason?: string;
}

function appendPart(parts: UnknownRecord[], incoming: UnknownRecord): void {
  const last = parts[parts.length - 1];
  if (
    last &&
    typeof last.text === "string" &&
    typeof incoming.text === "string" &&
    last.thought === incoming.thought
  ) {
    last.text += incoming.text;
    return;
  }
  parts.push({ ...incoming });
}

function sumUsage(
  left: SpanFields["usage"] | undefined,
  right: SpanFields["usage"] | undefined,
): SpanFields["usage"] {
  if (!left) return right;
  if (!right) return left;
  const add = (a: number | undefined, b: number | undefined) =>
    a === undefined && b === undefined ? undefined : (a ?? 0) + (b ?? 0);
  return compactUsage({
    inputTokens: add(left.inputTokens, right.inputTokens),
    outputTokens: add(left.outputTokens, right.outputTokens),
    totalTokens: add(left.totalTokens, right.totalTokens),
    cacheReadInputTokens: add(left.cacheReadInputTokens, right.cacheReadInputTokens),
    reasoningOutputTokens: add(left.reasoningOutputTokens, right.reasoningOutputTokens),
  });
}

function copyContent(content: UnknownRecord) {
  return {
    role: readString(content.role) ?? "model",
    parts: (asArray(content.parts) ?? []).flatMap((part) => {
      const record = asRecord(part);
      return record ? [{ ...record }] : [];
    }),
  };
}

function normalizeContentInput(input: unknown): unknown[] {
  if (input === undefined) return [];
  if (typeof input === "string") return [{ role: "user", parts: [{ text: input }] }];
  const items = asArray(input);
  if (items) {
    if (items.some((item) => readString(asRecord(item)?.role) || asArray(asRecord(item)?.parts))) {
      return [...items];
    }
    return [
      {
        role: "user",
        parts: items.flatMap((item) => {
          if (typeof item === "string") return [{ text: item }];
          const record = asRecord(item);
          return record ? [{ ...record }] : [];
        }),
      },
    ];
  }
  const record = asRecord(input);
  if (record && !readString(record.role) && !asArray(record.parts)) {
    return [{ role: "user", parts: [{ ...record }] }];
  }
  return [input];
}

function hasFunctionResponse(content: UnknownRecord | undefined): boolean {
  if (readString(content?.role) !== "user") return false;
  return (asArray(content?.parts) ?? []).some(
    (part) => asRecord(part)?.functionResponse !== undefined,
  );
}

class StreamAggregator {
  private candidates = new Map<number, CandidateAggregate>();
  private usage?: SpanFields["usage"];
  private committedUsage?: SpanFields["usage"];
  private attributes: Record<string, AttributeValue> = {};
  private priorOutput: unknown[] = [];
  private afcHistory?: unknown[];
  private responseId?: string;
  private responseModel?: string;

  constructor(private readonly requestInput: unknown) {}

  private currentOutput(): unknown[] {
    return [...this.candidates.entries()]
      .sort(([left], [right]) => left - right)
      .map(([, state]) => ({ role: state.role, parts: state.parts }));
  }

  private finishReasons(): string[] {
    return [...this.candidates.entries()]
      .sort(([left], [right]) => left - right)
      .map(([, state]) => state.finishReason)
      .filter((reason): reason is string => reason !== undefined);
  }

  private ensureAfcHistory(): unknown[] {
    if (!this.afcHistory) {
      this.afcHistory = normalizeContentInput(this.requestInput);
    }
    return this.afcHistory;
  }

  private foldTurn(toAfcHistory: boolean): void {
    const output = this.currentOutput();
    if (output.length > 0) {
      if (toAfcHistory) this.ensureAfcHistory().push(...output);
      else this.priorOutput.push(...output);
    }
    this.committedUsage = sumUsage(this.committedUsage, this.usage);
    this.candidates.clear();
    this.usage = undefined;
  }

  recordChunk(chunk: unknown): void {
    const record = asRecord(chunk) ?? {};
    const nextResponseId = readString(record.responseId);
    if (nextResponseId && this.responseId && nextResponseId !== this.responseId) {
      this.foldTurn(false);
    }
    if (nextResponseId) this.responseId = nextResponseId;
    const nextResponseModel = readString(record.modelVersion);
    if (nextResponseModel) this.responseModel = nextResponseModel;

    for (const candidate of asArray(record.candidates) ?? []) {
      const candidateRecord = asRecord(candidate) ?? {};
      const content = asRecord(candidateRecord.content);
      if (content && hasFunctionResponse(content)) {
        this.foldTurn(true);
        this.ensureAfcHistory().push(copyContent(content));
        continue;
      }
      const index = readNumber(candidateRecord.index) ?? 0;
      let state = this.candidates.get(index);
      if (!state) {
        state = { role: "model", parts: [] };
        this.candidates.set(index, state);
      }
      if (content) {
        const role = readString(content.role);
        if (role) state.role = role;
        for (const part of asArray(content.parts) ?? []) {
          const partRecord = asRecord(part);
          if (partRecord) appendPart(state.parts, partRecord);
        }
      }
      const finishReason = readString(candidateRecord.finishReason);
      if (finishReason) state.finishReason = finishReason;
    }
    const chunkUsage = usageFromMetadata(record.usageMetadata);
    if (chunkUsage) this.usage = chunkUsage;
    const afcHistory = asArray(record.automaticFunctionCallingHistory);
    if (afcHistory && afcHistory.length > 0) this.afcHistory = [...afcHistory];
    const chunkAttributes = responseAttributes(record);
    Object.assign(this.attributes, chunkAttributes);
  }

  finish(): SpanFields {
    const output =
      this.afcHistory && this.afcHistory.length > 0
        ? this.currentOutput()
        : [...this.priorOutput, ...this.currentOutput()];
    const finishReasons = this.finishReasons();
    const fields: SpanFields = {
      responseId: this.responseId,
      responseModel: this.responseModel,
      output: output.length > 0 ? output : undefined,
      usage: sumUsage(this.committedUsage, this.usage),
      finishReason: finishReasons[0],
    };
    const attributes = { ...this.attributes };
    if (this.afcHistory && this.afcHistory.length > 0) {
      fields.input = this.afcHistory;
      attributes["google_genai.automatic_function_calling"] = true;
    }
    if (finishReasons.length > 1) attributes["gen_ai.response.finish_reasons"] = finishReasons;
    if (Object.keys(attributes).length > 0) fields.attributes = attributes;
    if (attributes["google_genai.response.block_reason"]) {
      delete fields.output;
      delete fields.finishReason;
    }
    return fields;
  }
}

function createObservedStream(
  source: AsyncIterable<unknown>,
  span: SpanHandle,
  startedAt: number,
  end: (fields?: SpanFields) => void,
  requestInput: unknown,
): AsyncGenerator<unknown> {
  const iterator = source[Symbol.asyncIterator]();
  const aggregator = new StreamAggregator(requestInput);
  let sawFirst = false;
  let ended = false;

  const recordChunk = (chunk: unknown): void => {
    if (!sawFirst) {
      sawFirst = true;
      const record = asRecord(chunk) ?? {};
      span.update({
        timeToFirstChunkMs: Date.now() - startedAt,
        responseId: readString(record.responseId),
        responseModel: readString(record.modelVersion),
      });
    }
    try {
      aggregator.recordChunk(chunk);
    } catch {
      // Telemetry mapping must not affect stream delivery.
    }
  };

  const finish = (error?: unknown): void => {
    if (ended) return;
    ended = true;
    let finalFields: SpanFields = {};
    try {
      finalFields = aggregator.finish();
    } catch {
      finalFields = {};
    }
    if (error !== undefined) finalFields.error = error;
    end(finalFields);
  };

  return {
    [Symbol.asyncIterator]() {
      return this;
    },
    async next(value?: unknown) {
      try {
        const step = await iterator.next(value);
        if (step.done) finish();
        else recordChunk(step.value);
        return step;
      } catch (error) {
        finish(error);
        throw error;
      }
    },
    async return(value?: unknown) {
      try {
        if (iterator.return) {
          const step = await iterator.return(value);
          if (!step.done) recordChunk(step.value);
          else finish();
          return step;
        }
        finish();
        return { done: true, value };
      } catch (error) {
        finish(error);
        throw error;
      }
    },
    async throw(error?: unknown) {
      if (!iterator.throw) {
        finish(error);
        throw error;
      }
      try {
        const step = await iterator.throw(error);
        if (step.done) finish();
        else recordChunk(step.value);
        return step;
      } catch (thrown) {
        finish(thrown);
        throw thrown;
      }
    },
    async [Symbol.asyncDispose]() {
      await this.return(undefined);
    },
  };
}

function wrapStream(
  original: (...args: unknown[]) => unknown,
  mapRequest: (params: UnknownRecord) => RequestMapping,
  provider: string,
  fallback: RequestFallback,
): WrappedFunction {
  if (isWrapped(original)) return original;
  const wrapped = function (this: unknown, ...args: unknown[]) {
    const params = asRecord(args[0]) ?? {};
    const request = safeRequestMapping(params, mapRequest, fallback);
    const span = startSpan(request.name, { ...request.fields, provider });
    const end = endOnce(span);
    const startedAt = Date.now();
    try {
      const result = original.call(this, ...args);
      return Promise.resolve(result).then(
        (generator) =>
          createObservedStream(
            generator as AsyncIterable<unknown>,
            span,
            startedAt,
            end,
            request.fields.input,
          ),
        (error) => {
          end({ error });
          throw error;
        },
      );
    } catch (error) {
      end({ error });
      throw error;
    }
  };
  return markWrapped(wrapped as WrappedFunction, original);
}

function patchModelsMethod(
  models: UnknownRecord,
  key: string,
  mapRequest: (params: UnknownRecord) => RequestMapping,
  mapResponse: (response: unknown) => SpanFields,
  provider: string,
  fallback: RequestFallback,
  streaming = false,
): void {
  const current = models[key];
  if (isWrapped(current) && Object.prototype.hasOwnProperty.call(models, key)) return;
  const original = isWrapped(current) ? current[ORIGINAL] : current;
  if (typeof original !== "function") return;
  const bound = original.bind(models) as (...args: unknown[]) => unknown;
  models[key] = streaming
    ? wrapStream(bound, mapRequest, provider, fallback)
    : wrapUnary(
        bound,
        mapRequest,
        mapResponse,
        provider,
        fallback,
        key === "generateContent" ? models : undefined,
      );
}

export function wrapGoogleGenAI<T extends GoogleGenAIClientLike>(client: T): T {
  if (wrappedClients.has(client)) return client;
  const provider = client.vertexai === true ? "gcp.vertex_ai" : "gcp.gemini";
  const models = client.models as UnknownRecord;
  patchModelsMethod(
    models,
    "generateContent",
    generateRequestFields,
    generateResponseFields,
    provider,
    { namePrefix: "chat", type: "generation" },
  );
  patchModelsMethod(
    models,
    "generateContentStream",
    generateRequestFields,
    generateResponseFields,
    provider,
    { namePrefix: "chat", type: "generation" },
    true,
  );
  patchModelsMethod(models, "embedContent", embedRequestFields, embedResponseFields, provider, {
    namePrefix: "embeddings",
    type: "embedding",
  });
  wrappedClients.add(client);
  return client;
}
