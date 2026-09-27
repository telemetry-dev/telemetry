/**
 * Google GenAI instrumentation for telemetry.dev.
 *
 * TypeScript has no global instrumentation entry point: `@google/genai` implements
 * `generateContent`, `generateContentStream`, and `embedContent` as arrow-function
 * instance fields on `Models`, so prototype patching cannot intercept them. Use
 * `wrapGoogleGenAI(client)` on each client you want traced.
 */

import type { AsyncLocalStorage } from "node:async_hooks";
import {
  boundedCapture,
  boundedCaptureDetails,
  captureEnabled,
  startSpan,
  type SpanFields,
  type SpanHandle,
} from "@telemetry-dev/sdk";

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

type WrappedFunction = ((...args: unknown[]) => unknown) & {
  [WRAPPED]?: true;
  [ORIGINAL]?: (...args: unknown[]) => unknown;
};

interface UnknownRecord {
  // oxlint-disable-next-line anti-slop/no-unsafe-dictionary-type -- Google returns versioned external payloads; every consumed field is narrowed after this boundary.
  [key: string]: unknown;
}

interface RequestMapping {
  name: string;
  fields: SpanFields & { type: "generation" | "embedding" };
}

interface VideoOperationTracker {
  operations: WeakSet<object>;
}

export interface GoogleGenAIClientLike {
  models: object;
  operations?: object;
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

const skipBinaryCapture = (
  key: string,
  _item: unknown,
  _parent: unknown,
  path: readonly string[],
) => {
  const container = path.at(-1);
  const inContentPart = path.length === 1 || path.at(-2) === "parts";

  return (
    inContentPart &&
    ((key === "data" && (container === "inlineData" || container === "inline_data")) ||
      (key === "imageBytes" && container === "image") ||
      (key === "videoBytes" && container === "video"))
  );
};

function boundedTelemetryCapture(value: unknown) {
  return boundedCapture(value, { skip: skipBinaryCapture });
}

function boundedTelemetryCaptureDetails(value: unknown) {
  return boundedCaptureDetails(value, { skip: skipBinaryCapture });
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

  const result: NonNullable<SpanFields["usage"]> = {
    inputTokens: readNumber(metadata.promptTokenCount),
    outputTokens: readNumber(metadata.candidatesTokenCount),
    totalTokens: readNumber(metadata.totalTokenCount),
    cacheReadInputTokens: readNumber(metadata.cachedContentTokenCount),
    reasoningOutputTokens: readNumber(metadata.thoughtsTokenCount),
  };

  mapModalityTokens(metadata.promptTokensDetails, result, "input");
  mapModalityTokens(metadata.candidatesTokensDetails, result, "output");
  mapModalityTokens(metadata.cacheTokensDetails, result, "cache");

  return compactUsage(result);
}

function mapModalityTokens(
  details: unknown,
  usage: NonNullable<SpanFields["usage"]>,
  kind: "input" | "output" | "cache",
): void {
  for (const detail of asArray(details) ?? []) {
    const record = asRecord(detail);
    const count = readNumber(record?.tokenCount);
    const modality = readString(record?.modality)?.toUpperCase();

    if (count === undefined || !modality || !["TEXT", "IMAGE", "AUDIO"].includes(modality))
      continue;
    const prefix = modality.toLowerCase() as "text" | "image" | "audio";

    const suffix = {
      cache: "CacheReadInputTokens",
      input: "InputTokens",
      output: "OutputTokens",
    }[kind];

    const key = `${prefix}${suffix}`;

    const typedKey = key as keyof NonNullable<SpanFields["usage"]>;
    usage[typedKey] = (usage[typedKey] ?? 0) + count;
  }
}

function candidateOutput(response: UnknownRecord) {
  if (!captureEnabled("output")) return { value: undefined, truncated: false };
  const capture = boundedTelemetryCapture(response.candidates);
  const candidates = asArray(capture.value);

  if (!candidates || candidates.length === 0)
    return { value: undefined, truncated: capture.truncated };

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

  return { value: output.length > 0 ? output : undefined, truncated: capture.truncated };
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
  const input = captureEnabled("input") ? normalizeContentInput(params.contents) : undefined;

  const systemInstructions = captureEnabled("input")
    ? boundedTelemetryCapture(config?.systemInstruction)
    : undefined;

  const fields: RequestMapping["fields"] = {
    type: "generation",
    model,
    input: input?.value,
    systemInstructions: systemInstructions?.value,
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

  if (input?.truncated || systemInstructions?.truncated)
    attributes["telemetry.dev.capture.truncated"] = true;

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
  const output = candidateOutput(record);

  const fields: SpanFields = {
    responseModel: readString(record.modelVersion),
    responseId: readString(record.responseId),
    finishReason: finishReasons[0],
    output: output.value,
    usage: usageFromMetadata(record.usageMetadata),
  };

  if (output.truncated) attributes["telemetry.dev.capture.truncated"] = true;

  if (captureEnabled("input") && afcHistory && afcHistory.length > 0) {
    const capture = boundedTelemetryCapture(afcHistory);
    fields.input = capture.value;

    if (capture.truncated) attributes["telemetry.dev.capture.truncated"] = true;
  }

  if (Object.keys(attributes).length > 0) fields.attributes = attributes;

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

const MEDIA_CONFIG_KEYS = [
  "numberOfImages",
  "numberOfVideos",
  "aspectRatio",
  "guidanceScale",
  "seed",
  "safetyFilterLevel",
  "personGeneration",
  "language",
  "outputMimeType",
  "outputCompressionQuality",
  "addWatermark",
  "editMode",
  "baseSteps",
  "imageSize",
  "enhancePrompt",
  "fps",
  "durationSeconds",
  "resolution",
  "generateAudio",
  "compressionQuality",
  "resizeMode",
  "enhanceInputImage",
  "imagePreservationFactor",
] as const;

function mediaReference(value: unknown): UnknownRecord | undefined {
  const record = asRecord(value);

  if (!record) return undefined;
  const result: UnknownRecord = {};

  for (const key of ["gcsUri", "uri", "mimeType"] as const) {
    const item = readString(record[key]);

    if (item !== undefined) result[key] = item;
  }

  return Object.keys(result).length > 0 ? result : undefined;
}

function mediaRequestFields(
  params: UnknownRecord,
  kind: string,
  outputType: string,
): RequestMapping {
  const model = readString(params.model);
  const config = asRecord(params.config);

  const attributes: NonNullable<SpanFields["attributes"]> = {
    "gen_ai.operation.name": "generate_content",
    "google_genai.operation.type": kind,
  };

  const safeConfig: UnknownRecord = {};

  for (const key of MEDIA_CONFIG_KEYS) {
    const value = config?.[key];

    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      safeConfig[key] = value;
    }
  }

  if (Object.keys(safeConfig).length > 0) {
    setJsonAttribute(attributes, "google_genai.request.config", safeConfig);
  }

  const referenceCount = asArray(params.referenceImages)?.length;

  if (referenceCount !== undefined)
    attributes["google_genai.request.reference_image_count"] = referenceCount;
  const upscaleFactor = readString(params.upscaleFactor);

  if (upscaleFactor) attributes["google_genai.request.upscale_factor"] = upscaleFactor;
  const prompt = readString(params.prompt) ?? readString(asRecord(params.source)?.prompt);
  const source = asRecord(params.source);

  const inputMedia = [
    mediaReference(params.image),
    mediaReference(params.video),
    mediaReference(source?.image),
    mediaReference(source?.video),
  ].filter((value): value is UnknownRecord => value !== undefined);

  const contentInput: UnknownRecord = {};

  if (prompt) contentInput.prompt = prompt;
  const negativePrompt = readString(config?.negativePrompt);

  if (negativePrompt) contentInput.negativePrompt = negativePrompt;

  if (inputMedia.length > 0) contentInput.media = inputMedia;

  if (config?.labels !== undefined) contentInput.labels = config.labels;

  for (const key of ["outputGcsUri", "pubsubTopic"] as const) {
    const value = readString(config?.[key]);

    if (value !== undefined) contentInput[key] = value;
  }

  const input =
    captureEnabled("input") && Object.keys(contentInput).length > 0
      ? boundedTelemetryCapture(contentInput)
      : undefined;

  if (input?.truncated) attributes["telemetry.dev.capture.truncated"] = true;

  return {
    name: `generate_content ${model ?? "unknown"}`,
    fields: {
      type: "generation",
      model,
      input: input?.value,
      outputType,
      attributes,
    },
  };
}

function mediaResponseFields(response: unknown, outputType: "image" | "video"): SpanFields {
  const record = asRecord(response) ?? {};
  const operationResponse = asRecord(record.response);
  const responseRecord = operationResponse ?? record;

  const items =
    outputType === "image"
      ? (asArray(responseRecord.generatedImages) ?? [])
      : (asArray(responseRecord.generatedVideos) ?? []);

  const attributes: Record<string, AttributeValue> = {};

  attributes[`google_genai.response.${outputType}_count`] = items.length;

  const uris = items.flatMap((item) => {
    const media = asRecord(asRecord(item)?.[outputType]);
    const uri = readString(media?.gcsUri) ?? readString(media?.uri);

    return uri ? [uri] : [];
  });

  const operationName = readString(record.name);

  if (operationName) attributes["google_genai.response.operation_name"] = operationName;

  if (typeof record.done === "boolean")
    attributes["google_genai.response.operation_done"] = record.done;

  return {
    responseId: readString(record.responseId) ?? operationName,
    output: uris.length > 0 ? uris.map((uri) => ({ type: outputType, uri })) : undefined,
    attributes,
    error: record.error,
  };
}

function imageResponseFields(response: unknown): SpanFields {
  return mediaResponseFields(response, "image");
}

function videoResponseFields(response: unknown, tracker: VideoOperationTracker): SpanFields {
  const record = asRecord(response);
  const done = record?.done === true;

  if (record && !done) tracker.operations.add(record);

  return mediaResponseFields(response, "video");
}

function isWrapped<T>(fn: T): fn is T & WrappedFunction {
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
  captured: boolean;
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

function serializedByteLength(value: unknown): number {
  return new TextEncoder().encode(JSON.stringify(value)).byteLength;
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
    textInputTokens: add(left.textInputTokens, right.textInputTokens),
    textOutputTokens: add(left.textOutputTokens, right.textOutputTokens),
    textCacheReadInputTokens: add(left.textCacheReadInputTokens, right.textCacheReadInputTokens),
    imageInputTokens: add(left.imageInputTokens, right.imageInputTokens),
    imageOutputTokens: add(left.imageOutputTokens, right.imageOutputTokens),
    imageCacheReadInputTokens: add(left.imageCacheReadInputTokens, right.imageCacheReadInputTokens),
    audioInputTokens: add(left.audioInputTokens, right.audioInputTokens),
    audioOutputTokens: add(left.audioOutputTokens, right.audioOutputTokens),
    audioCacheReadInputTokens: add(left.audioCacheReadInputTokens, right.audioCacheReadInputTokens),
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

function normalizeContentInput(input: unknown) {
  const capture = boundedTelemetryCapture(input);
  const bounded = capture.value;

  if (bounded === undefined) return { value: [], truncated: capture.truncated };

  if (typeof bounded === "string")
    return { value: [{ role: "user", parts: [{ text: bounded }] }], truncated: capture.truncated };
  const items = asArray(bounded);

  if (items) {
    if (items.some((item) => readString(asRecord(item)?.role) || asArray(asRecord(item)?.parts))) {
      return { value: items, truncated: capture.truncated };
    }

    return {
      value: [
        {
          role: "user",
          parts: items.flatMap((item) => {
            if (typeof item === "string") return [{ text: item }];
            const record = asRecord(item);

            return record ? [record] : [];
          }),
        },
      ],
      truncated: capture.truncated,
    };
  }

  const record = asRecord(bounded);

  if (record && !readString(record.role) && !asArray(record.parts)) {
    return { value: [{ role: "user", parts: [record] }], truncated: capture.truncated };
  }

  return { value: [bounded], truncated: capture.truncated };
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
  private automaticFunctionCalling = false;
  private responseId?: string;
  private responseModel?: string;
  private outputCaptureBytes = 2;
  private outputCaptureItems = 0;
  private outputCaptureEntries = 0;
  private outputCaptureTruncated = false;
  private afcCaptureTruncated = false;

  constructor(private readonly requestInput: unknown) {}

  private currentCandidates(): unknown[] {
    return [...this.candidates.entries()]
      .sort(([left], [right]) => left - right)
      .filter(([, state]) => state.captured)
      .map(([, state]) => ({ role: state.role, parts: state.parts }));
  }

  private reserveOutput(byteCount: number, itemCount: number): boolean {
    if (
      this.outputCaptureBytes + byteCount > 48 * 1024 ||
      this.outputCaptureItems + itemCount > 1_000
    ) {
      this.outputCaptureTruncated = true;

      return false;
    }

    this.outputCaptureBytes += byteCount;
    this.outputCaptureItems += itemCount;

    return true;
  }

  private resetOutputCapture(): void {
    this.priorOutput = [];
    this.outputCaptureBytes = 2;
    this.outputCaptureItems = 0;
    this.outputCaptureEntries = 0;
    this.outputCaptureTruncated = false;
  }

  private currentOutput(forInput = false): unknown[] {
    if (!captureEnabled(forInput ? "input" : "output")) return [];

    return this.currentCandidates();
  }

  private finishReasons(): string[] {
    return [...this.candidates.entries()]
      .sort(([left], [right]) => left - right)
      .map(([, state]) => state.finishReason)
      .filter((reason): reason is string => reason !== undefined);
  }

  private ensureAfcHistory(): unknown[] {
    this.automaticFunctionCalling = true;

    if (!captureEnabled("input")) return [];

    if (!this.afcHistory) {
      this.afcHistory = [...(asArray(this.requestInput) ?? [])];
    }

    return this.afcHistory;
  }

  private appendAfcHistory(...entries: unknown[]): void {
    if (!captureEnabled("input")) return;

    for (const entry of entries) {
      const capture = boundedTelemetryCapture([...this.ensureAfcHistory(), entry]);

      if (capture.truncated) {
        this.afcCaptureTruncated = true;
        continue;
      }

      this.afcHistory = asArray(capture.value) ?? [];
    }

    if (this.outputCaptureTruncated) this.afcCaptureTruncated = true;
  }

  private replaceAfcHistory(history: unknown[]): void {
    const capture = boundedTelemetryCapture(history);
    this.afcHistory = asArray(capture.value) ?? [];
    this.afcCaptureTruncated = capture.truncated;
  }

  private foldTurn(toAfcHistory: boolean): void {
    const output = this.currentOutput(toAfcHistory);

    if (output.length > 0) {
      if (toAfcHistory) this.appendAfcHistory(...output);
      else this.priorOutput.push(...output);
    }

    if (toAfcHistory) this.resetOutputCapture();

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

    const afcHistory = asArray(record.automaticFunctionCallingHistory);

    if (afcHistory && afcHistory.length > 0) {
      this.automaticFunctionCalling = true;

      if (this.candidates.size === 0) this.resetOutputCapture();

      if (captureEnabled("input")) this.replaceAfcHistory(afcHistory);
    }

    const capturesContent = captureEnabled("input") || captureEnabled("output");
    const candidates = asArray(record.candidates) ?? [];

    for (const candidate of candidates) {
      const candidateRecord = asRecord(candidate) ?? {};
      const content = capturesContent ? asRecord(candidateRecord.content) : undefined;

      if (content && hasFunctionResponse(content)) {
        const capture = boundedTelemetryCapture(copyContent(content));

        if (capture.truncated) this.outputCaptureTruncated = true;
        const capturedContent = asRecord(capture.value);
        this.foldTurn(true);

        if (capturedContent) this.appendAfcHistory(copyContent(capturedContent));
        continue;
      }

      const index = readNumber(candidateRecord.index) ?? 0;
      let state = this.candidates.get(index);

      if (!state) {
        const role = "model";

        const byteCount =
          (this.outputCaptureEntries > 0 ? 1 : 0) + serializedByteLength({ role, parts: [] });

        const captured = capturesContent && this.reserveOutput(byteCount, 1);
        state = { role, parts: [], captured };

        if (captured) this.outputCaptureEntries += 1;
        this.candidates.set(index, state);
      }

      if (content) {
        const roleCapture = boundedTelemetryCaptureDetails(content.role);

        if (roleCapture.truncated) this.outputCaptureTruncated = true;
        const role = readString(roleCapture.value);

        if (role && role !== state.role) {
          const byteCount = roleCapture.bytes - serializedByteLength(state.role);

          if (!state.captured || this.reserveOutput(byteCount, 0)) state.role = role;
        }

        for (const part of asArray(content.parts) ?? []) {
          const capture = boundedTelemetryCaptureDetails(part);

          if (capture.truncated) this.outputCaptureTruncated = true;
          const partRecord = asRecord(capture.value);

          if (!partRecord || !state.captured) continue;
          const last = state.parts[state.parts.length - 1];

          const mergesText =
            last &&
            typeof last.text === "string" &&
            typeof partRecord.text === "string" &&
            last.thought === partRecord.thought;

          const byteCount = mergesText
            ? serializedByteLength(partRecord.text) - 2
            : (state.parts.length > 0 ? 1 : 0) + capture.bytes;

          if (this.reserveOutput(byteCount, 1)) appendPart(state.parts, partRecord);
        }
      }

      const finishReason = readString(candidateRecord.finishReason);

      if (finishReason) state.finishReason = finishReason;
    }

    const chunkUsage = usageFromMetadata(record.usageMetadata);

    if (chunkUsage) this.usage = chunkUsage;

    const chunkAttributes = responseAttributes(record);
    Object.assign(this.attributes, chunkAttributes);
  }

  finish(): SpanFields {
    const output = this.automaticFunctionCalling
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

    if (this.automaticFunctionCalling) {
      if (this.afcHistory && this.afcHistory.length > 0) fields.input = this.afcHistory;
      attributes["google_genai.automatic_function_calling"] = true;
    }

    if (finishReasons.length > 1) attributes["gen_ai.response.finish_reasons"] = finishReasons;

    if (this.afcCaptureTruncated || (captureEnabled("output") && this.outputCaptureTruncated)) {
      attributes["telemetry.dev.capture.truncated"] = true;
    }

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

  const recordChunk = (chunk: unknown, receivedAt: number): void => {
    if (chunkHasOutput(chunk)) span.recordOutputChunk?.(receivedAt);

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
        const receivedAt = performance.now();

        if (step.done) finish();
        else recordChunk(step.value, receivedAt);

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
          const receivedAt = performance.now();

          if (!step.done) recordChunk(step.value, receivedAt);
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
        const receivedAt = performance.now();

        if (step.done) finish();
        else recordChunk(step.value, receivedAt);

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

function chunkHasOutput(chunk: unknown): boolean {
  return (asArray(asRecord(chunk)?.candidates) ?? []).some((candidate) =>
    (asArray(asRecord(asRecord(candidate)?.content)?.parts) ?? []).some((part) => {
      const value = asRecord(part);
      const inlineData = asRecord(value?.inlineData ?? value?.inline_data);
      const functionCall = asRecord(value?.functionCall ?? value?.function_call);
      const args = asRecord(functionCall?.args);
      const partialArgs = asArray(functionCall?.partialArgs ?? functionCall?.partial_args) ?? [];

      return (
        (typeof value?.text === "string" && value.text.length > 0) ||
        (args !== undefined && Object.keys(args).length > 0) ||
        partialArgs.some((partialArg) => {
          const partial = asRecord(partialArg);
          const stringValue = partial?.stringValue ?? partial?.string_value;

          return (
            typeof partial?.boolValue === "boolean" ||
            typeof partial?.bool_value === "boolean" ||
            typeof partial?.numberValue === "number" ||
            typeof partial?.number_value === "number" ||
            (typeof stringValue === "string" && stringValue.length > 0) ||
            partial?.nullValue === "NULL_VALUE" ||
            partial?.null_value === "NULL_VALUE"
          );
        }) ||
        (typeof inlineData?.mimeType === "string" &&
          inlineData.mimeType.startsWith("audio/") &&
          typeof inlineData.data === "string" &&
          inlineData.data.length > 0)
      );
    }),
  );
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

function patchOperationsMethod(
  operations: UnknownRecord | undefined,
  method: "get" | "getVideosOperation",
  provider: string,
  tracker: VideoOperationTracker,
): void {
  if (!operations) return;
  const current = operations[method];

  if (isWrapped(current) && Object.prototype.hasOwnProperty.call(operations, method)) return;
  const original = isWrapped(current) ? current[ORIGINAL] : current;

  if (typeof original !== "function") return;
  const bound = original.bind(operations) as (...args: unknown[]) => unknown;

  const wrapped = function (...args: unknown[]) {
    const result = bound(...args);

    let operation: UnknownRecord | undefined;
    let operationName: string | undefined;

    try {
      operation = asRecord(asRecord(args[0])?.operation);
      operationName = readString(operation?.name);
    } catch {
      return result;
    }

    if (!operation || !tracker.operations.has(operation)) {
      return result;
    }

    const request = mediaRequestFields({}, "generateVideos.poll", "video");
    request.fields.responseId = operationName;
    const span = startSpan("generate_content video operation", { ...request.fields, provider });
    const end = endOnce(span);

    const finishResponse = (response: unknown) => {
      try {
        const record = asRecord(response);

        if (record?.done === true) tracker.operations.delete(operation);
        else if (record) tracker.operations.add(record);
      } catch {}

      const fields = safeResponseFields((value) => mediaResponseFields(value, "video"), response);
      end(fields);

      return response;
    };

    try {
      if (isThenable(result)) {
        return result.then(finishResponse, (error) => {
          end({ error });
          throw error;
        });
      }

      return finishResponse(result);
    } catch (error) {
      end({ error });
      throw error;
    }
  };

  operations[method] = markWrapped(wrapped as WrappedFunction, bound);
}

export function wrapGoogleGenAI<T extends GoogleGenAIClientLike>(client: T): T {
  if (wrappedClients.has(client)) return client;
  const provider = client.vertexai === true ? "gcp.vertex_ai" : "gcp.gemini";
  const models = client.models as UnknownRecord;
  const videoTracker: VideoOperationTracker = { operations: new WeakSet() };
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

  for (const [key, outputType] of [
    ["generateImages", "image"],
    ["editImage", "image"],
    ["upscaleImage", "image"],
    ["generateVideos", "video"],
  ] as const) {
    patchModelsMethod(
      models,
      key,
      (params) => mediaRequestFields(params, key, outputType),
      outputType === "video"
        ? (response) => videoResponseFields(response, videoTracker)
        : imageResponseFields,
      provider,
      { namePrefix: "generate_content", type: "generation" },
    );
  }

  const operations = asRecord(client.operations);
  patchOperationsMethod(operations, "get", provider, videoTracker);
  patchOperationsMethod(operations, "getVideosOperation", provider, videoTracker);
  wrappedClients.add(client);

  return client;
}
