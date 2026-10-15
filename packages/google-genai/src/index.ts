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
import * as telemetrySdk from "@telemetry-dev/sdk";

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

// Older cores enforce captureOutput and masking when the span ends but expose no policy, so
// withhold incomplete output from their mask. Retention uses the fixed stream bounds either way;
// maxAttributeLength is only here to satisfy the policy type.
const LEGACY_CAPTURE_POLICY: NonNullable<SpanHandle["capturePolicy"]> = {
  output: true,
  mask: true,
  maxAttributeLength: 65_536,
};

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
  /** Captured fields that were cut to the retention bound. */
  cut?: Array<"input" | "systemInstructions">;
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

// Reads like the capture walker does: own enumerable properties only.
function ownField(value: unknown, key: string): unknown {
  return value !== null &&
    typeof value === "object" &&
    Object.prototype.propertyIsEnumerable.call(value, key)
    ? (value as UnknownRecord)[key]
    : undefined;
}

// boundedCapture's default item limit; more candidates than this cannot all be kept.
const CAPTURE_MAX_ITEMS = 1_000;

// Retained finish reasons stay outside the output budget so a terminal one is still exported,
// so each is capped instead. Gemini's longest is 25 characters.
const FINISH_REASON_MAX_LENGTH = 128;

function candidateOutput(response: UnknownRecord) {
  if (!captureEnabled("output")) return { value: undefined, truncated: false, contentCut: false };
  const all = asArray(response.candidates);
  let projectionFailed = false;

  // Only candidate content is exported as output. Bounding a projection of it, rather than whole
  // candidates, keeps grounding or safety metadata from marking complete content as cut.
  const projected = all?.slice(0, CAPTURE_MAX_ITEMS + 1).map((candidate) => {
    try {
      const content = asRecord(ownField(candidate, "content"));

      return content
        ? { content: { role: ownField(content, "role"), parts: ownField(content, "parts") } }
        : {};
    } catch {
      projectionFailed = true;

      return {};
    }
  });

  const capture = boundedTelemetryCapture(projected);

  // Metadata that does not fit is still a loss for the span-level flag; both passes are bounded.
  const contentCut = capture.truncated || projectionFailed;
  const truncated = contentCut || boundedTelemetryCapture(response.candidates).truncated;

  const candidates = asArray(capture.value);

  if (!candidates || candidates.length === 0) return { value: undefined, truncated, contentCut };

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

  return {
    value: output.length > 0 ? output : undefined,
    truncated,
    contentCut,
  };
}

/**
 * A number, a boolean, an enum-style identifier, one of a closed set of strings, a list, or an
 * object with only the listed keys.
 */
type StructureShape =
  | "number"
  | "boolean"
  | "enum"
  | ReadonlySet<string>
  | readonly [StructureShape]
  | StructureObject;

interface StructureObject {
  readonly [key: string]: StructureShape;
}

// Response metadata is written as raw attributes, which skip captureOutput and the mask. Without
// raw output capture only these fields are kept; segment text, retrieved document text, titles,
// URIs, search queries, the search widget and block reason messages are output content. Keep in
// sync with Python.
const GROUNDING_STRUCTURE: StructureShape = {
  groundingChunks: [{ web: {}, retrievedContext: {}, maps: {}, image: {} }],
  groundingSupports: [
    {
      segment: { startIndex: "number", endIndex: "number", partIndex: "number" },
      groundingChunkIndices: ["number"],
      confidenceScores: ["number"],
    },
  ],
  retrievalMetadata: { googleSearchDynamicRetrievalScore: "number" },
};

const URL_RETRIEVAL_STATUSES: ReadonlySet<string> = new Set([
  "URL_RETRIEVAL_STATUS_UNSPECIFIED",
  "URL_RETRIEVAL_STATUS_SUCCESS",
  "URL_RETRIEVAL_STATUS_ERROR",
  "URL_RETRIEVAL_STATUS_PAYWALL",
  "URL_RETRIEVAL_STATUS_UNSAFE",
]);

const URL_CONTEXT_STRUCTURE: StructureShape = {
  urlMetadata: [{ urlRetrievalStatus: URL_RETRIEVAL_STATUSES }],
};

// Harm categories gain values often, so enum fields keep any identifier-shaped value.
const SAFETY_RATINGS_STRUCTURE: StructureShape = [
  {
    category: "enum",
    probability: "enum",
    probabilityScore: "number",
    severity: "enum",
    severityScore: "number",
    blocked: "boolean",
    overwrittenThreshold: "enum",
  },
];

const ENUM_VALUE = /^[A-Z][A-Z0-9_]{0,127}$/;

function projectStructure(value: unknown, shape: StructureShape): unknown {
  if (shape === "number") return readNumber(value);

  if (shape === "boolean") return typeof value === "boolean" ? value : undefined;

  if (shape === "enum")
    return typeof value === "string" && ENUM_VALUE.test(value) ? value : undefined;

  if (shape instanceof Set)
    return typeof value === "string" && shape.has(value) ? value : undefined;

  if (Array.isArray(shape)) return asArray(value)?.map((item) => projectStructure(item, shape[0]));
  const record = asRecord(value);

  if (!record || Array.isArray(record)) return undefined;
  const projected: UnknownRecord = {};

  for (const [key, nested] of Object.entries(shape as StructureObject)) {
    const field = projectStructure(record[key], nested);

    if (field !== undefined) projected[key] = field;
  }

  return projected;
}

/** `metadataContent` keeps response metadata whole instead of its structure. */
function responseAttributes(response: UnknownRecord, metadataContent: boolean) {
  const metadata = (value: unknown, shape: StructureShape) =>
    metadataContent ? value : projectStructure(value, shape);

  const attributes: Record<string, AttributeValue> = {};
  const candidates = asArray(response.candidates) ?? [];
  const promptFeedback = asRecord(response.promptFeedback);
  const blockReason = readString(promptFeedback?.blockReason);

  if (blockReason) attributes["google_genai.response.block_reason"] = blockReason;
  const blockReasonMessage = readString(promptFeedback?.blockReasonMessage);

  if (blockReasonMessage && metadataContent)
    attributes["google_genai.response.block_reason_message"] = blockReasonMessage;
  const promptSafetyRatings = asArray(promptFeedback?.safetyRatings);

  if (promptSafetyRatings && promptSafetyRatings.length > 0) {
    setJsonAttribute(
      attributes,
      "google_genai.response.prompt_safety_ratings",
      metadata(promptSafetyRatings, SAFETY_RATINGS_STRUCTURE),
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

      return {
        candidateIndex: readNumber(record?.index) ?? index,
        ratings: metadata(ratings, SAFETY_RATINGS_STRUCTURE),
      };
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
      metadata(firstCandidate.groundingMetadata, GROUNDING_STRUCTURE),
    );
  }

  if (firstCandidate?.urlContextMetadata !== undefined) {
    setJsonAttribute(
      attributes,
      "google_genai.response.url_context_metadata",
      metadata(firstCandidate.urlContextMetadata, URL_CONTEXT_STRUCTURE),
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

  const cut: RequestMapping["cut"] = [];

  if (input?.truncated) cut.push("input");

  if (systemInstructions?.truncated) cut.push("systemInstructions");

  return { name: `chat ${model ?? "unknown"}`, fields, cut };
}

function generateResponseFields(response: unknown, masked = false): SpanFields {
  const record = asRecord(response) ?? {};
  const candidates = asArray(record.candidates) ?? [];

  const finishReasons = candidates
    .map((candidate) => readString(asRecord(candidate)?.finishReason))
    .filter((reason): reason is string => reason !== undefined);

  const attributes = responseAttributes(record, captureEnabled("output") && !masked);
  const afcHistory = asArray(record.automaticFunctionCallingHistory);
  const output = candidateOutput(record);

  const fields: SpanFields = {
    responseModel: readString(record.modelVersion),
    responseId: readString(record.responseId),
    finishReason: finishReasons[0],
    output: output.contentCut && masked ? undefined : output.value,
    usage: usageFromMetadata(record.usageMetadata),
  };

  if (output.truncated) attributes["telemetry.dev.capture.truncated"] = true;

  if (captureEnabled("input") && afcHistory && afcHistory.length > 0) {
    const capture = boundedTelemetryCapture(afcHistory);
    fields.input = capture.truncated && masked ? undefined : capture.value;

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
    cut: input?.truncated ? ["input"] : undefined,
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

// A capture cut to the retention bound keeps its truncated flag, but a mask cannot be trusted to
// redact a value it only sees part of. Cores without a capture policy are treated as masked.
function capturesWithMask(span: SpanHandle): boolean {
  return span.capturePolicy?.mask ?? true;
}

// Cores older than captureMasked expose no mask policy before a span starts; treat them as
// masked, like LEGACY_CAPTURE_POLICY. A namespace read keeps them loadable without the export.
function captureMaskedBeforeStart(): boolean {
  const read = (telemetrySdk as { captureMasked?: () => boolean }).captureMasked;

  try {
    return typeof read === "function" ? read() : true;
  } catch {
    return true;
  }
}

// The decision is made before the span starts so samplers see the same attributes as before.
function startRequestSpan(request: RequestMapping, provider: string): SpanHandle {
  const fields = { ...request.fields, provider };
  const cut = request.cut?.filter((key) => fields[key] !== undefined) ?? [];

  if (cut.length === 0 || !captureMaskedBeforeStart()) return startSpan(request.name, fields);

  for (const key of cut) fields[key] = undefined;

  return startSpan(request.name, fields);
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
  mapResponse: (response: unknown, masked?: boolean) => SpanFields,
  provider: string,
  fallback: RequestFallback,
  models?: UnknownRecord,
): WrappedFunction {
  if (isWrapped(original)) return original;

  const wrapped = function (this: unknown, ...args: unknown[]) {
    const params = asRecord(args[0]) ?? {};
    const request = safeRequestMapping(params, mapRequest, fallback);
    const span = startRequestSpan(request, provider);
    const end = endOnce(span);
    const usageStore = afcUsageStore;

    const internalUsage: { usage?: SpanFields["usage"]; toolUsePromptTokens?: number } | undefined =
      usageStore && shouldAggregateAfcUsage(params) ? {} : undefined;

    if (internalUsage) installInternalUsageCollector(models);

    const finishResponse = (response: unknown) => {
      const fields = safeResponseFields(
        (value) => mapResponse(value, capturesWithMask(span)),
        response,
      );

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
  // The reconstructed AFC history holds a cut value, such as a folded model-output prefix.
  private afcInputIncomplete = false;

  constructor(
    private readonly requestInput: unknown,
    // The request input was cut to the retention bound, so history seeded from it is incomplete.
    private readonly requestInputCut: boolean,
    private readonly masked: boolean,
  ) {}

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

  get outputTruncated(): boolean {
    return this.outputCaptureTruncated;
  }

  get afcInputTruncated(): boolean {
    return this.afcInputIncomplete;
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

      if (this.requestInputCut) this.afcInputIncomplete = true;
    }

    return this.afcHistory;
  }

  /** Appends whole entries that fit and returns whether any of them was appended. */
  private appendAfcHistory(...entries: unknown[]): boolean {
    if (!captureEnabled("input")) return false;

    let appended = false;

    for (const entry of entries) {
      const capture = boundedTelemetryCapture([...this.ensureAfcHistory(), entry]);

      if (capture.truncated) {
        this.afcCaptureTruncated = true;
        continue;
      }

      this.afcHistory = asArray(capture.value) ?? [];
      appended = true;
    }

    if (this.outputCaptureTruncated) {
      this.afcCaptureTruncated = true;

      if (appended) this.afcInputIncomplete = true;
    }

    return appended;
  }

  private replaceAfcHistory(history: unknown[]): void {
    const capture = boundedTelemetryCapture(history);
    this.afcHistory = asArray(capture.value) ?? [];
    this.afcCaptureTruncated = capture.truncated;
    this.afcInputIncomplete = capture.truncated;
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
    // Candidates dropped by the index cap must not reach chunk attributes either.
    const rejected = new Set<unknown>();

    for (const candidate of candidates) {
      const candidateRecord = asRecord(candidate) ?? {};
      const content = capturesContent ? asRecord(candidateRecord.content) : undefined;

      if (content && hasFunctionResponse(content)) {
        const capture = boundedTelemetryCapture(copyContent(content));
        const capturedContent = asRecord(capture.value);
        this.foldTurn(true);

        const appended =
          capturedContent !== undefined && this.appendAfcHistory(copyContent(capturedContent));

        if (capture.truncated && captureEnabled("input")) {
          this.afcCaptureTruncated = true;

          if (appended) this.afcInputIncomplete = true;
        }

        continue;
      }

      const index = readNumber(candidateRecord.index) ?? 0;
      let state = this.candidates.get(index);

      if (!state) {
        // An entry outlives a failed output reservation so its finish reason is still
        // exported, so distinct indices are capped at the item bound; later ones are dropped.
        if (this.candidates.size >= CAPTURE_MAX_ITEMS) {
          this.outputCaptureTruncated = true;
          rejected.add(candidate);
          continue;
        }

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

      if (finishReason) state.finishReason = finishReason.slice(0, FINISH_REASON_MAX_LENGTH);
    }

    const chunkUsage = usageFromMetadata(record.usageMetadata);

    if (chunkUsage) this.usage = chunkUsage;

    const chunkAttributes = responseAttributes(
      rejected.size > 0
        ? { ...record, candidates: candidates.filter((candidate) => !rejected.has(candidate)) }
        : record,
      captureEnabled("output") && !this.masked,
    );

    // A chunk can repeat an index, so its own list would keep uncapped reasons; finish() builds
    // the list from the capped per-index state instead.
    delete chunkAttributes["gen_ai.response.finish_reasons"];
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
  requestInputCut = false,
): AsyncGenerator<unknown> {
  const iterator = source[Symbol.asyncIterator]();
  const masked = (span.capturePolicy ?? LEGACY_CAPTURE_POLICY).mask;
  const aggregator = new StreamAggregator(requestInput, requestInputCut, masked);
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

  // `completed` is true only when the source stream reached its end.
  const finish = (error?: unknown, completed = false): void => {
    if (ended) return;
    ended = true;
    let finalFields: SpanFields = {};

    try {
      finalFields = aggregator.finish();
    } catch {
      finalFields = {};
    }

    const interrupted = !completed || error !== undefined;

    if (interrupted) {
      finalFields.attributes = {
        ...finalFields.attributes,
        "telemetry.dev.capture.truncated": true,
      };
    }

    if (masked && (interrupted || aggregator.outputTruncated)) delete finalFields.output;

    // Reconstructed AFC history can carry a folded, incomplete model-output prefix.
    if (masked && aggregator.afcInputTruncated) delete finalFields.input;

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

        if (step.done) finish(undefined, true);
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
    const span = startRequestSpan(request, provider);
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
            request.cut?.includes("input") ?? false,
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
  mapResponse: (response: unknown, masked?: boolean) => SpanFields,
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
