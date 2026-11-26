import {
  type BaseMessage,
  isAIMessage,
  isBaseMessage,
  isToolMessage,
} from "@langchain/core/messages";
import type { UsageMetadata } from "@langchain/core/messages";
import type { LLMResult } from "@langchain/core/outputs";
import {
  captureEnabled,
  type GenerationFields,
  type SpanFields,
  type TokenUsage,
} from "@telemetry-dev/sdk";

import { plain, plainMessage } from "./plain.ts";

type Metadata = Record<string, unknown>;

// LangChain `ls_provider` values whose OpenTelemetry provider name is different.
const PROVIDERS: Record<string, string> = {
  amazon_bedrock: "aws.bedrock",
  azure: "azure.ai.openai",
  fireworks: "fireworks_ai",
  google_genai: "gcp.gen_ai",
  google_vertexai: "gcp.vertex_ai",
  mistralai: "mistral_ai",
  together: "together_ai",
};

/**
 * The request fields of a model run. LangChain puts the model, provider, and main sampling
 * parameters in the `ls_*` metadata of each chat model, and the full request in
 * `invocation_params`.
 */
export function generationRequest(
  extraParams: Metadata | undefined,
  metadata: Metadata | undefined,
): GenerationFields & { attributes?: Record<string, string> } {
  const params = record(extraParams?.["invocation_params"]);
  const provider = string(metadata?.["ls_provider"]);
  const tools = toolDefinitions(params?.["tools"]);

  return {
    attributes: tools.length > 0 ? { "gen_ai.tool.definitions": toolsJson(tools) } : undefined,
    model:
      string(metadata?.["ls_model_name"]) ??
      string(params?.["model"]) ??
      string(params?.["model_name"]) ??
      string(params?.["modelName"]),
    provider: provider === undefined ? undefined : (PROVIDERS[provider] ?? provider),
    temperature: number(metadata?.["ls_temperature"]) ?? number(params?.["temperature"]),
    maxTokens:
      number(metadata?.["ls_max_tokens"]) ??
      number(params?.["max_tokens"]) ??
      number(params?.["max_completion_tokens"]),
    topP: number(params?.["top_p"]) ?? number(params?.["topP"]),
    stopSequences: strings(metadata?.["ls_stop"]),
  };
}

/** The response fields of a model run, from its first generation. */
export function completionFields(output: LLMResult): SpanFields {
  const generation = output.generations[0]?.[0];

  if (!generation) return {};

  const message: BaseMessage | undefined =
    "message" in generation && isBaseMessage(generation.message) ? generation.message : undefined;

  const ai = message && isAIMessage(message) ? message : undefined;
  const response = record(message?.response_metadata);
  // A streamed Anthropic reply keeps the model and stop reason of the message here.
  const extra = record(message?.additional_kwargs);
  const info = record(generation.generationInfo);
  const llmOutput = record(output.llmOutput);

  return {
    output: message ? [plainMessage(message)] : generation.text,
    usage: usageOf(ai?.usage_metadata) ?? legacyUsage(record(llmOutput?.["tokenUsage"])),
    responseModel:
      string(response?.["model_name"]) ??
      string(response?.["model"]) ??
      string(extra?.["model"]) ??
      string(llmOutput?.["model_name"]),
    responseId: string(response?.["id"]) ?? providerId(message?.id),
    finishReason:
      string(response?.["finish_reason"]) ??
      string(response?.["stop_reason"]) ??
      string(extra?.["stop_reason"]) ??
      string(info?.["finish_reason"]),
  };
}

/** Tool input arrives as text: JSON for a structured tool. */
export function toolInput(input: string): unknown {
  try {
    return JSON.parse(input);
  } catch {
    return input;
  }
}

/** A tool called with a tool call returns a ToolMessage; record its content. */
export function toolOutput(output: unknown): unknown {
  if (isBaseMessage(output) && isToolMessage(output)) return output.content;

  return plain(output);
}

interface ToolDefinition {
  type: string;
  name?: string;
  description?: string;
  parameters?: unknown;
}

/**
 * The tools of a model run in the formats of the providers LangChain calls: OpenAI
 * (`function`), Anthropic (`input_schema`), Bedrock (`toolSpec`), and Gemini
 * (`functionDeclarations`). Descriptions and schemas are prompt content and need input capture
 * on. The core SDK also strips them with capture off; this check keeps them out with a core
 * older than that rule.
 */
function toolDefinitions(tools: unknown): ToolDefinition[] {
  if (!Array.isArray(tools)) return [];
  const content = captureEnabled("input");
  const definition = (
    type: string,
    name: string,
    description: unknown,
    parameters: unknown,
  ): ToolDefinition =>
    content ? { type, name, description: string(description), parameters } : { type, name };

  return tools.flatMap((item: unknown) => {
    const tool = record(item);
    const declarations = tool?.["functionDeclarations"];

    if (Array.isArray(declarations)) {
      return declarations.flatMap((item: unknown) => {
        const declaration = record(item);
        const name = string(declaration?.["name"]);

        return name
          ? [
              definition(
                "function",
                name,
                declaration?.["description"],
                declaration?.["parameters"],
              ),
            ]
          : [];
      });
    }

    const fn = record(tool?.["function"]);
    const spec = record(tool?.["toolSpec"]);
    const name = string(fn?.["name"]) ?? string(spec?.["name"]) ?? string(tool?.["name"]);

    const type = string(tool?.["type"]);

    // A built-in tool without a name, such as OpenAI's `{type: "web_search"}`, by its type.
    if (!name) return type && type !== "function" && type !== "custom" ? [{ type }] : [];

    // A built-in tool; a Responses-format function tool also has its fields at the top.
    if (
      fn === undefined &&
      spec === undefined &&
      type !== undefined &&
      type !== "custom" &&
      type !== "function"
    ) {
      return [{ type, name }];
    }
    const source = fn ?? spec ?? tool;

    return [
      definition(
        "function",
        name,
        source?.["description"],
        fn?.["parameters"] ??
          record(spec?.["inputSchema"])?.["json"] ??
          tool?.["input_schema"] ??
          tool?.["parameters"],
      ),
    ];
  });
}

function usageOf(usage: UsageMetadata | undefined): TokenUsage | undefined {
  if (!usage) return undefined;
  const input = usage.input_token_details;
  const output = usage.output_token_details;

  return {
    inputTokens: usage.input_tokens,
    outputTokens: usage.output_tokens,
    totalTokens: usage.total_tokens,
    cacheReadInputTokens: input?.cache_read,
    cacheCreationInputTokens: input?.cache_creation,
    reasoningOutputTokens: output?.reasoning,
    textInputTokens: input?.text,
    textOutputTokens: output?.text,
    imageInputTokens: input?.image,
    imageOutputTokens: output?.image,
    audioInputTokens: input?.audio,
    audioOutputTokens: output?.audio,
  };
}

// Older integrations report usage only in `llmOutput.tokenUsage`.
function legacyUsage(usage: Metadata | undefined): TokenUsage | undefined {
  if (!usage) return undefined;

  return {
    inputTokens: number(usage["promptTokens"]),
    outputTokens: number(usage["completionTokens"]),
    totalTokens: number(usage["totalTokens"]),
  };
}

// LangChain gives a message without a provider id the id of its run, as "run-<uuid>".
function providerId(id: string | undefined): string | undefined {
  return id?.startsWith("run-") ? undefined : id;
}

function record(value: unknown): Metadata | undefined {
  return value !== null && typeof value === "object" ? (value as Metadata) : undefined;
}

function string(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function number(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function strings(value: unknown): string[] | undefined {
  return Array.isArray(value) && value.length > 0 && value.every((item) => typeof item === "string")
    ? value
    : undefined;
}

/**
 * The definitions as JSON, or only their types and names when a schema cannot be serialized,
 * such as one holding a BigInt or a cycle: the wrapper maps the request outside any catch.
 */
function toolsJson(definitions: Array<{ type: string; name?: string }>): string {
  try {
    return JSON.stringify(definitions);
  } catch {
    return JSON.stringify(definitions.map(({ type, name }) => ({ type, name })));
  }
}
