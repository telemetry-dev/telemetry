import type { DocumentInterface } from "@langchain/core/documents";
import {
  type BaseMessage,
  isAIMessage,
  isBaseMessage,
  isToolMessage,
} from "@langchain/core/messages";

// Past this depth, values go to JSON.stringify as they are; this also stops at reference cycles.
const MAX_DEPTH = 8;

const ROLES: Record<string, string> = {
  human: "user",
  ai: "assistant",
  system: "system",
  developer: "developer",
  tool: "tool",
  function: "function",
};

// Anthropic, LangChain standard, OpenAI Responses, and Gemini spellings.
const TOOL_CALL_BLOCKS = new Set(["tool_use", "tool_call", "function_call", "functionCall"]);

/**
 * A copy of a run's input or output in which messages, prompt values, and documents are plain
 * objects. Without it, these LangChain objects serialize to their verbose constructor form.
 */
export function plain(value: unknown, depth = 0): unknown {
  if (value === null || typeof value !== "object" || depth > MAX_DEPTH) return value;

  if (isBaseMessage(value)) return plainMessage(value);

  if (isPromptValue(value)) return value.toChatMessages().map(plainMessage);

  if (isDocument(value)) return plainDocument(value);

  if (Array.isArray(value)) return value.map((item: unknown) => plain(item, depth + 1));

  const prototype: unknown = Object.getPrototypeOf(value);

  if (prototype !== Object.prototype && prototype !== null) return value;

  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [key, plain(item, depth + 1)]),
  );
}

/** A message in the OpenAI chat shape that the ingest reads. */
export function plainMessage(message: BaseMessage): Record<string, unknown> {
  const role =
    message.type === "generic"
      ? String((message as BaseMessage & { role?: unknown }).role)
      : (ROLES[message.type] ?? message.type);

  const result: Record<string, unknown> = { role, content: message.content };

  if (message.name) result.name = message.name;

  if (isAIMessage(message) && message.tool_calls?.length) {
    // The provider's own tool call blocks repeat the tool calls in the content.
    if (Array.isArray(message.content)) {
      result.content = message.content.filter(
        (block) => !TOOL_CALL_BLOCKS.has(String((block as { type?: unknown }).type)),
      );
    }

    result.tool_calls = message.tool_calls.map((call) => ({
      id: call.id,
      type: "function",
      function: { name: call.name, arguments: JSON.stringify(call.args) },
    }));
  }

  if (isToolMessage(message)) result.tool_call_id = message.tool_call_id;

  return result;
}

export function plainDocument(document: DocumentInterface): Record<string, unknown> {
  return { pageContent: document.pageContent, metadata: document.metadata };
}

function isPromptValue(value: object): value is { toChatMessages(): BaseMessage[] } {
  return "toChatMessages" in value && typeof value.toChatMessages === "function";
}

function isDocument(value: object): value is DocumentInterface {
  return "pageContent" in value && typeof value.pageContent === "string" && "metadata" in value;
}
