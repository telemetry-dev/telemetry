from __future__ import annotations

from typing import Literal

SCOPE_NAME = "telemetry_dev"

SpanType = Literal["span", "generation", "tool", "agent", "embedding"]
LogLevel = Literal["debug", "info", "warn", "warning", "error"]

# Plain spans ALWAYS map to "function" — an operation-less span carrying a propagated
# gen_ai.conversation.id would otherwise be classified as "llm" by the ingest heuristic.
SPAN_TYPE_TO_OPERATION: dict[str, str] = {
    "span": "function",
    "generation": "chat",
    "tool": "execute_tool",
    "agent": "invoke_agent",
    "embedding": "embeddings",
}

DURATION_METRIC_OPERATIONS = frozenset({"chat", "invoke_agent", "embeddings", "execute_tool"})
TOKEN_METRIC_OPERATIONS = frozenset({"chat", "invoke_agent", "embeddings"})

# Buckets verbatim from packages/ai/src/otel.ts (OTel GenAI semconv recommendations).
DURATION_BUCKETS = [
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
]
TOKEN_BUCKETS = [
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
]

SEVERITY: dict[str, int] = {"debug": 5, "info": 9, "warn": 13, "error": 17}

RESERVED_METADATA_KEYS = frozenset({"userId", "sessionId", "user_id", "session_id"})

ATTR_OPERATION = "gen_ai.operation.name"
ATTR_PROVIDER = "gen_ai.provider.name"
ATTR_REQUEST_MODEL = "gen_ai.request.model"
ATTR_RESPONSE_MODEL = "gen_ai.response.model"
ATTR_RESPONSE_ID = "gen_ai.response.id"
ATTR_OUTPUT_TYPE = "gen_ai.output.type"
ATTR_FINISH_REASONS = "gen_ai.response.finish_reasons"
ATTR_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
ATTR_INPUT_MESSAGES = "gen_ai.input.messages"
ATTR_OUTPUT_MESSAGES = "gen_ai.output.messages"
ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_TOOL_CALL_ID = "gen_ai.tool.call.id"
ATTR_TOOL_DESCRIPTION = "gen_ai.tool.description"
ATTR_TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
ATTR_TOOL_RESULT = "gen_ai.tool.call.result"
ATTR_AGENT_NAME = "gen_ai.agent.name"
ATTR_AGENT_ID = "gen_ai.agent.id"
ATTR_COST = "gen_ai.usage.cost"
ATTR_USER_ID = "user.id"
ATTR_SESSION_ID = "gen_ai.conversation.id"
ATTR_ERROR_TYPE = "error.type"
# Emitted in SECONDS — the ingest multiplies non-"ms" duration keys by 1000.
ATTR_TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"

METADATA_PREFIX = "td.metadata."

USAGE_ATTRS: dict[str, str] = {
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "total_tokens": "gen_ai.usage.total_tokens",
    "cache_read_input_tokens": "gen_ai.usage.cache_read.input_tokens",
    "cache_creation_input_tokens": "gen_ai.usage.cache_creation.input_tokens",
    "reasoning_output_tokens": "gen_ai.usage.reasoning.output_tokens",
}

SAMPLING_ATTRS: dict[str, str] = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "top_k": "gen_ai.request.top_k",
    "max_tokens": "gen_ai.request.max_tokens",
    "stop_sequences": "gen_ai.request.stop_sequences",
    "seed": "gen_ai.request.seed",
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
}

METRIC_ATTR_KEYS = (ATTR_OPERATION, ATTR_PROVIDER, ATTR_REQUEST_MODEL, ATTR_RESPONSE_MODEL)


def input_key(operation: str) -> str:
    return ATTR_TOOL_ARGUMENTS if operation == "execute_tool" else ATTR_INPUT_MESSAGES


def output_key(operation: str) -> str:
    return ATTR_TOOL_RESULT if operation == "execute_tool" else ATTR_OUTPUT_MESSAGES
