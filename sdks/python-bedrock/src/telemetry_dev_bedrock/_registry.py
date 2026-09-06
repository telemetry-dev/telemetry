from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from ._fields import PROVIDER, clean, merge_fields, metadata_fields, usage_from_converse
from ._invoke_model import is_embedding_model
from ._invoke_model import request_fields as invoke_request_fields
from ._invoke_model import response_fields as invoke_response_fields
from ._messages import normalize_content_list, normalize_messages, normalize_output_message
from ._streams import (
    AgentStreamState,
    ConverseStreamState,
    FlowStreamState,
    InvokeModelStreamState,
    RagStreamState,
    StreamState,
)

SpanKind = Literal["generation", "embedding", "agent", "span"]


@dataclass(frozen=True)
class OperationSpec:
    span_name: Callable[[dict[str, Any]], str]
    span_type: Callable[[dict[str, Any]], SpanKind]
    request_fields: Callable[[dict[str, Any]], dict[str, Any]]
    response_fields: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    stream_key: str | None = None
    stream_state_factory: Callable[[bool], StreamState] | None = None


def _model(params: dict[str, Any]) -> str | None:
    return params.get("modelId") if isinstance(params.get("modelId"), str) else None


def _converse_request(params: dict[str, Any]) -> dict[str, Any]:
    inference = (
        params.get("inferenceConfig") if isinstance(params.get("inferenceConfig"), dict) else {}
    )
    guardrail = (
        params.get("guardrailConfig") if isinstance(params.get("guardrailConfig"), dict) else None
    )
    return clean(
        {
            "provider": PROVIDER,
            "model": _model(params),
            "input": normalize_messages(params.get("messages")),
            "system_instructions": normalize_content_list(params.get("system")) or None,
            "temperature": inference.get("temperature"),
            "top_p": inference.get("topP"),
            "max_tokens": inference.get("maxTokens"),
            "stop_sequences": inference.get("stopSequences"),
            "metadata": clean(
                {
                    "guardrail_id": guardrail.get("guardrailIdentifier") if guardrail else None,
                    "guardrail_version": guardrail.get("guardrailVersion") if guardrail else None,
                }
            )
            or None,
        }
    )


def _converse_response(_params: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    trace = response.get("trace") if isinstance(response.get("trace"), dict) else {}
    router = trace.get("promptRouter") if isinstance(trace.get("promptRouter"), dict) else {}
    guard = trace.get("guardrail") if isinstance(trace.get("guardrail"), dict) else {}
    metrics = response.get("metrics") if isinstance(response.get("metrics"), dict) else {}
    message = (
        response.get("output", {}).get("message")
        if isinstance(response.get("output"), dict)
        else None
    )
    stop = response.get("stopReason") if isinstance(response.get("stopReason"), str) else None
    return merge_fields(
        metadata_fields(response),
        clean(
            {
                "output": normalize_output_message(message, stop),
                "usage": usage_from_converse(response.get("usage")),
                "finish_reason": stop,
                "response_model": router.get("invokedModelId"),
                "metadata": clean(
                    {
                        "server_latency_ms": metrics.get("latencyMs"),
                        "guardrail_action": response.get("guardrailAction") or guard.get("action"),
                        "guardrail_action_reason": guard.get("actionReason"),
                    }
                )
                or None,
            }
        ),
    )


def _invoke_response(params: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    body = response.get("_telemetry_dev_parsed_body")
    metadata = (
        response.get("ResponseMetadata")
        if isinstance(response.get("ResponseMetadata"), dict)
        else {}
    )
    headers = metadata.get("HTTPHeaders") if isinstance(metadata.get("HTTPHeaders"), dict) else None
    return merge_fields(
        metadata_fields(response), invoke_response_fields(_model(params), body, headers)
    )


def _apply_request(params: dict[str, Any]) -> dict[str, Any]:
    return clean(
        {
            "provider": PROVIDER,
            "input": clean(
                {
                    "guardrailIdentifier": params.get("guardrailIdentifier"),
                    "guardrailVersion": params.get("guardrailVersion"),
                    "source": params.get("source"),
                    "content": params.get("content"),
                }
            ),
        }
    )


def _apply_response(_params: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    return merge_fields(
        metadata_fields(response),
        {
            "output": clean(
                {"outputs": response.get("outputs"), "assessments": response.get("assessments")}
            ),
            "metadata": clean(
                {
                    "guardrail_action": response.get("action"),
                    "guardrail_action_reason": response.get("actionReason"),
                }
            ),
        },
    )


def _agent_input(text: Any) -> list[dict[str, Any]] | None:
    return (
        [{"role": "user", "parts": [{"type": "text", "content": text}]}]
        if isinstance(text, str)
        else None
    )


def _agent_request(params: dict[str, Any]) -> dict[str, Any]:
    return clean(
        {
            "provider": PROVIDER,
            "input": _agent_input(params.get("inputText")),
            "agent_id": params.get("agentId"),
            "metadata": clean(
                {
                    "bedrock_agent_alias_id": params.get("agentAliasId"),
                    "bedrock_session_id": params.get("sessionId"),
                    "bedrock_memory_id": params.get("memoryId"),
                }
            )
            or None,
        }
    )


def _inline_agent_request(params: dict[str, Any]) -> dict[str, Any]:
    return clean(
        {
            "provider": PROVIDER,
            "model": params.get("foundationModel"),
            "input": _agent_input(params.get("inputText")),
            "agent_name": params.get("agentName"),
            "metadata": clean({"bedrock_session_id": params.get("sessionId")}) or None,
        }
    )


def _agent_response(_params: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    return merge_fields(
        metadata_fields(response),
        {
            "metadata": clean(
                {
                    "bedrock_session_id": response.get("sessionId"),
                    "bedrock_memory_id": response.get("memoryId"),
                }
            )
        },
    )


def _retrieve_request(params: dict[str, Any]) -> dict[str, Any]:
    return clean(
        {
            "provider": PROVIDER,
            "input": params.get("retrievalQuery"),
            "metadata": clean({"knowledge_base_id": params.get("knowledgeBaseId")}) or None,
        }
    )


def _retrieve_response(_params: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    results = (
        response.get("retrievalResults")
        if isinstance(response.get("retrievalResults"), list)
        else []
    )
    return merge_fields(
        metadata_fields(response),
        clean(
            {
                "output": results,
                "metadata": clean(
                    {
                        "citation_count": len(results),
                        "guardrail_action": response.get("guardrailAction"),
                    }
                ),
            }
        ),
    )


def _rag_model(params: dict[str, Any]) -> str | None:
    config = params.get("retrieveAndGenerateConfiguration")
    if not isinstance(config, dict):
        return None
    kb = (
        config.get("knowledgeBaseConfiguration")
        if isinstance(config.get("knowledgeBaseConfiguration"), dict)
        else {}
    )
    external = (
        config.get("externalSourcesConfiguration")
        if isinstance(config.get("externalSourcesConfiguration"), dict)
        else {}
    )
    model = kb.get("modelArn") or external.get("modelArn")
    return model if isinstance(model, str) else None


def _rag_name(params: dict[str, Any]) -> str:
    model = _rag_model(params)
    basename = model.replace(":", "/").split("/")[-1] if model else "knowledge-base"
    return f"retrieve_and_generate {basename}"


def _rag_request(params: dict[str, Any]) -> dict[str, Any]:
    input_value = params.get("input") if isinstance(params.get("input"), dict) else {}
    text = input_value.get("text") if isinstance(input_value.get("text"), str) else None
    return clean(
        {
            "provider": PROVIDER,
            "model": _rag_model(params),
            "input": _agent_input(text),
            "metadata": clean({"bedrock_session_id": params.get("sessionId")}) or None,
        }
    )


def _rag_response(_params: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    output = response.get("output") if isinstance(response.get("output"), dict) else {}
    text = output.get("text") if isinstance(output.get("text"), str) else None
    citations = response.get("citations") if isinstance(response.get("citations"), list) else []
    return merge_fields(
        metadata_fields(response),
        clean(
            {
                "output": (
                    [{"role": "assistant", "parts": [{"type": "text", "content": text}]}]
                    if text
                    else None
                ),
                "metadata": clean(
                    {
                        "citation_count": len(citations),
                        "guardrail_action": response.get("guardrailAction"),
                        "bedrock_session_id": response.get("sessionId"),
                    }
                ),
            }
        ),
    )


def _flow_request(params: dict[str, Any]) -> dict[str, Any]:
    return clean(
        {
            "provider": PROVIDER,
            "input": params.get("inputs"),
            "agent_id": params.get("flowIdentifier"),
        }
    )


def _flow_name(params: dict[str, Any]) -> str:
    return f"invoke_flow {params.get('flowIdentifier') or 'unknown'}"


def _invoke_name(params: dict[str, Any]) -> str:
    operation = "embeddings" if is_embedding_model(_model(params)) else "chat"
    return f"{operation} {_model(params) or 'unknown'}"


def _invoke_type(params: dict[str, Any]) -> SpanKind:
    return "embedding" if is_embedding_model(_model(params)) else "generation"


RUNTIME_OPERATIONS: dict[str, OperationSpec] = {
    "Converse": OperationSpec(
        span_name=lambda params: f"chat {_model(params) or 'unknown'}",
        span_type=lambda _params: "generation",
        request_fields=_converse_request,
        response_fields=_converse_response,
    ),
    "ConverseStream": OperationSpec(
        span_name=lambda params: f"chat {_model(params) or 'unknown'}",
        span_type=lambda _params: "generation",
        request_fields=_converse_request,
        response_fields=_converse_response,
        stream_key="stream",
        stream_state_factory=lambda _capture: ConverseStreamState(),
    ),
    "InvokeModel": OperationSpec(
        span_name=_invoke_name,
        span_type=_invoke_type,
        request_fields=invoke_request_fields,
        response_fields=_invoke_response,
    ),
    "InvokeModelWithResponseStream": OperationSpec(
        span_name=_invoke_name,
        span_type=_invoke_type,
        request_fields=invoke_request_fields,
        response_fields=_invoke_response,
        stream_key="body",
        stream_state_factory=lambda _capture: InvokeModelStreamState(),
    ),
    "ApplyGuardrail": OperationSpec(
        span_name=lambda params: (
            f"apply_guardrail {params.get('guardrailIdentifier') or 'unknown'}"
        ),
        span_type=lambda _params: "span",
        request_fields=_apply_request,
        response_fields=_apply_response,
    ),
}

AGENT_OPERATIONS: dict[str, OperationSpec] = {
    "InvokeAgent": OperationSpec(
        span_name=lambda params: f"invoke_agent {params.get('agentId') or 'unknown'}",
        span_type=lambda _params: "agent",
        request_fields=_agent_request,
        response_fields=_agent_response,
        stream_key="completion",
        stream_state_factory=lambda capture: AgentStreamState(capture_trace=capture),
    ),
    "InvokeInlineAgent": OperationSpec(
        span_name=lambda params: f"invoke_agent {params.get('agentName') or 'inline-agent'}",
        span_type=lambda _params: "agent",
        request_fields=_inline_agent_request,
        response_fields=_agent_response,
        stream_key="completion",
        stream_state_factory=lambda capture: AgentStreamState(capture_trace=capture),
    ),
    "Retrieve": OperationSpec(
        span_name=lambda params: f"retrieve {params.get('knowledgeBaseId') or 'unknown'}",
        span_type=lambda _params: "span",
        request_fields=_retrieve_request,
        response_fields=_retrieve_response,
    ),
    "RetrieveAndGenerate": OperationSpec(
        span_name=_rag_name,
        span_type=lambda _params: "generation",
        request_fields=_rag_request,
        response_fields=_rag_response,
    ),
    "RetrieveAndGenerateStream": OperationSpec(
        span_name=_rag_name,
        span_type=lambda _params: "generation",
        request_fields=_rag_request,
        response_fields=_rag_response,
        stream_key="stream",
        stream_state_factory=lambda _capture: RagStreamState(),
    ),
    "InvokeFlow": OperationSpec(
        span_name=_flow_name,
        span_type=lambda _params: "agent",
        request_fields=_flow_request,
        response_fields=lambda _params, response: metadata_fields(response),
        stream_key="responseStream",
        stream_state_factory=lambda _capture: FlowStreamState(),
    ),
}

SERVICE_OPERATIONS: dict[str, dict[str, OperationSpec]] = {
    "bedrock-runtime": RUNTIME_OPERATIONS,
    "bedrock-agent-runtime": AGENT_OPERATIONS,
}


def lookup_operation(service_name: str, operation_name: str) -> OperationSpec | None:
    service = SERVICE_OPERATIONS.get(service_name)
    if service is None:
        return None
    return service.get(operation_name)
