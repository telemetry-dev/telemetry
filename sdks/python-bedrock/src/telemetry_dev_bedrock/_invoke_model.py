from __future__ import annotations

import json
from typing import Any

from ._fields import clean


def parse_body(body: Any, content_type: Any = "application/json") -> Any:
    if isinstance(content_type, str) and "json" not in content_type:
        return None
    if isinstance(body, bytes | bytearray):
        raw = bytes(body).decode("utf-8")
    elif isinstance(body, str):
        raw = body
    elif all(hasattr(body, name) for name in ("tell", "read", "seek")):
        seekable = getattr(body, "seekable", None)
        if callable(seekable):
            try:
                if not seekable():
                    return None
            except Exception:
                return None
        try:
            position = body.tell()
            body.seek(position)
            value = body.read()
            body.seek(position)
        except Exception:
            return None
        if isinstance(value, bytes | bytearray):
            raw = bytes(value).decode("utf-8")
        elif isinstance(value, str):
            raw = value
        else:
            return None
    else:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def is_embedding_model(model_id: Any) -> bool:
    model = model_id.lower() if isinstance(model_id, str) else ""
    return "titan-embed" in model or "cohere.embed" in model or "-embedding" in model


def request_fields(params: dict[str, Any]) -> dict[str, Any]:
    model = params.get("modelId") if isinstance(params.get("modelId"), str) else None
    body = parse_body(params.get("body"), params.get("contentType"))
    fields = {"provider": "amazon-bedrock", "model": model, "input": body}
    if is_embedding_model(model):
        fields["output_type"] = "embedding"
    fields.update(sampling_fields(model, body))
    return clean(fields)


def sampling_fields(model: str | None, body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    lower = model.lower() if isinstance(model, str) else ""
    if "amazon.titan" in lower:
        cfg = (
            body.get("textGenerationConfig")
            if isinstance(body.get("textGenerationConfig"), dict)
            else {}
        )
        return clean(
            {
                "temperature": cfg.get("temperature"),
                "top_p": cfg.get("topP"),
                "max_tokens": cfg.get("maxTokenCount"),
                "stop_sequences": cfg.get("stopSequences"),
            }
        )
    if "amazon.nova" in lower:
        cfg = body.get("inferenceConfig") if isinstance(body.get("inferenceConfig"), dict) else {}
        return clean(
            {
                "temperature": cfg.get("temperature"),
                "top_p": cfg.get("topP") or cfg.get("top_p"),
                "top_k": cfg.get("topK"),
                "max_tokens": cfg.get("maxTokens") or cfg.get("max_new_tokens"),
                "stop_sequences": cfg.get("stopSequences"),
            }
        )
    if "anthropic.claude" in lower:
        return clean(
            {
                "temperature": body.get("temperature"),
                "top_p": body.get("top_p"),
                "top_k": body.get("top_k"),
                "max_tokens": body.get("max_tokens"),
                "stop_sequences": body.get("stop_sequences"),
            }
        )
    if "meta.llama" in lower:
        return clean(
            {
                "temperature": body.get("temperature"),
                "top_p": body.get("top_p"),
                "max_tokens": body.get("max_gen_len"),
            }
        )
    return clean(
        {
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p") or body.get("topP"),
            "max_tokens": body.get("max_tokens") or body.get("maxTokens"),
        }
    )


def response_fields(
    model: str | None, body: Any, headers: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not isinstance(body, dict):
        return header_usage_fields(headers)
    lower = model.lower() if isinstance(model, str) else ""
    if is_embedding_model(model):
        return _with_header_usage(
            {
                "output": body,
                "output_type": "embedding",
                "usage": clean({"input_tokens": body.get("inputTextTokenCount")}) or None,
            },
            headers,
        )
    if "amazon.titan" in lower:
        first = body.get("results", [{}])[0] if isinstance(body.get("results"), list) else {}
        first = first if isinstance(first, dict) else {}
        return clean(
            {
                "output": body,
                "finish_reason": first.get("completionReason"),
                "usage": clean(
                    {
                        "input_tokens": body.get("inputTextTokenCount"),
                        "output_tokens": first.get("tokenCount"),
                    }
                )
                or None,
            }
        )
    if "amazon.nova" in lower:
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return clean(
            {
                "output": body,
                "finish_reason": body.get("stopReason"),
                "usage": clean(
                    {
                        "input_tokens": usage.get("inputTokens"),
                        "output_tokens": usage.get("outputTokens"),
                        "total_tokens": usage.get("totalTokens"),
                    }
                )
                or None,
            }
        )
    if "anthropic.claude" in lower:
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        parsed_usage = clean(
            {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            }
        )
        return clean(
            {
                "output": body,
                "finish_reason": body.get("stop_reason"),
                "usage": parsed_usage or header_usage_fields(headers).get("usage"),
            }
        )
    if "meta.llama" in lower:
        return clean(
            {
                "output": body,
                "finish_reason": body.get("stop_reason"),
                "usage": clean(
                    {
                        "input_tokens": body.get("prompt_token_count"),
                        "output_tokens": body.get("generation_token_count"),
                    }
                )
                or None,
            }
        )
    if "mistral" in lower:
        first = body.get("outputs", [{}])[0] if isinstance(body.get("outputs"), list) else {}
        first = first if isinstance(first, dict) else {}
        return _with_header_usage(
            {"output": body, "finish_reason": first.get("stop_reason")}, headers
        )
    if "cohere.command-r" in lower:
        return _with_header_usage(
            {"output": body, "finish_reason": body.get("finish_reason")}, headers
        )
    if "cohere.command" in lower:
        first = (
            body.get("generations", [{}])[0] if isinstance(body.get("generations"), list) else {}
        )
        first = first if isinstance(first, dict) else {}
        return _with_header_usage(
            {"output": body, "finish_reason": first.get("finish_reason")}, headers
        )
    return _with_header_usage({"output": body}, headers)


def _with_header_usage(fields: dict[str, Any], headers: dict[str, Any] | None) -> dict[str, Any]:
    cleaned = clean(fields)
    if cleaned.get("usage"):
        return cleaned
    return clean({**cleaned, **header_usage_fields(headers)})


def header_usage_fields(headers: dict[str, Any] | None) -> dict[str, Any]:
    if not headers:
        return {}
    input_tokens = _int_header(headers, "x-amzn-bedrock-input-token-count")
    output_tokens = _int_header(headers, "x-amzn-bedrock-output-token-count")
    return clean(
        {
            "usage": clean({"input_tokens": input_tokens, "output_tokens": output_tokens}) or None,
        }
    )


def _int_header(headers: dict[str, Any], key: str) -> int | None:
    value = headers.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
