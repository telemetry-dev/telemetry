from __future__ import annotations

import json
from typing import Any


def normalize_messages(messages: Any) -> list[dict[str, Any]] | None:
    if not isinstance(messages, list):
        return None
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        normalized.append(
            {
                "role": message.get("role"),
                "parts": normalize_content_list(message.get("content")),
            }
        )
    return normalized


def normalize_output_message(
    message: Any, finish_reason: str | None = None
) -> list[dict[str, Any]] | None:
    if not isinstance(message, dict):
        return None
    output = {"role": message.get("role"), "parts": normalize_content_list(message.get("content"))}
    if finish_reason is not None:
        output["finish_reason"] = finish_reason
    return [output]


def normalize_content_list(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict):
            parts.extend(normalize_content_block(block))
    return parts


def normalize_content_block(block: dict[str, Any]) -> list[dict[str, Any]]:
    text = block.get("text")
    if isinstance(text, str):
        part: dict[str, Any] = {"type": "text", "content": text}
        if isinstance(block.get("citations"), list):
            part["citations"] = block["citations"]
        return [part]

    tool_use = block.get("toolUse")
    if isinstance(tool_use, dict):
        return [
            {
                key: value
                for key, value in {
                    "type": "tool_call",
                    "id": tool_use.get("toolUseId"),
                    "name": tool_use.get("name"),
                    "arguments": tool_use.get("input"),
                }.items()
                if value is not None
            }
        ]

    tool_result = block.get("toolResult")
    if isinstance(tool_result, dict):
        return [
            {
                key: value
                for key, value in {
                    "type": "tool_call_response",
                    "id": tool_result.get("toolUseId"),
                    "response": simplify_tool_result(tool_result.get("content")),
                }.items()
                if value is not None
            }
        ]

    reasoning = block.get("reasoningContent")
    if isinstance(reasoning, dict):
        reasoning_text = reasoning.get("reasoningText")
        if isinstance(reasoning_text, dict) and isinstance(reasoning_text.get("text"), str):
            return [{"type": "reasoning", "content": reasoning_text["text"]}]
        if reasoning.get("redactedContent") is not None:
            return [{"type": "reasoning", "content": "[redacted]"}]

    for modality in ("image", "document", "video", "audio"):
        media = block.get(modality)
        if not isinstance(media, dict):
            continue
        source = media.get("source") if isinstance(media.get("source"), dict) else {}
        fmt = media.get("format") if isinstance(media.get("format"), str) else None
        if source.get("bytes") is not None:
            part = {"type": "blob", "modality": modality}
            if fmt:
                part["mime_type"] = f"{modality}/{fmt}"
            return [part]
        s3 = source.get("s3Location") if isinstance(source.get("s3Location"), dict) else None
        uri = source.get("uri") if isinstance(source.get("uri"), str) else None
        if uri is None and s3 is not None:
            uri = s3.get("uri") if isinstance(s3.get("uri"), str) else None
        if uri is None and s3 is not None:
            bucket = s3.get("bucket")
            key = s3.get("key")
            if isinstance(bucket, str) and isinstance(key, str):
                uri = f"s3://{bucket}/{key}"
        if uri:
            part = {"type": "uri", "uri": uri, "modality": modality}
            if fmt:
                part["mime_type"] = fmt
            return [part]

    citations_content = block.get("citationsContent")
    if isinstance(citations_content, dict):
        content_value = citations_content.get("content")
        content: list[Any] = content_value if isinstance(content_value, list) else []
        citations_value = citations_content.get("citations")
        citations: list[Any] | None = citations_value if isinstance(citations_value, list) else None
        parts: list[dict[str, Any]] = []
        for content_block in content:
            if isinstance(content_block, dict):
                if citations is not None:
                    content_block = {**content_block, "citations": citations}
                parts.extend(normalize_content_block(content_block))
        return parts

    guard = block.get("guardContent")
    if isinstance(guard, dict):
        guard_text = guard.get("text")
        if isinstance(guard_text, dict) and isinstance(guard_text.get("text"), str):
            return [{"type": "text", "content": guard_text["text"]}]
        if isinstance(guard_text, str):
            return [{"type": "text", "content": guard_text}]

    try:
        return [{"type": "text", "content": json.dumps(block, default=str)}]
    except Exception:
        return [{"type": "text", "content": "[unsupported content block]"}]


def simplify_tool_result(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    if (
        len(content) == 1
        and isinstance(content[0], dict)
        and isinstance(content[0].get("text"), str)
    ):
        return content[0]["text"]
    return content
