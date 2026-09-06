from __future__ import annotations

import json
from typing import Any

PROVIDER = "amazon-bedrock"


def clean(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not None}


def metadata_fields(response: dict[str, Any] | BaseException | None) -> dict[str, Any]:
    metadata: dict[str, Any] | None = None
    if isinstance(response, dict) and isinstance(response.get("ResponseMetadata"), dict):
        metadata = response["ResponseMetadata"]
    else:
        error_response = getattr(response, "response", None)
        if isinstance(error_response, dict) and isinstance(
            error_response.get("ResponseMetadata"), dict
        ):
            metadata = error_response["ResponseMetadata"]
    if metadata is None:
        return {}
    retry_attempts = metadata.get("RetryAttempts")
    attempts = retry_attempts + 1 if isinstance(retry_attempts, int) else None
    total_retry_delay = metadata.get("TotalRetryDelay")
    if total_retry_delay is None:
        total_retry_delay = metadata.get("totalRetryDelay")
    attributes = clean(
        {
            "aws.http.status_code": metadata.get("HTTPStatusCode"),
            "aws.request.attempts": attempts if attempts and attempts > 1 else None,
            "aws.request.total_retry_delay_ms": total_retry_delay,
        }
    )
    return clean(
        {
            "response_id": metadata.get("RequestId"),
            "attributes": attributes or None,
        }
    )


def error_fields(error: BaseException) -> dict[str, Any]:
    fields = metadata_fields(error)
    response = getattr(error, "response", None)
    code = None
    if isinstance(response, dict) and isinstance(response.get("Error"), dict):
        code = response["Error"].get("Code")
    fields["error"] = error
    if code:
        attrs = dict(fields.get("attributes") or {})
        attrs["aws.error.code"] = code
        fields["attributes"] = attrs
    return fields


def usage_from_converse(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    usage = clean(
        {
            "input_tokens": value.get("inputTokens"),
            "output_tokens": value.get("outputTokens"),
            "total_tokens": value.get("totalTokens"),
            "cache_read_input_tokens": value.get("cacheReadInputTokens"),
            "cache_creation_input_tokens": value.get("cacheWriteInputTokens"),
        }
    )
    return usage or None


def metadata_attr_value(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, default=repr, ensure_ascii=False)


def merge_fields(*parts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        metadata = part.get("metadata")
        attributes = part.get("attributes")
        usage = part.get("usage")
        for key, value in part.items():
            if key not in {"metadata", "attributes", "usage"} and value is not None:
                merged[key] = value
        if isinstance(metadata, dict):
            merged["metadata"] = {
                **merged.get("metadata", {}),
                **{
                    key: normalized
                    for key, value in metadata.items()
                    if (normalized := metadata_attr_value(value)) is not None
                },
            }
        if isinstance(attributes, dict):
            merged["attributes"] = {**merged.get("attributes", {}), **attributes}
        if isinstance(usage, dict):
            merged["usage"] = {**merged.get("usage", {}), **usage}
    return merged
