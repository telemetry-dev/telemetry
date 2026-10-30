"""Scores on traces, spans, and sessions, as ``POST /v1/scores`` takes them."""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any, cast

from ._api import request, resolve_api
from ._config import report_error
from ._serialize import utf16_length

USER_FEEDBACK = "user_feedback"
MAX_SCORE_NAME_LENGTH = 64
MAX_LABEL_LENGTH = 64
MAX_REASON_LENGTH = 4000
_MAX_SESSION_ID_LENGTH = 1024
_TRACE_ID = re.compile(r"(?!0+$)[0-9a-f]{32}")
_SPAN_ID = re.compile(r"(?!0+$)[0-9a-f]{16}")
_TIMEOUT_S = 10.0


class ScoreError(Exception):
    """Why score failed. ``code`` is the code of the API error (``insufficient_scope``,
    ``invalid_api_key``, ...), or ``missing_api_key``, ``invalid_score``, ``timeout``,
    ``network_error``, or ``invalid_response``. A response with a different status and no API
    error gives ``http_<status>``. ``status`` is the HTTP status of the response."""

    def __init__(self, code: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def score(
    *,
    name: str,
    trace_id: str | None = None,
    span_id: str | None = None,
    session_id: str | None = None,
    value: float | None = None,
    label: str | None = None,
    comment: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str | None:
    """Record a score on a trace, a span, or a session, for example the feedback of a user on a
    reply. ``user_feedback`` is a thumbs up (1) or down (0). Pass ``trace_id``, ``session_id``, or
    the two, and ``value``, ``label``, or the two.

    Like the telemetry functions, score never raises: it returns the id of the score, or None
    after it sends the error to ``on_error``. It makes one request, with no retry, so that a call
    in a request handler waits 10 seconds at most for each socket operation. The trace id of a
    span from a client that is not enabled is all zeros, and score reports it as
    ``invalid_score``.
    """
    api = resolve_api(api_key, base_url)
    try:
        problem = _check_score(name, trace_id, span_id, session_id, value, label, comment)
        if problem is not None:
            raise ScoreError("invalid_score", problem)
        if not api.api_key:
            raise ScoreError(
                "missing_api_key", "No API key: pass api_key or set TELEMETRY_DEV_API_KEY"
            )
        fields = {
            "traceId": trace_id,
            "spanId": span_id,
            "sessionId": session_id,
            "name": name.strip(),
            "value": value,
            "label": None if label is None else label.strip(),
            "comment": comment,
        }
        body = request(
            api,
            ScoreError,
            "POST",
            "/v1/scores",
            body={key: field for key, field in fields.items() if field is not None},
            timeout_s=_TIMEOUT_S,
            idempotent=False,
            attempts=1,
        )
        score_id = cast("dict[str, Any]", body).get("id") if isinstance(body, dict) else None
        if not isinstance(score_id, str):
            raise ScoreError("invalid_response", "telemetry.dev returned a score it cannot read")
        return score_id
    except Exception as exc:
        report_error(api.on_error, "sending the score failed", exc)
        return None


async def score_async(
    *,
    name: str,
    trace_id: str | None = None,
    span_id: str | None = None,
    session_id: str | None = None,
    value: float | None = None,
    label: str | None = None,
    comment: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str | None:
    """``score`` in a worker thread, so that the request does not block the event loop."""
    return await asyncio.to_thread(
        score,
        name=name,
        trace_id=trace_id,
        span_id=span_id,
        session_id=session_id,
        value=value,
        label=label,
        comment=comment,
        api_key=api_key,
        base_url=base_url,
    )


def check_value(name: str, value: object, label: object) -> str | None:
    """Why the server would reject the value and the label of a score, or None. A pass/fail score
    has the label ``pass`` with the value 1, or ``fail`` with 0."""
    if value is not None and not is_finite(value):
        return "value must be a finite number"
    if label is not None and not (
        isinstance(label, str) and 0 < utf16_length(label) <= MAX_LABEL_LENGTH
    ):
        return f"label must be a string of 1 to {MAX_LABEL_LENGTH} characters"
    if value is None and label is None:
        return "Pass a value or a label"
    if name == USER_FEEDBACK and not (label is None and value in (0, 1)):
        return f"{USER_FEEDBACK} takes the value 1 for up or 0 for down, and no label"
    if label == "pass" and value is not None and value != 1:
        return "The label pass goes with the value 1"
    if label == "fail" and value is not None and value != 0:
        return "The label fail goes with the value 0"
    return None


def _check_score(
    name: object,
    trace_id: object,
    span_id: object,
    session_id: object,
    value: object,
    label: object,
    comment: object,
) -> str | None:
    if not isinstance(name, str) or not 0 < utf16_length(name.strip()) <= MAX_SCORE_NAME_LENGTH:
        return f"name must be a string of 1 to {MAX_SCORE_NAME_LENGTH} characters"
    if trace_id is None and session_id is None:
        return "Pass trace_id or session_id"
    if trace_id is not None and not (isinstance(trace_id, str) and _TRACE_ID.fullmatch(trace_id)):
        return "trace_id must be 32 lowercase hex digits and not all zeros"
    if span_id is not None and not (isinstance(span_id, str) and _SPAN_ID.fullmatch(span_id)):
        return "span_id must be 16 lowercase hex digits and not all zeros"
    if span_id is not None and trace_id is None:
        return "Pass the trace_id of the span"
    if session_id is not None and not (
        isinstance(session_id, str) and 0 < utf16_length(session_id) <= _MAX_SESSION_ID_LENGTH
    ):
        return f"session_id must be a string of 1 to {_MAX_SESSION_ID_LENGTH} characters"
    if comment is not None and not (
        isinstance(comment, str) and utf16_length(comment) <= MAX_REASON_LENGTH
    ):
        return f"comment must be a string of at most {MAX_REASON_LENGTH} characters"
    return check_value(name.strip(), value, label.strip() if isinstance(label, str) else label)


def is_finite(value: object) -> bool:
    # True and False are ints in Python, but JSON sends them as booleans, which are not numbers.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        # An int too large for a float, which JavaScript reads as Infinity.
        return False
