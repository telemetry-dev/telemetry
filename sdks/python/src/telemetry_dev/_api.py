"""Calls to the telemetry.dev API: the key and base URL, one JSON request, and its errors."""

from __future__ import annotations

import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
from email.utils import parsedate_to_datetime
from typing import IO, Any, Literal, TypeVar, cast

from ._client import get_client
from ._config import resolve_config

# A longer Retry-After waits only this long: the API counts requests by the minute.
_MAX_RETRY_AFTER_S = 60.0
# IMF-fixdate, the HTTP date that servers send: Sun, 06 Nov 1994 08:49:37 GMT.
_HTTP_DATE = re.compile(r"[A-Z][a-z]{2}, \d{2} [A-Z][a-z]{2} \d{4} \d{2}:\d{2}:\d{2} GMT", re.ASCII)


# An error class with the shape of PromptError: a code, a message, and the status.
E = TypeVar("E", bound=Exception)


@dataclass(frozen=True)
class Api:
    api_key: str | None
    base_url: str
    on_error: Callable[[BaseException], None] | None


def resolve_api(api_key: str | None, base_url: str | None) -> Api:
    """The key and base URL of a call: its arguments, then the enabled client, then the env."""
    client = get_client()
    enabled = client if client is not None and client.enabled else None
    config = enabled.config if enabled is not None else resolve_config()
    return Api(
        api_key=api_key if api_key is not None else config.api_key,
        base_url=(base_url or config.base_url).rstrip("/"),
        on_error=enabled.on_error if enabled is not None else None,
    )


@dataclass(frozen=True)
class _Failure:
    error: Exception
    status: int | None
    retry_after: float | None


def request(
    api: Api,
    error_type: Callable[[str, str, int | None], E],
    method: Literal["GET", "POST"],
    path: str,
    *,
    timeout_s: float,
    idempotent: bool,
    body: object = None,
    query: Mapping[str, str | None] | None = None,
    attempts: int = 3,
) -> Any:
    """Send one JSON request and return the JSON of the response.

    A response that is not 2xx raises the ``{"error": {"code", "message"}}`` of its body, or
    ``http_<status>``. With ``idempotent``, the server takes a repeat at most once, so a timeout, a
    network error, or a 5xx tries again. Other requests try again only after a 429 or a 503: the
    server did not take the request then. A retry waits for the Retry-After of the response, or
    for 1 and then 2 seconds. The timeout applies to each socket operation of urllib.
    """
    url = f"{api.base_url}{path}"
    params = {key: value for key, value in (query or {}).items() if value is not None}
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    headers = {"Authorization": f"Bearer {api.api_key}", "X-Telemetry-Dev-SDK": "telemetry-dev"}
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        headers["Content-Type"] = "application/json"
    attempt = 1
    while True:
        result = _send(error_type, method, url, path, data, headers, timeout_s)
        if not isinstance(result, _Failure):
            return result
        if attempt >= attempts or not _retryable(result.status, idempotent):
            raise result.error
        time.sleep(min(result.retry_after or float(attempt), _MAX_RETRY_AFTER_S))
        attempt += 1


class NoRedirect(urllib.request.HTTPRedirectHandler):
    # A redirect to another host must not get the API key, so a 3xx fails like any status that is
    # not 2xx.
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


opener = urllib.request.build_opener(NoRedirect)


def api_error(error_type: Callable[[str, str, int | None], E], status: int, body: bytes) -> E:
    """The error of a response that is not 2xx: its API code and message, or ``http_<status>``."""
    try:
        error = json.loads(body).get("error")
    except (ValueError, AttributeError):
        error = None
    fields = cast("dict[str, Any]", error) if isinstance(error, dict) else {}
    code = fields.get("code")
    message = fields.get("message")
    return error_type(
        code if isinstance(code, str) else f"http_{status}",
        message if isinstance(message, str) else f"telemetry.dev returned {status}",
        status,
    )


def retry_after(status: int, value: str | None) -> float | None:
    """The seconds that the Retry-After of a 429 or a 503 asks for: seconds or an HTTP date."""
    value = (value or "").strip()
    if status not in (429, 503) or not value:
        return None
    if value.isascii() and value.isdigit():
        seconds = float(value)
    elif _HTTP_DATE.fullmatch(value):
        try:
            date = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        seconds = (date - datetime.now(timezone.utc)).total_seconds()
    else:
        return None
    return seconds if seconds > 0 else None


def _send(
    error_type: Callable[[str, str, int | None], E],
    method: str,
    url: str,
    path: str,
    data: bytes | None,
    headers: dict[str, str],
    timeout_s: float,
) -> Any:
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with opener.open(request, timeout=timeout_s) as response:
            body: bytes = response.read()
    except urllib.error.HTTPError as exc:
        with exc:
            return _Failure(
                api_error(error_type, exc.code, exc.read()),
                exc.code,
                retry_after(exc.code, exc.headers.get("Retry-After")),
            )
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        timed_out = isinstance(exc, TimeoutError) or isinstance(
            getattr(exc, "reason", None), TimeoutError
        )
        error = (
            error_type("timeout", f"{method} {path} took more than {timeout_s:g} s", None)
            if timed_out
            else error_type("network_error", "Could not reach telemetry.dev", None)
        )
        error.__cause__ = exc
        return _Failure(error, None, None)
    try:
        return json.loads(body)
    except ValueError:
        return None


def _retryable(status: int | None, idempotent: bool) -> bool:
    if status in (429, 503):
        return True
    return idempotent and (status is None or status >= 500)
