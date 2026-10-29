from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import random
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import Message
from email.utils import parsedate_to_datetime
from time import monotonic
from typing import IO, Any, Literal, TypeAlias, TypedDict, cast, overload
from weakref import WeakKeyDictionary

from opentelemetry import context as otel_context
from opentelemetry.context import Context

from ._client import Client, get_client
from ._config import SDK_VERSION, report_error, resolve_config
from ._context import propagated_attributes
from ._semconv import (
    ATTR_PROMPT_LABEL,
    ATTR_PROMPT_NAME,
    ATTR_PROMPT_VARIABLES,
    ATTR_PROMPT_VERSION,
    ATTR_SESSION_ID,
)
from ._serialize import AttributeValue, json_dumps

_FETCH_TIMEOUT_S = 10.0
# How long a failed fetch with no copy answers calls, so that they do not each fetch again.
_ERROR_TTL_S = 10.0
# The longest Retry-After that get_prompt honors, so that a bad header cannot stop fetches.
_MAX_RETRY_AFTER_S = 3600.0
_MAX_CACHE_ENTRIES = 1000
# IMF-fixdate, the HTTP date that servers send: Sun, 06 Nov 1994 08:49:37 GMT.
_HTTP_DATE = re.compile(r"[A-Z][a-z]{2}, \d{2} [A-Z][a-z]{2} \d{4} \d{2}:\d{2}:\d{2} GMT", re.ASCII)
_VARIABLE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_MODEL_CALL_OPERATIONS = frozenset({"chat", "text_completion", "generate_content"})
_PROMPT_KEY = otel_context.create_key("telemetry_dev.prompt")


class PromptModel(TypedDict):
    provider: str
    name: str


class PromptError(Exception):
    """Why get_prompt failed. ``code`` is the code of the API error (``prompt_not_found``,
    ``label_not_found``, ``insufficient_scope``, ``invalid_api_key``, ...), or
    ``missing_api_key``, ``timeout``, ``network_error``, ``invalid_response``, or
    ``type_mismatch``. A response with a different status and no API error gives
    ``http_<status>``, for example ``http_302`` for a redirect: get_prompt does not go to the
    location of a redirect. ``status`` is the HTTP status of the response."""

    def __init__(self, code: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        # The Retry-After of a 429 or 503 response, in seconds.
        self._retry_after: float | None = None


@dataclass(frozen=True, eq=False)
class ChatPrompt:
    """A chat prompt version. ``compile()`` makes the messages to send to the model."""

    name: str
    version: int | None
    """The version that the app got; None for the fallback."""
    label: str | None
    """The label that the prompt was fetched by; None when it was fetched by version."""
    messages: list[dict[str, Any]]
    model: PromptModel | None
    parameters: dict[str, Any]
    tools: list[dict[str, Any]]
    response_format: dict[str, Any]
    is_fallback: bool
    """True when get_prompt could not fetch the prompt and returned the ``fallback``."""
    type: Literal["chat"] = field(default="chat", init=False)
    text: None = field(default=None, init=False)

    def compile(
        self, variables: Mapping[str, Any] | None = None, /, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Fill in the variables and replace each placeholder message with the list of messages
        in its variable. ``with_prompt`` records the variables of the last call before it starts
        or inside it."""
        values = _record(self, variables, kwargs)
        compiled: list[dict[str, Any]] = []
        for message in self.messages:
            if message.get("role") != "placeholder":
                compiled.append(
                    {"role": message["role"], "content": _fill(str(message["content"]), values)}
                )
                continue
            name = str(message["name"])
            value = values.get(name)
            if value is None:
                continue
            if not isinstance(value, list | tuple):
                raise TypeError(f"The {name} variable must be a list of messages")
            compiled.extend(cast("list[dict[str, Any]] | tuple[dict[str, Any], ...]", value))
        return compiled


@dataclass(frozen=True, eq=False)
class TextPrompt:
    """A text prompt version. ``compile()`` makes the text to send to the model."""

    name: str
    version: int | None
    """The version that the app got; None for the fallback."""
    label: str | None
    """The label that the prompt was fetched by; None when it was fetched by version."""
    text: str
    model: PromptModel | None
    parameters: dict[str, Any]
    tools: list[dict[str, Any]]
    response_format: dict[str, Any]
    is_fallback: bool
    """True when get_prompt could not fetch the prompt and returned the ``fallback``."""
    type: Literal["text"] = field(default="text", init=False)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return []

    def compile(self, variables: Mapping[str, Any] | None = None, /, **kwargs: Any) -> str:
        """Fill in the variables. ``with_prompt`` records the variables of the last call before it
        starts or inside it."""
        return _fill(self.text, _record(self, variables, kwargs))


Prompt: TypeAlias = ChatPrompt | TextPrompt


@overload
def get_prompt(
    name: str,
    *,
    type: Literal["chat"] = "chat",
    label: str | None = None,
    version: int | None = None,
    cache_ttl_seconds: float = 60,
    fallback: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ChatPrompt: ...


@overload
def get_prompt(
    name: str,
    *,
    type: Literal["text"],
    label: str | None = None,
    version: int | None = None,
    cache_ttl_seconds: float = 60,
    fallback: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> TextPrompt: ...


def get_prompt(
    name: str,
    *,
    type: Literal["chat", "text"] = "chat",
    label: str | None = None,
    version: int | None = None,
    cache_ttl_seconds: float = 60,
    fallback: list[dict[str, Any]] | str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Prompt:
    """Fetch a prompt version from telemetry.dev and keep it.

    Without ``label`` and ``version`` it fetches the ``production`` label. When the copy is older
    than ``cache_ttl_seconds``, a call returns it at once and fetches a new one in the background.
    Only a call with no copy waits for a fetch, at most 10 seconds. Calls for the same prompt share
    one fetch. When a fetch fails, it returns the last copy that it kept; with no copy it returns
    ``fallback``. With neither it raises ``PromptError``: unlike the telemetry functions,
    get_prompt cannot fail open, because the app cannot run without the prompt. With no copy, the
    calls of the next 10 seconds (or the TTL, if it is shorter) fail the same way without a fetch.

    A label with a gradual rollout gives each session one version: the choice uses the session
    id of ``propagate_attributes``, or a random choice per call without one.
    """
    return _get_prompt(name, type, label, version, cache_ttl_seconds, fallback, api_key, base_url)


@overload
async def get_prompt_async(
    name: str,
    *,
    type: Literal["chat"] = "chat",
    label: str | None = None,
    version: int | None = None,
    cache_ttl_seconds: float = 60,
    fallback: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ChatPrompt: ...


@overload
async def get_prompt_async(
    name: str,
    *,
    type: Literal["text"],
    label: str | None = None,
    version: int | None = None,
    cache_ttl_seconds: float = 60,
    fallback: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> TextPrompt: ...


async def get_prompt_async(
    name: str,
    *,
    type: Literal["chat", "text"] = "chat",
    label: str | None = None,
    version: int | None = None,
    cache_ttl_seconds: float = 60,
    fallback: list[dict[str, Any]] | str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Prompt:
    """``get_prompt`` in a worker thread, so a fetch does not block the event loop."""
    return await asyncio.to_thread(
        _get_prompt, name, type, label, version, cache_ttl_seconds, fallback, api_key, base_url
    )


@contextmanager
def with_prompt(prompt: Prompt) -> Generator[None]:
    """Record ``prompt`` on each model call (generation span) started inside this context:
    ``td.prompt.name``, ``td.prompt.version``, ``td.prompt.label``, and, when input capture is on,
    ``td.prompt.variables``. The variables are those of the last ``compile()`` of this prompt
    object before the context starts, or of the last ``compile()`` inside it, so a compile in
    another thread or task does not change them once the context runs. When concurrent tasks in
    the context each compile, a span records the last compile of its own task; a compile in a
    worker thread (``asyncio.to_thread``) counts only for a task that has not compiled."""
    scope = _PromptScope(prompt, _compiled_variables.get(prompt))
    token = otel_context.attach(otel_context.set_value(_PROMPT_KEY, scope))
    task_token = _task_variables.set(_task_variables.get())
    try:
        yield
    finally:
        try:
            _task_variables.reset(task_token)
        except ValueError:
            # The block ended in another Context, as a generator closed elsewhere does. Keep
            # going like otel_context.detach does, so that the detach below still runs.
            pass
        otel_context.detach(token)


def prompt_attributes(
    client: Client, context: Context | None, operation: object, capture_input: bool
) -> dict[str, AttributeValue]:
    """The td.prompt.* attributes for a span with this operation that starts in ``context``."""
    if operation not in _MODEL_CALL_OPERATIONS:
        return {}
    scope = otel_context.get_value(_PROMPT_KEY, context) or otel_context.get_value(_PROMPT_KEY)
    if not isinstance(scope, _PromptScope):
        return {}
    prompt = scope.prompt
    attrs: dict[str, AttributeValue] = {ATTR_PROMPT_NAME: prompt.name}
    if prompt.version is not None:
        attrs[ATTR_PROMPT_VERSION] = prompt.version
    if prompt.label is not None:
        attrs[ATTR_PROMPT_LABEL] = prompt.label
    task = _task_variables.get()
    variables = task[1] if task is not None and task[0] is scope else scope.variables
    if variables is not None and capture_input:
        serialized = client.serialize(variables, ATTR_PROMPT_VARIABLES)
        if serialized is not None:
            attrs[ATTR_PROMPT_VARIABLES] = serialized
    return attrs


def _get_prompt(
    name: str,
    type: Literal["chat", "text"],
    label: str | None,
    version: int | None,
    cache_ttl_seconds: float,
    fallback: list[dict[str, Any]] | str | None,
    api_key: str | None,
    base_url: str | None,
) -> Prompt:
    _check(name, type, label, version, cache_ttl_seconds, fallback)
    client = get_client()
    enabled = client if client is not None and client.enabled else None
    config = enabled.config if enabled is not None else resolve_config()
    key = api_key if api_key is not None else config.api_key
    url = (base_url or config.base_url).rstrip("/")
    asked_label = (label if label is not None else "production") if version is None else None
    on_error = enabled.on_error if enabled is not None else None

    try:
        if not key:
            raise PromptError(
                "missing_api_key", "No API key: pass api_key or set TELEMETRY_DEV_API_KEY"
            )
        query = {"name": name}
        if asked_label is None:
            query["version"] = str(version)
        else:
            query["label"] = asked_label
        response = _load(
            (
                url,
                hashlib.sha256(key.encode()).hexdigest()[:32],
                name,
                f"version:{version}" if asked_label is None else f"label:{asked_label}",
            ),
            cache_ttl_seconds,
            lambda: _fetch(
                f"{url}/v1/prompts?{urllib.parse.urlencode(query)}",
                {
                    "Authorization": f"Bearer {key}",
                    "X-Telemetry-Dev-SDK": "telemetry-dev",
                    # urllib's default User-Agent, Python-urllib/3.x, is refused by bot filters
                    # such as Cloudflare's Browser Integrity Check.
                    "User-Agent": f"telemetry-dev-python/{SDK_VERSION}",
                },
            ),
            on_error,
        )
    except Exception as exc:
        if fallback is None:
            raise
        report_error(on_error, "fetching the prompt failed; using the fallback", exc)
        content: dict[str, Any] = {
            "messages" if type == "chat" else "text": fallback,
            "model": None,
            "parameters": {},
            "tools": [],
            "responseFormat": {"type": "text"},
        }
        return _make_prompt(name, type, asked_label, None, content, True)

    if response["type"] != type:
        raise PromptError(
            "type_mismatch",
            f'{name} is a {response["type"]} prompt; pass type="{response["type"]}"',
        )
    rollout = response.get("rollout")
    chosen: Mapping[str, Any] = response
    if rollout is not None and _in_rollout(response, rollout["percent"], enabled):
        chosen = rollout
    return _make_prompt(response["name"], type, response["label"], chosen["version"], chosen, False)


_compiled_variables: WeakKeyDictionary[ChatPrompt | TextPrompt, dict[str, Any]] = (
    WeakKeyDictionary()
)


@dataclass(eq=False)
class _PromptScope:
    """The prompt of a ``with_prompt`` context and the variables of the last ``compile()`` in it.
    Concurrent tasks in the context share the scope; see ``_task_variables``."""

    prompt: Prompt
    variables: dict[str, Any] | None


# The variables of the last compile() of a scope in the current task. A new asyncio task or a
# copied context gets a copy, so its compile does not change the value of other tasks.
_task_variables: ContextVar[tuple[_PromptScope, dict[str, Any]] | None] = ContextVar(
    "telemetry_dev_prompt_variables", default=None
)


def _record(
    prompt: Prompt, variables: Mapping[str, Any] | None, kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    values = {**(variables or {}), **kwargs}
    _compiled_variables[prompt] = values
    scope = otel_context.get_value(_PROMPT_KEY)
    if isinstance(scope, _PromptScope) and scope.prompt is prompt:
        scope.variables = values
        _task_variables.set((scope, values))
    return values


def _fill(text: str, values: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        value = values.get(match.group(1))
        if value is None:
            return ""
        return value if isinstance(value, str) else json_dumps(value)

    return _VARIABLE.sub(replace, text)


def _check(
    name: object,
    type: object,
    label: str | None,
    version: object,
    cache_ttl_seconds: object,
    fallback: object,
) -> None:
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty string")
    if type not in ("chat", "text"):
        raise TypeError('type must be "chat" or "text"')
    if label is not None and version is not None:
        raise TypeError("Pass label or version, not both")
    if version is not None and (
        isinstance(version, bool) or not isinstance(version, int) or version < 1
    ):
        raise ValueError("version must be a positive integer")
    if (
        isinstance(cache_ttl_seconds, bool)
        or not isinstance(cache_ttl_seconds, int | float)
        or not 0 <= cache_ttl_seconds < float("inf")
    ):
        raise ValueError("cache_ttl_seconds must be a non-negative number")
    if fallback is not None and not isinstance(fallback, list if type == "chat" else str):
        raise TypeError(
            "fallback must be a list of messages" if type == "chat" else "fallback must be a string"
        )


@dataclass
class _Entry:
    response: dict[str, Any] | None = None
    # Why the last fetch failed, when there was no response to keep.
    error: Exception | None = None
    # When the last fetch ended.
    checked_at: float = 0.0
    # No fetch before this time: the Retry-After of the last 429 or 503 response.
    retry_at: float = 0.0
    # Set when the running fetch ends.
    pending: threading.Event | None = None
    # When calls stop waiting for the running fetch.
    deadline: float = 0.0


_cache: OrderedDict[tuple[str, str, str, str], _Entry] = OrderedDict()
_cache_lock = threading.Lock()


def _load(
    key: tuple[str, str, str, str],
    ttl_seconds: float,
    fetch: Callable[[], dict[str, Any]],
    on_error: Callable[[BaseException], None] | None,
) -> dict[str, Any]:
    """Return the kept copy of a prompt, and fetch a new copy in a background thread when it is
    older than the TTL. Only a call with no copy waits for a fetch, until the fetch's deadline: the
    timeout of urllib applies to each socket operation, not to the whole fetch. A failed fetch with
    no copy fails the calls of the next ``_ERROR_TTL_S`` (or the TTL, if it is shorter) without a
    fetch."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            if len(_cache) >= _MAX_CACHE_ENTRIES:
                _cache.popitem(last=False)
            entry = _cache[key] = _Entry()
        now = monotonic()
        ttl = ttl_seconds if entry.response is not None else min(ttl_seconds, _ERROR_TTL_S)
        if now >= max(entry.checked_at + ttl, entry.retry_at) and entry.pending is None:
            entry.pending = threading.Event()
            entry.deadline = now + _FETCH_TIMEOUT_S
            threading.Thread(
                target=_refresh, args=(entry, entry.pending, fetch, on_error), daemon=True
            ).start()
        if entry.response is not None:
            return entry.response
        pending, deadline = entry.pending, entry.deadline
    if pending is not None and not pending.wait(deadline - monotonic()):
        raise PromptError("timeout", f"Fetching the prompt took more than {_FETCH_TIMEOUT_S:g} s")
    with _cache_lock:
        response, error = entry.response, entry.error
    if response is not None:
        return response
    assert error is not None  # A fetch that ends with no copy keeps its error.
    # Raise a new exception: each raise of one exception object adds to its traceback.
    if isinstance(error, PromptError):
        raise PromptError(error.code, error.message, error.status) from error
    raise PromptError("network_error", "Could not reach telemetry.dev") from error


def _refresh(
    entry: _Entry,
    done: threading.Event,
    fetch: Callable[[], dict[str, Any]],
    on_error: Callable[[BaseException], None] | None,
) -> None:
    response: dict[str, Any] | None = None
    error: Exception | None = None
    try:
        response = fetch()
    except Exception as exc:
        error = exc
    with _cache_lock:
        stale = entry.response
        delay = (
            error._retry_after  # pyright: ignore[reportPrivateUsage]
            if isinstance(error, PromptError)
            else None
        )
        entry.retry_at = 0.0 if delay is None else monotonic() + delay
        if response is not None:
            entry.response = response
        elif stale is None:
            entry.error = error
        # A failed fetch counts too, so that calls do not each start a fetch while it fails.
        entry.checked_at = monotonic()
        entry.pending = None
    if error is not None and stale is not None:
        report_error(on_error, "fetching the prompt failed; using the last copy", error)
    done.set()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
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


_opener = urllib.request.build_opener(_NoRedirect)


def _fetch(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with _opener.open(request, timeout=_FETCH_TIMEOUT_S) as response:
            body: bytes = response.read()
    except urllib.error.HTTPError as exc:
        with exc:
            error = _api_error(exc.code, exc.read())
            if exc.code in (429, 503):
                retry_after = _retry_after(exc.headers.get("Retry-After"))
                error._retry_after = retry_after  # pyright: ignore[reportPrivateUsage]
            raise error from None
    except TimeoutError as exc:
        raise PromptError(
            "timeout", f"Fetching the prompt took more than {_FETCH_TIMEOUT_S:g} s"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        if isinstance(getattr(exc, "reason", None), TimeoutError):
            raise PromptError(
                "timeout", f"Fetching the prompt took more than {_FETCH_TIMEOUT_S:g} s"
            ) from exc
        raise PromptError("network_error", "Could not reach telemetry.dev") from exc
    try:
        parsed = _parse_response(json.loads(body))
    except ValueError:
        parsed = None
    if parsed is None:
        raise PromptError("invalid_response", "telemetry.dev returned a prompt it cannot read")
    return parsed


def _api_error(status: int, body: bytes) -> PromptError:
    try:
        error = json.loads(body).get("error")
    except (ValueError, AttributeError):
        error = None
    fields = cast("dict[str, Any]", error) if isinstance(error, dict) else {}
    code = fields.get("code")
    message = fields.get("message")
    return PromptError(
        code if isinstance(code, str) else f"http_{status}",
        message if isinstance(message, str) else f"telemetry.dev returned {status}",
        status,
    )


def _retry_after(value: str | None) -> float | None:
    """Seconds to wait from a Retry-After header of seconds or an IMF-fixdate HTTP date
    (``Sun, 06 Nov 1994 08:49:37 GMT``), capped."""
    value = (value or "").strip()
    if not value:
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
    return min(seconds, _MAX_RETRY_AFTER_S) if seconds > 0 else None


def _parse_response(data: object) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    body = cast("dict[str, Any]", data)
    prompt_type = body.get("type")
    label = body.get("label")
    if (
        not isinstance(body.get("name"), str)
        or prompt_type not in ("chat", "text")
        or "label" not in body
        or not (label is None or isinstance(label, str))
        or not _is_version(body.get("version"))
        or not _valid_content(body, prompt_type)
    ):
        return None
    rollout = body.get("rollout")
    if rollout is None:
        return body
    if not isinstance(rollout, dict):
        return None
    rollout_body = cast("dict[str, Any]", rollout)
    percent = rollout_body.get("percent")
    if (
        not _is_version(rollout_body.get("version"))
        or isinstance(percent, bool)
        or not isinstance(percent, int)
        or not 0 <= percent <= 100
        or not _valid_content(rollout_body, prompt_type)
    ):
        return None
    return body


def _valid_content(body: Mapping[str, Any], prompt_type: str) -> bool:
    messages = body.get("messages")
    model = body.get("model")
    tools = body.get("tools")
    response_format = body.get("responseFormat")
    valid_body = (
        isinstance(messages, list) and all(_is_message(m) for m in cast("list[Any]", messages))
        if prompt_type == "chat"
        else isinstance(body.get("text"), str)
    )
    return (
        valid_body
        and isinstance(model, dict)
        and isinstance(cast("dict[str, Any]", model).get("provider"), str)
        and isinstance(cast("dict[str, Any]", model).get("name"), str)
        and isinstance(body.get("parameters"), dict)
        and isinstance(tools, list)
        and all(
            isinstance(tool, dict) and isinstance(cast("dict[str, Any]", tool).get("name"), str)
            for tool in cast("list[Any]", tools)
        )
        and isinstance(response_format, dict)
        and isinstance(cast("dict[str, Any]", response_format).get("type"), str)
    )


def _is_message(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    message = cast("dict[str, Any]", value)
    if message.get("role") == "placeholder":
        return isinstance(message.get("name"), str)
    return message.get("role") in ("system", "user", "assistant") and isinstance(
        message.get("content"), str
    )


def _is_version(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _make_prompt(
    name: str,
    prompt_type: Literal["chat", "text"],
    label: str | None,
    version: int | None,
    content: Mapping[str, Any],
    is_fallback: bool,
) -> Prompt:
    # Each prompt gets its own copy, so an app cannot change the cached response through it.
    model: PromptModel | None = copy.deepcopy(content["model"])
    parameters: dict[str, Any] = copy.deepcopy(content["parameters"])
    tools: list[dict[str, Any]] = copy.deepcopy(content["tools"])
    response_format: dict[str, Any] = copy.deepcopy(content["responseFormat"])
    if prompt_type == "chat":
        return ChatPrompt(
            name=name,
            version=version,
            label=label,
            messages=copy.deepcopy(content["messages"]),
            model=model,
            parameters=parameters,
            tools=tools,
            response_format=response_format,
            is_fallback=is_fallback,
        )
    return TextPrompt(
        name=name,
        version=version,
        label=label,
        text=content["text"],
        model=model,
        parameters=parameters,
        tools=tools,
        response_format=response_format,
        is_fallback=is_fallback,
    )


def _in_rollout(response: Mapping[str, Any], percent: int, client: Client | None) -> bool:
    session_id = propagated_attributes().get(ATTR_SESSION_ID)
    if not isinstance(session_id, str) and client is not None:
        session_id = client._process_session_id  # pyright: ignore[reportPrivateUsage]
    if not isinstance(session_id, str):
        return random.randrange(100) < percent
    return _fnv1a(f"{response['name']}:{response['label']}:{session_id}") % 100 < percent


def _fnv1a(text: str) -> int:
    """FNV-1a 32-bit over the UTF-8 bytes, as TextEncoder makes them in the TypeScript SDK."""
    data = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace").encode()
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value
