from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

import telemetry_dev
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from botocore.response import StreamingBody

from ._fields import error_fields, merge_fields, metadata_fields
from ._invoke_model import parse_body
from ._registry import OperationSpec, lookup_operation
from ._streams import InstrumentedEventStream

_T = TypeVar("_T", bound=BaseClient)
_SUPPORTED_SERVICES = frozenset({"bedrock-runtime", "bedrock-agent-runtime"})
_WRAPPED_ATTR = "_telemetry_dev_bedrock_wrapped"

_original_api_call: Callable[..., Any] | None = None
_patched_api_call: Callable[..., Any] | None = None
_LOCK = threading.Lock()


def wrap_bedrock(client: _T, *, capture_agent_trace: bool = False) -> _T:
    with _LOCK:
        if getattr(client, _WRAPPED_ATTR, False):
            return client
        mapping = getattr(client.meta, "method_to_api_mapping", {})
        if not isinstance(mapping, dict):
            return client
        for method_name, operation_name in mapping.items():
            if not isinstance(method_name, str) or not isinstance(operation_name, str):
                continue
            service_name = client.meta.service_model.service_name
            if lookup_operation(service_name, operation_name) is None or not hasattr(
                client, method_name
            ):
                continue
            original = getattr(client, method_name)
            setattr(
                client,
                method_name,
                _make_method_wrapper(client, operation_name, original, capture_agent_trace),
            )
        setattr(client, _WRAPPED_ATTR, True)
        return client


def instrument_bedrock(*, capture_agent_trace: bool = False) -> None:
    with _LOCK:
        global _original_api_call, _patched_api_call
        if _original_api_call is not None:
            return
        original = BaseClient._make_api_call
        _original_api_call = original

        def patched(self: BaseClient, operation_name: str, api_params: dict[str, Any]) -> Any:
            if getattr(self, _WRAPPED_ATTR, False):
                return original(self, operation_name, api_params)
            return _call_with_span(
                lambda op_name, params: original(self, op_name, params),
                self,
                operation_name,
                api_params,
                capture_agent_trace,
            )

        _patched_api_call = patched
        BaseClient._make_api_call = patched  # type: ignore[method-assign]


def uninstrument_bedrock() -> None:
    with _LOCK:
        global _original_api_call, _patched_api_call
        if _original_api_call is None or _patched_api_call is None:
            return
        if BaseClient._make_api_call is not _patched_api_call:
            return
        BaseClient._make_api_call = _original_api_call  # type: ignore[method-assign]
        _original_api_call = None
        _patched_api_call = None


def _make_method_wrapper(
    client: BaseClient,
    operation_name: str,
    original: Callable[..., Any],
    capture_agent_trace: bool,
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if args:
            return original(*args, **kwargs)
        return _call_with_span(
            lambda _op, params: original(**params),
            client,
            operation_name,
            kwargs,
            capture_agent_trace,
        )

    return wrapped


def _call_with_span(
    original: Callable[[str, dict[str, Any]], Any],
    client: BaseClient,
    operation_name: str,
    api_params: dict[str, Any],
    capture_agent_trace: bool,
) -> Any:
    service_name = client.meta.service_model.service_name
    if service_name not in _SUPPORTED_SERVICES:
        return original(operation_name, api_params)
    spec = lookup_operation(service_name, operation_name)
    if spec is None:
        return original(operation_name, api_params)

    started_at = time.perf_counter()
    name, start_fields = _safe_start_fields(spec, api_params)
    handle = telemetry_dev.start_span(name, **start_fields)
    end = _end_once(handle)

    try:
        response = original(operation_name, api_params)
    except ClientError as exc:
        end(**error_fields(exc))
        raise
    except BaseException as exc:
        end(error=exc)
        raise

    if not isinstance(response, dict):
        end()
        return response

    if operation_name == "InvokeModel" and isinstance(response.get("body"), StreamingBody):
        return _handle_streaming_body(spec, api_params, response, end)

    stream_key = spec.stream_key
    if stream_key is not None and response.get(stream_key) is not None:
        state_factory = spec.stream_state_factory
        if state_factory is None:
            end(**metadata_fields(response))
            return response
        state = state_factory(capture_agent_trace)
        try:
            base_fields = merge_fields(
                metadata_fields(response), spec.response_fields(api_params, response)
            )
        except Exception:
            base_fields = metadata_fields(response)

        def finish(fields: dict[str, Any]) -> None:
            if fields.keys() == {"time_to_first_chunk_ms"}:
                try:
                    handle.update(**fields)
                except Exception:
                    pass
                return
            end(**merge_fields(base_fields, fields))

        return {
            **response,
            stream_key: InstrumentedEventStream(response[stream_key], state, finish, started_at),
        }

    end(**merge_fields(metadata_fields(response), spec.response_fields(api_params, response)))
    return response


class _TeedRawStream:
    def __init__(self, raw_stream: Any, body: _InstrumentedStreamingBody) -> None:
        self._raw_stream = raw_stream
        self._body = body
        self._pending = b""

    def read(self, amt: int | None = None) -> bytes:
        try:
            if not self._pending or amt == 0:
                chunk = self._raw_stream.read(amt)
            elif amt is None or amt < 0:
                chunk = self._pending + self._raw_stream.read(amt)
                self._pending = b""
            else:
                chunk = self._pending[:amt]
                self._pending = self._pending[amt:]
                if len(chunk) < amt:
                    chunk += self._raw_stream.read(amt - len(chunk))
        except BaseException as exc:
            self._body._finish(complete=False, error=exc)
            raise
        self._capture(chunk)
        if (
            amt is None
            or amt < 0
            or (not chunk and amt > 0)
            or self._body._content_length_reached()
        ):
            self._body._finish(complete=True)
        return chunk

    def readline(self, size: int = -1) -> bytes:
        try:
            if not self._pending or size == 0:
                chunk = self._raw_stream.readline(size)
            else:
                chunk = self._pending
                self._pending = b""
                if chunk != b"\n" and size != 1:
                    chunk += self._raw_stream.readline(-1 if size < 0 else size - 1)
        except BaseException as exc:
            self._body._finish(complete=False, error=exc)
            raise
        self._capture(chunk)
        if (not chunk and size != 0) or self._body._content_length_reached():
            self._body._finish(complete=True)
        return chunk

    def readlines(self, hint: int = -1) -> list[bytes]:
        try:
            if self._pending:
                first = self._pending
                self._pending = b""
                if first != b"\n":
                    first += self._raw_stream.readline()
                remaining_hint = hint - len(first) if hint > 0 else hint
                lines = [first]
                if hint <= 0 or remaining_hint > 0:
                    lines.extend(self._raw_stream.readlines(remaining_hint))
            else:
                lines = self._raw_stream.readlines(hint)
            for line in lines:
                self._capture(line)
            complete = not lines or hint <= 0 or self._body._content_length_reached()
            if not complete:
                self._pending = self._raw_stream.read(1)
                complete = not self._pending
        except BaseException as exc:
            self._body._finish(complete=False, error=exc)
            raise
        if complete:
            self._body._finish(complete=True)
        return lines

    def readinto(self, b: Any) -> int:
        try:
            view = memoryview(b)
            if view.readonly:
                raise TypeError("readinto() argument must be a writable buffer")
            if self._pending:
                byte_view = view.cast("B")
                chunk = self.read(len(byte_view))
                byte_view[: len(chunk)] = chunk
                return len(chunk)
            amount_read = self._raw_stream.readinto(b)
        except BaseException as exc:
            self._body._finish(complete=False, error=exc)
            raise
        self._body._amount_read += amount_read
        if amount_read > 0:
            self._body._capture(view.cast("B")[:amount_read])
        if (amount_read == 0 and len(b) > 0) or self._body._content_length_reached():
            self._body._finish(complete=True)
        return amount_read

    def __iter__(self) -> _TeedRawStream:
        return self

    def __next__(self) -> bytes:
        try:
            if self._pending:
                chunk = self._pending
                self._pending = b""
                try:
                    chunk += next(self._raw_stream)
                except StopIteration:
                    self._capture(chunk)
                    self._body._finish(complete=True)
                    return chunk
            else:
                chunk = next(self._raw_stream)
        except StopIteration:
            self._body._finish(complete=True)
            raise
        except BaseException as exc:
            self._body._finish(complete=False, error=exc)
            raise
        self._capture(chunk)
        if self._body._content_length_reached():
            self._body._finish(complete=True)
        return chunk

    def close(self) -> Any:
        try:
            return self._raw_stream.close()
        finally:
            self._body._finish(complete=False)

    def _capture(self, chunk: bytes | bytearray | memoryview) -> None:
        self._body._amount_read += len(chunk)
        self._body._capture(chunk)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw_stream, name)


class _InstrumentedStreamingBody(StreamingBody):
    def __init__(
        self,
        body: StreamingBody,
        finish: Callable[[bytes | None, BaseException | None], None],
    ) -> None:
        super().__init__(body, getattr(body, "_content_length", None))
        self._body = body
        self._finish_callback = finish
        self._budget = telemetry_dev.CaptureBudget.from_client()
        self._captured = bytearray()
        self._ended = False

    def read(self, amt: int | None = None) -> bytes:
        try:
            chunk = super().read(amt)
        except BaseException as exc:
            self._finish(complete=False, error=exc)
            raise
        self._capture(chunk)
        if amt is None or amt < 0 or (not chunk and amt > 0) or self._content_length_reached():
            self._finish(complete=True)
        return chunk

    def readinto(self, b: Any) -> int:
        try:
            amount_read = super().readinto(b)
        except BaseException as exc:
            self._finish(complete=False, error=exc)
            raise
        if amount_read > 0:
            self._capture(memoryview(b)[:amount_read])
        if (amount_read == 0 and len(b) > 0) or self._content_length_reached():
            self._finish(complete=True)
        return amount_read

    def readlines(self) -> list[bytes]:
        try:
            lines = self._body.readlines()
        except BaseException as exc:
            self._finish(complete=False, error=exc)
            raise
        for line in lines:
            self._capture(line)
            if self._budget.truncated:
                break
        self._finish(complete=True)
        return lines

    def set_socket_timeout(self, timeout: float) -> None:
        self._body.set_socket_timeout(timeout)

    def __enter__(self) -> _TeedRawStream:
        return _TeedRawStream(self._body.__enter__(), self)

    def __exit__(self, type_: object, value: object, traceback: object) -> None:
        try:
            self._body.__exit__(type_, value, traceback)
        finally:
            self._finish(complete=False)

    def close(self) -> None:
        try:
            self._body.close()
        finally:
            self._finish(complete=False)

    def __del__(self) -> None:
        try:
            self._finish(complete=False)
        except Exception:
            pass

    def _capture(self, chunk: bytes | bytearray | memoryview) -> None:
        retained = self._budget.capture_bytes(chunk)
        if retained:
            self._captured.extend(retained)

    def _content_length_reached(self) -> bool:
        if self._content_length is None:
            return False
        try:
            return self._amount_read == int(self._content_length)
        except (TypeError, ValueError):
            return False

    def _finish(self, *, complete: bool, error: BaseException | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        captured = bytes(self._captured) if complete and not self._budget.truncated else None
        self._finish_callback(captured, error)


def _handle_streaming_body(
    spec: OperationSpec,
    params: dict[str, Any],
    response: dict[str, Any],
    end: Callable[..., None],
) -> dict[str, Any]:
    body = cast(StreamingBody, response["body"])
    base_fields = metadata_fields(response)

    def finish(raw: bytes | None, error: BaseException | None) -> None:
        fields = base_fields
        if raw is not None:
            parsed = parse_body(raw, response.get("contentType"))
            response_for_fields = {**response, "_telemetry_dev_parsed_body": parsed}
            try:
                fields = merge_fields(
                    base_fields, spec.response_fields(params, response_for_fields)
                )
            except Exception:
                fields = base_fields
        if error is not None:
            fields = merge_fields(fields, error_fields(error))
        end(**fields)

    return {**response, "body": _InstrumentedStreamingBody(body, finish)}


def _safe_start_fields(spec: OperationSpec, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        name = spec.span_name(params)
    except Exception:
        name = "bedrock"
    try:
        fields = spec.request_fields(params)
    except Exception:
        fields = {}
    try:
        fields = {"type": spec.span_type(params), **fields}
    except Exception:
        fields = {"type": "span", **fields}
    return name, fields


def _end_once(handle: telemetry_dev.SpanHandle) -> Callable[..., None]:
    ended = False

    def end(**fields: Any) -> None:
        nonlocal ended
        if ended:
            return
        ended = True
        try:
            handle.end(**fields)
        except Exception:
            pass

    return end
