from __future__ import annotations

import atexit
import copy
import json
import math
import os
import threading
import weakref
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
from typing import Any, TypeVar, cast

import requests
from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from ._config import logger, report_error
from ._process_lock import ProcessLock
from ._serialize import TRUNCATION_MARKER, truncate, utf16_length

# The largest OTLP request the exporters send. The ingest rejects a body over 16 MiB after gzip
# decoding; keep this equal to DECOMPRESSED_BODY_LIMIT in the ingest. A record that alone is
# larger has its longest values cut at export, so provider integrations keep at most this much
# of one value or stream.
MAX_RECORD_BYTES = 16 * 1024 * 1024
_CAPTURE_TRUNCATED = "telemetry.dev.capture.truncated"

_Record = TypeVar("_Record")
_Result = TypeVar("_Result")


# The key of a log record's body among the values that _fit_values may cut.
_BODY: Any = object()


def _fit_values(
    attributes: Mapping[str, Any],
    size: Callable[[dict[str, Any], Any], int],
    body: Any = None,
) -> tuple[dict[str, Any], Any] | None:
    """Cut the longest string values of a record that alone serializes over the body limit.

    The values are the attributes and, for a log record, its body. Adds the truncation flag to
    the attributes. Returns the fitted attributes and body, or None when no string value is left
    to cut.
    """
    values: dict[Any, Any] = {**attributes, _CAPTURE_TRUNCATED: True, _BODY: body}
    lengths = {key: utf16_length(value) for key, value in values.items() if isinstance(value, str)}
    while True:
        fitted = {key: value for key, value in values.items() if key is not _BODY}
        excess_bytes = size(fitted, values[_BODY]) - MAX_RECORD_BYTES
        if excess_bytes <= 0:
            return fitted, values[_BODY]
        # Cut the excess from the longest values before the next serialization, since each
        # serialization of a record this large is costly. Each UTF-16 unit takes at least one
        # UTF-8 byte, so removing excess_bytes units removes at least the excess. Multi-byte text
        # removes more, so when no value is left to cut, the next serialization decides.
        cut_any = False
        while excess_bytes > 0:
            longest_key = max(lengths, key=lengths.__getitem__, default=None)
            if longest_key is None or lengths[longest_key] <= len(TRUNCATION_MARKER):
                break
            cut = min(excess_bytes, lengths[longest_key] - len(TRUNCATION_MARKER))
            values[longest_key] = truncate(values[longest_key], lengths[longest_key] - cut)
            lengths[longest_key] = utf16_length(values[longest_key])
            excess_bytes -= cut
            cut_any = True
        if not cut_any:
            return None


def _span_with_attributes(span: ReadableSpan, attributes: Mapping[str, Any]) -> ReadableSpan:
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def _fit_span(span: ReadableSpan) -> ReadableSpan | None:
    fitted = _fit_values(
        span.attributes or {},
        lambda attributes, _: encode_spans([_span_with_attributes(span, attributes)]).ByteSize(),
    )
    return None if fitted is None else _span_with_attributes(span, fitted[0])


def _log_with_values(item: Any, attributes: Mapping[str, Any], body: Any) -> Any:
    # Copies keep the caller's record intact. The record type differs across OpenTelemetry
    # releases (LogData, then a frozen ReadableLogRecord), and both hold log_record.
    log_record = copy.copy(item.log_record)
    log_record.attributes = attributes
    log_record.body = body
    fitted = copy.copy(item)
    object.__setattr__(fitted, "log_record", log_record)
    return fitted


def _fit_log(item: Any) -> Any:
    fitted = _fit_values(
        item.log_record.attributes or {},
        lambda attributes, body: encode_logs([_log_with_values(item, attributes, body)]).ByteSize(),
        item.log_record.body,
    )
    return None if fitted is None else _log_with_values(item, *fitted)


# Upper bounds of the protobuf size of OTLP fields, without encoding them. Python counts code
# points, and each takes at most 4 UTF-8 bytes, or 1 in an ASCII string. Each field or message
# also takes a tag and a length of at most 2 and 5 bytes, and a number at most 10; the constants
# round those up.
_FIELD_BOUND = 16
_RECORD_BOUND = 256
_EVENT_BOUND = 64
_LINK_BOUND = 128
_MAX_BOUND_DEPTH = 32


def _text_bound(text: str | None) -> int:
    text = text or ""
    return _FIELD_BOUND + (len(text) if text.isascii() else 4 * len(text))


def _value_bound(value: object, depth: int = 0) -> int | None:
    """At least the encoded size of ``value`` as an OTLP AnyValue, or None for a value that the
    encoder may treat in a way this does not cover."""
    if depth > _MAX_BOUND_DEPTH:
        return None
    if value is None or isinstance(value, bool | int | float):
        return _FIELD_BOUND
    if isinstance(value, str):
        return _text_bound(value)
    if isinstance(value, bytes):
        return _FIELD_BOUND + len(value)
    total = _FIELD_BOUND
    # Sequence first, as the encoder tests them.
    if isinstance(value, Sequence):
        for item in cast("Sequence[object]", value):
            item_bound = _value_bound(item, depth + 1)
            if item_bound is None:
                return None
            total += _FIELD_BOUND + item_bound
        return total
    if isinstance(value, Mapping):
        for key, item in cast("Mapping[object, object]", value).items():
            item_bound = _value_bound(item, depth + 1)
            if not isinstance(key, str) or item_bound is None:
                return None
            total += _FIELD_BOUND + _text_bound(key) + item_bound
        return total
    return None


def _sum_bounds(bounds: Sequence[int | None]) -> int | None:
    return None if None in bounds else sum(cast("Sequence[int]", bounds))


def _origin_bound(resource: Any, scope: Any) -> int | None:
    """The resource and scope of a record. The encoder writes each once per group of records;
    counting them for each record keeps this an upper bound."""
    if scope is None:
        scope_bound: int | None = _FIELD_BOUND
    else:
        scope_bound = _sum_bounds(
            [
                _FIELD_BOUND,
                _text_bound(scope.name),
                _text_bound(scope.version),
                _text_bound(getattr(scope, "schema_url", None)),
                _value_bound(dict(getattr(scope, "attributes", None) or {})),
            ]
        )
    return _sum_bounds(
        [
            scope_bound,
            _value_bound(dict(resource.attributes)),
            _text_bound(getattr(resource, "schema_url", None)),
        ]
    )


def _span_bound(span: ReadableSpan) -> int | None:
    bounds: list[int | None] = [
        _RECORD_BOUND,
        _text_bound(span.name),
        _text_bound(span.context.trace_state.to_header() if span.context else None),
        _text_bound(span.status.description),
        _value_bound(dict(span.attributes or {})),
        _origin_bound(span.resource, span.instrumentation_scope),
    ]
    for event in span.events:
        bounds += [
            _EVENT_BOUND,
            _text_bound(event.name),
            _value_bound(dict(event.attributes or {})),
        ]
    for link in span.links:
        bounds += [
            _LINK_BOUND,
            _text_bound(link.context.trace_state.to_header()),
            _value_bound(dict(link.attributes or {})),
        ]
    return _sum_bounds(bounds)


def _log_bound(item: Any) -> int | None:
    record = item.log_record
    return _sum_bounds(
        [
            _RECORD_BOUND,
            _text_bound(record.severity_text),
            _text_bound(getattr(record, "event_name", None)),
            _value_bound(record.body),
            _value_bound(dict(record.attributes or {})),
            _origin_bound(item.resource, getattr(item, "instrumentation_scope", None)),
        ]
    )


def _fits_by_bound(records: Sequence[_Record], bound: Callable[[_Record], int | None]) -> bool:
    """Whether the records surely fit in one request, which spares an encode to measure them."""
    total = 0
    for record in records:
        try:
            record_bound = bound(record)
        except Exception:
            # The bound only spares an encode, so a record it cannot read is measured instead.
            return False
        if record_bound is None:
            return False
        total += record_bound
        if total > MAX_RECORD_BYTES:
            return False
    return True


class _Dropped(Enum):
    DROPPED = "dropped"


def _send_within_limit(
    records: Sequence[_Record],
    *,
    bound: Callable[[_Record], int | None],
    size: Callable[[Sequence[_Record]], int],
    fit: Callable[[_Record], _Record | None],
    send: Callable[[Sequence[_Record]], _Result],
    failure: _Result,
    label: str,
    on_error: Callable[[BaseException], None] | None,
) -> _Result | _Dropped:
    """Send records in requests of at most MAX_RECORD_BYTES, like the TypeScript exporter.

    Stops at the first failed request. Returns DROPPED when no request was sent because every
    record was dropped: that is not a failure, but nothing reached the ingest either.
    """
    if not records or _fits_by_bound(records, bound) or size(records) <= MAX_RECORD_BYTES:
        return send(records)
    if len(records) > 1:
        middle = (len(records) + 1) // 2
        delivered: _Result | _Dropped = _Dropped.DROPPED
        for part in (records[:middle], records[middle:]):
            result = _send_within_limit(
                part,
                bound=bound,
                size=size,
                fit=fit,
                send=send,
                failure=failure,
                label=label,
                on_error=on_error,
            )
            if result is failure:
                return failure
            if result is not _Dropped.DROPPED:
                delivered = result
        return delivered
    fitted = fit(records[0])
    if fitted is None:
        # A record the ingest can never accept is dropped instead of failing the batch forever.
        message = f"{label} record exceeds max export size, dropped"
        report_error(on_error, message, ValueError(message))
        return _Dropped.DROPPED
    message = f"{label} record exceeds max export size; its longest values were truncated"
    report_error(on_error, message, ValueError(message))
    return send([fitted])


DEFAULT_MAX_QUEUE_BYTES = 64 * 1024 * 1024


def resolve_byte_bound(name: str, value: object, default: int) -> int:
    """``value`` as a byte bound, or ``default`` when it is None. ``math.inf`` means no bound,
    and a float with no fraction, such as ``1e8``, counts as its integer. Raises ValueError for
    anything else that is not a non-negative integer."""
    if value is None:
        return default
    if isinstance(value, float) and value >= 0:
        if value == math.inf:
            return cast(int, value)
        if value.is_integer():
            return int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be None, a non-negative integer, or math.inf")
    return value


def _value_size(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(_value_size(item) for item in cast("Sequence[Any]", value))
    return 0


def _attributes_size(attributes: Mapping[str, Any] | None) -> int:
    return sum(_value_size(value) for value in (attributes or {}).values())


def _record_size(record: Any) -> int:
    """The length of the record's string attribute values, event attributes, and body."""
    log_record = getattr(record, "log_record", None)
    if log_record is not None:
        return _attributes_size(log_record.attributes) + _value_size(log_record.body)
    return _attributes_size(record.attributes) + sum(
        _attributes_size(event.attributes) for event in record.events
    )


class ExportQueueBudget:
    """Bounds the records that wait in a batch processor's queue by count and by the size of
    their text, since a count alone does not bound memory once values are sent whole.

    The processor calls admit() before it queues a record, and the exporter from releasing()
    releases each batch it receives. The count limit equals the processor's max_queue_size, so
    the processor never drops a record the budget counted. A record over the size limit is still
    admitted into an empty queue, so that one large record is not dropped forever. The first drop
    of a series is reported to on_error. Records are admitted on application threads and released
    on the export thread, so the counters hold a lock.
    """

    def __init__(
        self,
        max_count: int,
        max_size: int,
        label: str,
        on_error: Callable[[BaseException], None] | None,
    ) -> None:
        self._max_count = max_count
        self._max_size = max_size
        self._label = label
        self._on_error = on_error
        self._lock = ProcessLock(threading.Lock)
        self._count = 0
        self._size = 0
        self._reported = False
        if hasattr(os, "register_at_fork"):
            # The batch processor clears its queue in a forked child, so the budget starts over.
            reset = weakref.WeakMethod(self._reset)
            os.register_at_fork(after_in_child=lambda: (method := reset()) and method())

    def _reset(self) -> None:
        self._count = 0
        self._size = 0
        self._reported = False

    def admit(self, record: Any) -> bool:
        size = _record_size(record)
        with self._lock():
            full = self._count >= self._max_count or (
                self._count > 0 and self._size + size > self._max_size
            )
            if not full:
                self._reported = False
                self._count += 1
                self._size += size
                return True
            report = not self._reported
            self._reported = True
        if report:
            message = (
                f"the {self._label} export queue is full ({self._max_count} records or "
                f"{self._max_size} bytes); dropping {self._label} records until it drains"
            )
            report_error(self._on_error, message, RuntimeError(message))
        return False

    def release(self, records: Sequence[Any]) -> None:
        size = sum(_record_size(record) for record in records)
        with self._lock():
            self._count = max(self._count - len(records), 0)
            self._size = max(self._size - size, 0)

    def releasing(self, exporter: SpanExporter | LogRecordExporter) -> Any:
        """Wraps an exporter so that each batch it receives leaves the budget. The wrapper has
        the methods that the batch processors call, for spans and for logs."""
        return _ReleasingExporter(self, exporter)


class _ReleasingExporter:
    def __init__(
        self, budget: ExportQueueBudget, exporter: SpanExporter | LogRecordExporter
    ) -> None:
        self._budget = budget
        self._exporter = exporter

    def export(self, batch: Any) -> Any:
        self._budget.release(batch)
        return self._exporter.export(batch)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        inner_flush = getattr(self._exporter, "force_flush", None)
        return inner_flush(timeout_millis) if inner_flush is not None else True

    def shutdown(self) -> None:
        self._exporter.shutdown()


_export_state = threading.local()


@contextmanager
def exporting() -> Generator[None]:
    """Marks the current thread as inside an export of this SDK while the block runs."""
    if not getattr(_export_state, "depth", 0):
        _export_state.refused = None
    _export_state.depth = getattr(_export_state, "depth", 0) + 1
    try:
        yield
    finally:
        _export_state.depth -= 1


class IngestError(RuntimeError):
    """A failed export, with the HTTP status that the ingest answered."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def otlp_session() -> requests.Session:
    """A session for an OTLP exporter of this SDK. It keeps the last response of an export that
    the ingest refused, so that the failure names the reason that the ingest gave. The OTLP
    exporter sends on the thread of the export, thus the response is kept for that thread."""
    session = requests.Session()
    session.hooks["response"].append(_keep_refused)
    return session


def _keep_refused(response: requests.Response, *args: Any, **kwargs: Any) -> None:
    _export_state.refused = None if response.ok else response


def export_error(label: str) -> RuntimeError:
    """The error of a failed export of this thread: the status and the message of the response
    that the ingest refused it with, when there was one."""
    response: requests.Response | None = getattr(_export_state, "refused", None)
    if response is None:
        return RuntimeError(f"OTLP {label} export failed (check the API key and ingest URL)")
    reason = _status_message(response)
    return IngestError(
        f"telemetry.dev {_LABELS.get(label, label)} ingest failed: {response.status_code}"
        + (f" {reason}" if reason else ""),
        response.status_code,
    )


# The labels of the TypeScript transport, so that both SDKs report a failure the same way.
_LABELS = {"span": "trace"}


def _status_message(response: requests.Response) -> str | None:
    """The message of the JSON or protobuf Status of an OTLP error response, as the telemetry.dev
    ingest sends it, such as "This API key cannot send telemetry. Create a key with Send
    telemetry access." None for another content type, or a body that cannot be read."""
    content_type = response.headers.get("content-type", "")
    try:
        # The exporter has read the whole body; a cut could split the Status.
        body = response.content
        if "json" in content_type:
            message = json.loads(body).get("message")
        elif "protobuf" in content_type:
            # Imported here, so that a protobuf stack that cannot load it loses only the message.
            from google.rpc.status_pb2 import Status

            message = Status.FromString(body).message
        else:
            return None
    except Exception:
        return None
    return message.strip() or None if isinstance(message, str) else None


# The keys of the work items whose thread has not finished, each with the process of its thread.
# A fork that runs no hooks, as uWSGI's fork of its workers by default, copies the keys but not
# the threads, so a key of another process does not count.
_pending_lock = ProcessLock(threading.Lock)
_pending: dict[object, int] = {}


def _reset_pending_in_child() -> None:
    """Drops the keys of the parent, which no longer count in the child, so that it does not
    keep them."""
    _pending.clear()


# os.register_at_fork cannot remove a hook, and importlib.reload runs this module again with the
# same globals, so the hook of the first run stays. Windows has no fork.
if hasattr(os, "register_at_fork") and "_pending_fork_hook" not in globals():
    _pending_fork_hook = True
    os.register_at_fork(after_in_child=_reset_pending_in_child)


def after_export(key: object, work: Callable[[], object], *, shutdown: bool = False) -> bool:
    """Runs work on a new thread when this thread is inside an export of this SDK.

    The batch processors and the metric reader hold a lock while they export, and flush and
    shutdown wait for that lock. An on_error callback runs inside the export, so a flush or
    shutdown it calls would wait for its own thread forever. On the new thread, the work waits
    until the export ends, unless the exporter holds no lock, as in immediate mode. A work item
    with the same key that has not finished yet takes the place of this one, so a burst of
    reports starts one thread.

    A ``shutdown`` runs on a non-daemon thread, so the process exits only after it, and at exit
    if the thread cannot start; it is bounded by the timeouts of the shutdown. A flush runs on a
    daemon thread: it can wait for every queued record to be sent, which in an outage has no
    bound, and the shutdown at exit sends what is left. Returns whether this thread must skip
    the work.
    """
    if not getattr(_export_state, "depth", 0):
        return False
    pid = os.getpid()
    with _pending_lock():
        if _pending.get(key) == pid:
            return True
        _pending[key] = pid

    def discard() -> None:
        with _pending_lock():
            if _pending.get(key) == pid:
                del _pending[key]

    def run() -> None:
        try:
            work()
        finally:
            discard()

    try:
        # Set explicitly: a new thread inherits daemon from the export thread, which is a daemon.
        threading.Thread(target=run, name="telemetry-dev-after-export", daemon=not shutdown).start()
    except BaseException:
        # Running the work here waits for this thread, and returning False does the same.
        if shutdown:
            # The key stays pending until exit, so a burst of reports registers one exit handler.
            logger.warning(
                "telemetry-dev: could not start a thread for a shutdown; it runs at exit"
            )
            atexit.register(run)
        else:
            discard()
            logger.warning("telemetry-dev: could not start a thread for a flush")
    return True


class ExportFailures:
    """Reports the first failed export of a series to on_error and the SDK logger.

    The series stops when an export reaches the ingest, and the next failed export starts a new
    series. During an outage, each batch is a failed export, and one report for each batch is too
    many. The exporters of one client or TelemetrySpanProcessor use one instance, so an outage
    that stops traces, logs, and metrics is one report. Exporters run on background threads, thus
    report() and clear() hold a lock.
    """

    def __init__(self, on_error: Callable[[BaseException], None] | None) -> None:
        self._on_error = on_error
        self._lock = ProcessLock(threading.Lock)
        self._reported = False
        self._rejected: IngestError | None = None

    @property
    def rejected(self) -> IngestError | None:
        """The last export that the ingest refused with 401 (the API key is not valid) or 403
        (the key cannot send telemetry) since an export reached the ingest."""
        return self._rejected

    def report(self, message: str, exc: BaseException) -> None:
        with self._lock():
            if isinstance(exc, IngestError) and exc.status in (401, 403):
                self._rejected = exc
            if self._reported:
                return
            self._reported = True
        # report_error runs outside the lock. Thus a slow on_error does not stop other exporters,
        # and an on_error that sends telemetry does not deadlock.
        report_error(self._on_error, message, exc)

    def clear(self) -> None:
        """Records that an export reached the ingest, so that report() sends the next failure."""
        with self._lock():
            self._reported = False
            self._rejected = None


class ReportingSpanExporter(SpanExporter):
    """Reports export failures through its ExportFailures (TS parity). The OTLP exporter of
    OpenTelemetry only logs them."""

    def __init__(
        self,
        inner: SpanExporter,
        failures: ExportFailures,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._inner = inner
        self._failures = failures
        self._on_error = on_error

    def export(self, spans: Any) -> SpanExportResult:
        with exporting():
            try:
                result = _send_within_limit(
                    list(spans),
                    bound=_span_bound,
                    size=lambda batch: encode_spans(batch).ByteSize(),
                    fit=_fit_span,
                    send=self._inner.export,
                    failure=SpanExportResult.FAILURE,
                    label="span",
                    on_error=self._on_error,
                )
            except BaseException as exc:
                self._failures.report("span export failed", exc)
                return SpanExportResult.FAILURE
            if result is _Dropped.DROPPED:
                # Every record was dropped: the export did not fail, and the series goes on.
                return SpanExportResult.SUCCESS
            if result is SpanExportResult.FAILURE:
                self._failures.report("span export failed", export_error("span"))
            else:
                self._failures.clear()
            return result

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._inner.shutdown()


class ReportingLogExporter(LogRecordExporter):
    """Log-exporter twin of ReportingSpanExporter."""

    def __init__(
        self,
        inner: LogRecordExporter,
        failures: ExportFailures,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._inner = inner
        self._failures = failures
        self._on_error = on_error

    def export(self, batch: Any) -> LogRecordExportResult:
        with exporting():
            try:
                result = _send_within_limit(
                    list(batch),
                    bound=_log_bound,
                    size=lambda records: encode_logs(records).ByteSize(),
                    fit=_fit_log,
                    send=self._inner.export,
                    failure=LogRecordExportResult.FAILURE,
                    label="log",
                    on_error=self._on_error,
                )
            except BaseException as exc:
                self._failures.report("log export failed", exc)
                return LogRecordExportResult.FAILURE
            if result is _Dropped.DROPPED:
                # Every record was dropped: the export did not fail, and the series goes on.
                return LogRecordExportResult.SUCCESS
            if result is LogRecordExportResult.FAILURE:
                self._failures.report("log export failed", export_error("log"))
            else:
                self._failures.clear()
            return result

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        inner_flush = getattr(self._inner, "force_flush", None)
        return inner_flush(timeout_millis) if inner_flush is not None else True

    def shutdown(self) -> None:
        self._inner.shutdown()
