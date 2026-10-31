from __future__ import annotations

import copy
import os
import threading
import weakref
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar, cast

from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from ._config import logger, report_error
from ._serialize import TRUNCATION_MARKER, truncate, utf16_length

# The ingest rejects a body over 16 MiB after gzip decoding, so this limit applies to the
# serialized request. Keep it equal to DECOMPRESSED_BODY_LIMIT in the ingest.
MAX_BODY_BYTES = 16 * 1024 * 1024
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
        excess_bytes = size(fitted, values[_BODY]) - MAX_BODY_BYTES
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


def _send_within_limit(
    records: Sequence[_Record],
    *,
    size: Callable[[Sequence[_Record]], int],
    fit: Callable[[_Record], _Record | None],
    send: Callable[[Sequence[_Record]], _Result],
    success: _Result,
    failure: _Result,
    label: str,
    on_error: Callable[[BaseException], None] | None,
) -> _Result:
    """Send records in requests of at most MAX_BODY_BYTES, like the TypeScript exporter."""
    if not records or size(records) <= MAX_BODY_BYTES:
        return send(records)
    if len(records) > 1:
        middle = (len(records) + 1) // 2
        results = [
            _send_within_limit(
                part,
                size=size,
                fit=fit,
                send=send,
                success=success,
                failure=failure,
                label=label,
                on_error=on_error,
            )
            for part in (records[:middle], records[middle:])
        ]
        return failure if failure in results else success
    fitted = fit(records[0])
    if fitted is None:
        # A record the ingest can never accept is dropped instead of failing the batch forever.
        message = f"{label} record exceeds max export size, dropped"
        report_error(on_error, message, ValueError(message))
        return success
    message = f"{label} record exceeds max export size; its longest values were truncated"
    report_error(on_error, message, ValueError(message))
    return send([fitted])


DEFAULT_MAX_QUEUE_BYTES = 64 * 1024 * 1024


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
        self._lock = threading.Lock()
        self._count = 0
        self._size = 0
        self._reported = False
        if hasattr(os, "register_at_fork"):
            # The batch processor clears its queue in a forked child, so the budget starts over.
            reset = weakref.WeakMethod(self._reset)
            os.register_at_fork(after_in_child=lambda: (method := reset()) and method())

    def _reset(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._size = 0
        self._reported = False

    def admit(self, record: Any) -> bool:
        size = _record_size(record)
        with self._lock:
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
        with self._lock:
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
    _export_state.depth = getattr(_export_state, "depth", 0) + 1
    try:
        yield
    finally:
        _export_state.depth -= 1


_pending_lock = threading.Lock()
_pending: set[object] = set()


def after_export(key: object, work: Callable[[], object]) -> bool:
    """Runs work on a new thread when this thread is inside an export of this SDK.

    The batch processors and the metric reader hold a lock while they export, and flush and
    shutdown wait for that lock. An on_error callback runs inside the export, so a flush or
    shutdown it calls would wait for its own thread forever. On the new thread, the work waits
    until the export ends. A work item with the same key that has not finished yet takes the
    place of this one, so a burst of reports starts one thread. The thread is not a daemon, so
    the process exits only after the work. Returns whether this thread must skip the work.
    """
    if not getattr(_export_state, "depth", 0):
        return False
    with _pending_lock:
        if key in _pending:
            return True
        _pending.add(key)

    def run() -> None:
        try:
            work()
        finally:
            with _pending_lock:
                _pending.discard(key)

    try:
        threading.Thread(target=run, name="telemetry-dev-after-export").start()
    except BaseException:
        with _pending_lock:
            _pending.discard(key)
        logger.warning("telemetry-dev: could not start a thread for a flush or shutdown")
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
        self._lock = threading.Lock()
        self._reported = False

    def report(self, message: str, exc: BaseException) -> None:
        with self._lock:
            if self._reported:
                return
            self._reported = True
        # report_error runs outside the lock. Thus a slow on_error does not stop other exporters,
        # and an on_error that sends telemetry does not deadlock.
        report_error(self._on_error, message, exc)

    def clear(self) -> None:
        """Records that an export reached the ingest, so that report() sends the next failure."""
        with self._lock:
            self._reported = False


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
                    size=lambda batch: encode_spans(batch).ByteSize(),
                    fit=_fit_span,
                    send=self._inner.export,
                    success=SpanExportResult.SUCCESS,
                    failure=SpanExportResult.FAILURE,
                    label="span",
                    on_error=self._on_error,
                )
            except BaseException as exc:
                self._failures.report("span export failed", exc)
                return SpanExportResult.FAILURE
            if result is SpanExportResult.FAILURE:
                self._failures.report(
                    "span export failed",
                    RuntimeError("OTLP span export failed (check the API key and ingest URL)"),
                )
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
                    size=lambda records: encode_logs(records).ByteSize(),
                    fit=_fit_log,
                    send=self._inner.export,
                    success=LogRecordExportResult.SUCCESS,
                    failure=LogRecordExportResult.FAILURE,
                    label="log",
                    on_error=self._on_error,
                )
            except BaseException as exc:
                self._failures.report("log export failed", exc)
                return LogRecordExportResult.FAILURE
            if result is LogRecordExportResult.FAILURE:
                self._failures.report(
                    "log export failed",
                    RuntimeError("OTLP log export failed (check the API key and ingest URL)"),
                )
            else:
                self._failures.clear()
            return result

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        inner_flush = getattr(self._inner, "force_flush", None)
        return inner_flush(timeout_millis) if inner_flush is not None else True

    def shutdown(self) -> None:
        self._inner.shutdown()
