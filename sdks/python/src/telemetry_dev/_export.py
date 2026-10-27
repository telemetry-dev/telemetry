from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from ._config import report_error


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

    def __init__(self, inner: SpanExporter, failures: ExportFailures) -> None:
        self._inner = inner
        self._failures = failures

    def export(self, spans: Any) -> SpanExportResult:
        try:
            result = self._inner.export(spans)
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

    def __init__(self, inner: LogRecordExporter, failures: ExportFailures) -> None:
        self._inner = inner
        self._failures = failures

    def export(self, batch: Any) -> LogRecordExportResult:
        try:
            result = self._inner.export(batch)
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
