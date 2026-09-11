from __future__ import annotations

import atexit
import threading
import uuid
from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from opentelemetry._logs import Logger as OtelLogger
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    LogRecordExporter,
    LogRecordExportResult,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import Sampler
from opentelemetry.trace import Tracer

from ._config import (
    SDK_VERSION,
    LogLevelOption,
    ResolvedConfig,
    SessionMode,
    configure_logger,
    logger,
    report_error,
    resolve_config,
)
from ._context import SessionSampler
from ._metrics import GuardedOTLPMetricExporter, MetricsRecorder, OutputChunkAggregation
from ._processor import ExportMode, StampingSpanProcessor
from ._semconv import SCOPE_NAME
from ._serialize import Mask, serialize_content

_lock = threading.RLock()
_client: Client | None = None


def _default_scope_filter(span: ReadableSpan) -> bool:
    scope = span.instrumentation_scope
    return scope is not None and scope.name == SCOPE_NAME


class _ReportingSpanExporter(SpanExporter):
    """Surfaces transport failures to on_error (TS parity: exporters route errors to onError,
    the stock OTLP exporter only logs them)."""

    def __init__(
        self, inner: SpanExporter, on_error: Callable[[BaseException], None] | None
    ) -> None:
        self._inner = inner
        self._on_error = on_error

    def export(self, spans: Any) -> SpanExportResult:
        try:
            result = self._inner.export(spans)
        except BaseException as exc:
            report_error(self._on_error, "span export failed", exc)
            return SpanExportResult.FAILURE
        if result is SpanExportResult.FAILURE:
            report_error(
                self._on_error,
                "span export failed",
                RuntimeError("OTLP span export failed (check the API key and ingest URL)"),
            )
        return result

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._inner.shutdown()


class _ReportingLogExporter(LogRecordExporter):
    """Log-exporter twin of _ReportingSpanExporter."""

    def __init__(
        self, inner: LogRecordExporter, on_error: Callable[[BaseException], None] | None
    ) -> None:
        self._inner = inner
        self._on_error = on_error

    def export(self, batch: Any) -> LogRecordExportResult:
        try:
            result = self._inner.export(batch)
        except BaseException as exc:
            report_error(self._on_error, "log export failed", exc)
            return LogRecordExportResult.FAILURE
        if result is LogRecordExportResult.FAILURE:
            report_error(
                self._on_error,
                "log export failed",
                RuntimeError("OTLP log export failed (check the API key and ingest URL)"),
            )
        return result

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        inner_flush = getattr(self._inner, "force_flush", None)
        return inner_flush(timeout_millis) if inner_flush is not None else True

    def shutdown(self) -> None:
        self._inner.shutdown()


class Client:
    """Holds the telemetry pipelines for one init() call. Obtain via telemetry_dev.init()."""

    def __init__(
        self,
        *,
        config: ResolvedConfig,
        enabled: bool,
        session_mode: SessionMode = "explicit",
        register_global: bool = False,
        export_mode: ExportMode = "batched",
        capture_input: bool = True,
        capture_output: bool = True,
        mask: Mask | None = None,
        max_attribute_length: int = 65536,
        span_filter: Callable[[ReadableSpan], bool] | None = None,
        sampler: Sampler | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        disable_atexit: bool = False,
        timeout: float = 10.0,
        span_exporter: SpanExporter | None = None,
        log_exporter: LogRecordExporter | None = None,
        metric_reader: MetricReader | None = None,
    ) -> None:
        self.enabled = enabled
        self.config = config
        self.capture_input = capture_input
        self.capture_output = capture_output
        self.mask = mask
        self.max_attribute_length = max_attribute_length
        self.on_error = on_error
        self._process_session_id: str | None = (
            str(uuid.uuid4()) if enabled and session_mode == "process" else None
        )
        self._shutdown = False
        self._atexit_registered = False

        self.processor: StampingSpanProcessor | None = None
        self.tracer: Tracer | None = None
        self.otel_logger: OtelLogger | None = None
        self._tracer_provider: TracerProvider | None = None
        self._logger_provider: LoggerProvider | None = None
        self._meter_provider: MeterProvider | None = None
        self._output_chunks: OutputChunkAggregation | None = None

        if not enabled:
            return

        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "X-Telemetry-Dev-SDK": "telemetry-dev",
        }
        resource = Resource.create(
            {
                "service.name": config.service_name,
                "deployment.environment.name": config.environment,
            }
        )

        # The funnel keeps capped content (marker included) within max_attribute_length, so
        # the backstop can use the same cap for raw attributes set outside the funnel.
        self._tracer_provider = TracerProvider(
            resource=resource,
            sampler=sampler,
            shutdown_on_exit=False,
            span_limits=SpanLimits(
                max_attribute_length=max_attribute_length,
                max_span_attribute_length=max_attribute_length,
            ),
        )
        self._tracer_provider.sampler = SessionSampler(self._tracer_provider.sampler)

        # Without an api key (enabled via a test seam), never construct real network
        # exporters — they would POST to the ingest with a bogus Authorization header.
        if metric_reader is None and config.api_key is not None:
            self._output_chunks = OutputChunkAggregation()
            metric_exporter = GuardedOTLPMetricExporter(
                endpoint=f"{config.base_url}/v1/metrics",
                headers=headers,
                compression=Compression.Gzip,
                timeout=timeout,
                preferred_temporality={Histogram: AggregationTemporality.DELTA},
                on_error=on_error,
            )
            metric_exporter.output_chunks = self._output_chunks
            metric_reader = PeriodicExportingMetricReader(
                metric_exporter, export_interval_millis=60_000
            )
        metrics_recorder: MetricsRecorder | None = None
        if metric_reader is not None:
            self._meter_provider = MeterProvider(
                metric_readers=[metric_reader], resource=resource, shutdown_on_exit=False
            )
            meter = self._meter_provider.get_meter(SCOPE_NAME, SDK_VERSION)
            metrics_recorder = MetricsRecorder(
                meter, on_error=on_error, output_chunks=self._output_chunks
            )

        if span_exporter is None:
            span_exporter = _ReportingSpanExporter(
                OTLPSpanExporter(
                    endpoint=f"{config.base_url}/v1/traces",
                    headers=headers,
                    compression=Compression.Gzip,
                    timeout=timeout,
                ),
                on_error,
            )
        if span_filter is None and register_global:
            span_filter = _default_scope_filter
        self.processor = StampingSpanProcessor(
            span_exporter,
            export_mode=export_mode,
            span_filter=span_filter,
            metrics_recorder=metrics_recorder,
            on_error=on_error,
        )
        self._tracer_provider.add_span_processor(self.processor)
        self.tracer = self._tracer_provider.get_tracer(SCOPE_NAME, SDK_VERSION)

        if log_exporter is None and config.api_key is not None:
            log_exporter = _ReportingLogExporter(
                OTLPLogExporter(
                    endpoint=f"{config.base_url}/v1/logs",
                    headers=headers,
                    compression=Compression.Gzip,
                    timeout=timeout,
                ),
                on_error,
            )
        if log_exporter is not None:
            self._logger_provider = LoggerProvider(resource=resource, shutdown_on_exit=False)
            if export_mode == "immediate":
                self._logger_provider.add_log_record_processor(
                    SimpleLogRecordProcessor(log_exporter)
                )
            else:
                self._logger_provider.add_log_record_processor(
                    BatchLogRecordProcessor(log_exporter)
                )
            self.otel_logger = self._logger_provider.get_logger(SCOPE_NAME, SDK_VERSION)

        if register_global:
            try:
                trace.set_tracer_provider(self._tracer_provider)
            except BaseException as exc:
                self.report("failed to register global tracer provider", exc)

        if not disable_atexit:
            atexit.register(self.shutdown)
            self._atexit_registered = True

    def report(self, message: str, exc: BaseException) -> None:
        report_error(self.on_error, message, exc)

    def serialize(self, value: Any, key: str) -> str | None:
        return serialize_content(
            value,
            key=key,
            mask=self.mask,
            max_len=self.max_attribute_length,
            on_error=self.on_error,
        )

    def flush(self, timeout_s: float = 10.0) -> None:
        if not self.enabled or self._shutdown:
            return
        timeout_millis = int(timeout_s * 1000)
        try:
            if self._tracer_provider is not None:
                self._tracer_provider.force_flush(timeout_millis)
            if self._logger_provider is not None:
                self._logger_provider.force_flush(timeout_millis)
            if self._meter_provider is not None:
                self._meter_provider.force_flush(timeout_millis=timeout_millis)
        except BaseException as exc:
            self.report("flush failed", exc)

    def shutdown(self, timeout_s: float = 10.0) -> None:
        global _client
        if self._shutdown:
            return
        self._shutdown = True
        if self._atexit_registered:
            try:
                atexit.unregister(self.shutdown)
            except BaseException:
                pass
        timeout_millis = int(timeout_s * 1000)
        try:
            if self._tracer_provider is not None:
                self._tracer_provider.shutdown()
            if self._logger_provider is not None:
                self._logger_provider.shutdown()
            if self._meter_provider is not None:
                self._meter_provider.shutdown(timeout_millis=timeout_millis)
        except BaseException as exc:
            self.report("shutdown failed", exc)
        with _lock:
            if _client is self:
                _client = None


def init(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    environment: str | None = None,
    service_name: str | None = None,
    enabled: bool = True,
    session_mode: SessionMode = "explicit",
    register_global: bool = False,
    export_mode: ExportMode = "batched",
    log_level: LogLevelOption = "warn",
    capture_input: bool = True,
    capture_output: bool = True,
    mask: Mask | None = None,
    max_attribute_length: int = 65536,
    span_filter: Callable[[ReadableSpan], bool] | None = None,
    sampler: Sampler | None = None,
    on_error: Callable[[BaseException], None] | None = None,
    disable_atexit: bool = False,
    timeout: float = 10.0,
    span_exporter: SpanExporter | None = None,
    log_exporter: LogRecordExporter | None = None,
    metric_reader: MetricReader | None = None,
) -> Client:
    """Initialize the telemetry.dev SDK. Fail-open: a missing API key yields a silent no-op
    client; internal errors are routed to on_error and the 'telemetry_dev' logger, never raised."""
    global _client
    configure_logger(log_level)
    config = resolve_config(
        api_key=api_key,
        base_url=base_url,
        environment=environment,
        service_name=service_name,
    )
    has_seam = span_exporter is not None or log_exporter is not None or metric_reader is not None
    effective_enabled = enabled and (has_seam or config.api_key is not None)
    if enabled and not effective_enabled:
        logger.debug("telemetry-dev disabled: no API key")

    try:
        if session_mode not in ("explicit", "process"):
            raise ValueError(f"invalid session_mode: {session_mode!r}")
        client = Client(
            config=config,
            enabled=effective_enabled,
            session_mode=session_mode,
            register_global=register_global,
            export_mode=export_mode,
            capture_input=capture_input,
            capture_output=capture_output,
            mask=mask,
            max_attribute_length=max_attribute_length,
            span_filter=span_filter,
            sampler=sampler,
            on_error=on_error,
            disable_atexit=disable_atexit,
            timeout=timeout,
            span_exporter=span_exporter,
            log_exporter=log_exporter,
            metric_reader=metric_reader,
        )
    except BaseException as exc:
        report_error(on_error, "init failed; telemetry disabled", exc)
        client = Client(config=config, enabled=False)

    with _lock:
        previous = _client
        _client = client
    if previous is not None:
        logger.warning("telemetry-dev: init() called again; replacing the previous client")
        try:
            previous.shutdown()
        except BaseException as exc:
            report_error(on_error, "failed to shut down the previous client", exc)
    return client


def get_client() -> Client | None:
    return _client


def flush(timeout_s: float = 10.0) -> None:
    """Force-flush pending traces, logs, and metrics (serverless-friendly)."""
    client = _client
    if client is not None:
        client.flush(timeout_s)


def shutdown(timeout_s: float = 10.0) -> None:
    """Flush and tear down the SDK; resets the singleton to uninitialized."""
    client = _client
    if client is not None:
        client.shutdown(timeout_s)
