"""Bring-your-own-OpenTelemetry surface for telemetry.dev.

Attach ``TelemetrySpanProcessor`` to your own ``TracerProvider`` (OTel SDK,
opentelemetry-distro, ...) to ship its spans to telemetry.dev without calling
``telemetry_dev.init()``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from ._config import SDK_VERSION, logger, report_error, resolve_config
from ._metrics import GuardedOTLPMetricExporter, MetricsRecorder
from ._processor import ExportMode, StampingSpanProcessor
from ._semconv import SCOPE_NAME

__all__ = ["ExportMode", "TelemetrySpanProcessor", "create_telemetry_span_exporter"]

_EXPORTER_TIMEOUT_S = 10.0
_BATCHED_METRIC_INTERVAL_MILLIS = 60_000
_DORMANT_METRIC_INTERVAL_MILLIS = 2**31 - 1


class _NoOpSpanExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def create_telemetry_span_exporter(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = _EXPORTER_TIMEOUT_S,
) -> SpanExporter:
    """Build a telemetry.dev OTLP span exporter, or a no-op exporter without an API key."""
    config = resolve_config(api_key=api_key, base_url=base_url)
    if config.api_key is None:
        logger.debug(
            "telemetry-dev: no API key (api_key option or TELEMETRY_DEV_API_KEY); "
            "span exporter is a no-op"
        )
        return _NoOpSpanExporter()
    return OTLPSpanExporter(
        endpoint=f"{config.base_url}/v1/traces",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "X-Telemetry-Dev-SDK": "telemetry-dev",
        },
        compression=Compression.Gzip,
        timeout=timeout,
    )


class TelemetrySpanProcessor(SpanProcessor):
    """Span processor for BYO OpenTelemetry SDK setups.

    Builds telemetry.dev exporters when an API key is configured; otherwise no-ops unless
    ``span_exporter`` is supplied.
    """

    def __init__(
        self,
        span_exporter: SpanExporter | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        export_mode: ExportMode = "batched",
        max_export_batch_size: int = 64,
        schedule_delay_millis: float = 1000,
        max_queue_size: int = 2048,
        export_timeout_millis: float = 30000,
        span_filter: Callable[[ReadableSpan], bool] | None = None,
        metrics: bool = True,
        service_name: str | None = None,
        environment: str | None = None,
        metrics_recorder: MetricsRecorder | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._inner: StampingSpanProcessor | None = None
        self._meter_provider: MeterProvider | None = None
        self._on_error = on_error
        config = resolve_config(
            api_key=api_key,
            base_url=base_url,
            environment=environment,
            service_name=service_name,
        )

        if span_exporter is None:
            span_exporter = create_telemetry_span_exporter(
                api_key=config.api_key,
                base_url=config.base_url,
            )
            if isinstance(span_exporter, _NoOpSpanExporter):
                return

        if metrics_recorder is None and metrics and config.api_key is not None:
            resource = Resource.create(
                {
                    "service.name": config.service_name,
                    "deployment.environment.name": config.environment,
                }
            )
            metric_exporter = GuardedOTLPMetricExporter(
                endpoint=f"{config.base_url}/v1/metrics",
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "X-Telemetry-Dev-SDK": "telemetry-dev",
                },
                compression=Compression.Gzip,
                timeout=_EXPORTER_TIMEOUT_S,
                preferred_temporality={Histogram: AggregationTemporality.DELTA},
            )
            reader = PeriodicExportingMetricReader(
                metric_exporter,
                export_interval_millis=(
                    _BATCHED_METRIC_INTERVAL_MILLIS
                    if export_mode == "batched"
                    else _DORMANT_METRIC_INTERVAL_MILLIS
                ),
            )
            self._meter_provider = MeterProvider(
                metric_readers=[reader], resource=resource, shutdown_on_exit=False
            )
            meter = self._meter_provider.get_meter(SCOPE_NAME, SDK_VERSION)
            metrics_recorder = MetricsRecorder(meter, on_error=on_error)

        self._inner = StampingSpanProcessor(
            span_exporter,
            export_mode=export_mode,
            max_export_batch_size=max_export_batch_size,
            schedule_delay_millis=schedule_delay_millis,
            max_queue_size=max_queue_size,
            export_timeout_millis=export_timeout_millis,
            span_filter=span_filter,
            metrics_recorder=metrics_recorder,
            on_error=on_error,
        )

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        if self._inner is not None:
            self._inner.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        if self._inner is not None:
            self._inner.on_end(span)

    def shutdown(self) -> None:
        if self._inner is not None:
            try:
                self._inner.shutdown()
            except BaseException as exc:
                report_error(self._on_error, "span processor shutdown failed", exc)
        if self._meter_provider is not None:
            try:
                self._meter_provider.shutdown(timeout_millis=30_000)
            except BaseException as exc:
                report_error(self._on_error, "metric provider shutdown failed", exc)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        ok = True
        if self._inner is not None:
            try:
                ok = self._inner.force_flush(timeout_millis)
            except BaseException as exc:
                report_error(self._on_error, "span processor force_flush failed", exc)
                ok = False
        if self._meter_provider is not None:
            try:
                ok = self._meter_provider.force_flush(timeout_millis=timeout_millis) and ok
            except BaseException as exc:
                report_error(self._on_error, "metric provider force_flush failed", exc)
                ok = False
        return ok
