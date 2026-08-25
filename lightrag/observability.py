"""Small OpenTelemetry bridge for the API server and shared runtime paths."""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

_initialized = False
_instruments: dict[str, Any] = {}


def _otel_api() -> tuple[Any, Any] | tuple[None, None]:
    try:
        from opentelemetry import metrics, trace

        return trace, metrics
    except ImportError:
        return None, None


def _provider_name(base_url: str | None) -> str:
    host = (urlparse(base_url).hostname or "").lower()
    if "deepseek" in host:
        return "deepseek"
    if "openai" in host:
        return "openai"
    if "azure" in host:
        return "azure.ai.openai"
    return "openai_compatible"


def setup_telemetry(app: Any) -> bool:
    """Configure OTLP/HTTP tracing and metrics when an endpoint is supplied."""
    global _initialized, _instruments
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").rstrip("/")
    if _initialized or not endpoint:
        return _initialized

    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", "lightrag"),
            "service.version": os.getenv("LIGHTRAG_VERSION", "unknown"),
            "deployment.environment.name": os.getenv(
                "OTEL_DEPLOYMENT_ENVIRONMENT", "production"
            ),
        }
    )
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(trace_provider)

    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics"),
        export_interval_millis=int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "60000")),
    )
    metrics.set_meter_provider(
        MeterProvider(resource=resource, metric_readers=[metric_reader])
    )
    meter = metrics.get_meter("lightrag")
    _instruments = {
        "token_usage": meter.create_histogram(
            "gen_ai.client.token.usage", unit="{token}"
        ),
        "llm_duration": meter.create_histogram(
            "gen_ai.client.operation.duration", unit="s"
        ),
        "mcp_duration": meter.create_histogram(
            "lightrag.mcp.operation.duration", unit="s"
        ),
        "pipeline_duration": meter.create_histogram(
            "lightrag.pipeline.operation.duration", unit="s"
        ),
        "query_duration": meter.create_histogram(
            "lightrag.query.operation.duration", unit="s"
        ),
    }
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls=os.getenv("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS", "health"),
    )
    _initialized = True
    return True


@contextmanager
def traced_operation(name: str, attributes: dict[str, Any] | None = None):
    trace, _ = _otel_api()
    if trace is None:
        yield None
        return
    with trace.get_tracer("lightrag").start_as_current_span(
        name, attributes=attributes or {}
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute("error.type", type(exc).__name__)
            raise


def record_gen_ai_usage(
    *,
    operation: str,
    model: str,
    base_url: str | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    duration_s: float | None = None,
    role: str | None = None,
) -> None:
    trace, _ = _otel_api()
    if trace is None:
        return
    attrs = {
        "gen_ai.operation.name": operation,
        "gen_ai.provider.name": _provider_name(base_url),
        "gen_ai.request.model": model,
    }
    if role:
        attrs["lightrag.llm.role"] = role
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attributes(
            {
                **attrs,
                "gen_ai.usage.input_tokens": input_tokens,
                "gen_ai.usage.output_tokens": output_tokens,
            }
        )
    token_histogram = _instruments.get("token_usage")
    if token_histogram:
        token_histogram.record(input_tokens, {**attrs, "gen_ai.token.type": "input"})
        token_histogram.record(
            output_tokens, {**attrs, "gen_ai.token.type": "output"}
        )
    if duration_s is not None and (duration := _instruments.get("llm_duration")):
        duration.record(duration_s, attrs)


def record_operation_duration(metric: str, duration_s: float, status: str) -> None:
    if instrument := _instruments.get(metric):
        instrument.record(duration_s, {"lightrag.status": status})


def traced_async(name: str, metric: str | None = None):
    """Trace an async operation without changing its public signature."""

    def decorate(func: Callable[..., Any]):
        @functools.wraps(func)
        async def wrapped(*args: Any, **kwargs: Any):
            started = time.perf_counter()
            status = "ok"
            try:
                with traced_operation(name):
                    return await func(*args, **kwargs)
            except Exception:
                status = "error"
                raise
            finally:
                if metric and (instrument := _instruments.get(metric)):
                    instrument.record(
                        time.perf_counter() - started, {"lightrag.status": status}
                    )

        return wrapped

    return decorate
