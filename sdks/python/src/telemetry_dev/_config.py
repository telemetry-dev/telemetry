from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

DEFAULT_BASE_URL = "https://ingest.telemetry.dev"


def _resolve_version() -> str:
    try:
        from importlib.metadata import version

        return version("telemetry-dev")
    except Exception:  # pragma: no cover - dev tree without installed dist
        return "0.0.0"


SDK_VERSION = _resolve_version()

LogLevelOption = Literal["debug", "info", "warn", "error", "silent"]
SessionMode = Literal["explicit", "process"]

logger = logging.getLogger("telemetry_dev")

_LOG_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "silent": logging.CRITICAL + 1,
}


@dataclass(frozen=True)
class ResolvedConfig:
    api_key: str | None
    base_url: str
    environment: str
    service_name: str


def resolve_config(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    environment: str | None = None,
    service_name: str | None = None,
) -> ResolvedConfig:
    url = base_url or os.environ.get("TELEMETRY_DEV_BASE_URL") or DEFAULT_BASE_URL
    return ResolvedConfig(
        api_key=api_key or os.environ.get("TELEMETRY_DEV_API_KEY") or None,
        base_url=re.sub(r"/+$", "", url),
        environment=environment or os.environ.get("TELEMETRY_DEV_ENVIRONMENT") or "production",
        service_name=service_name or os.environ.get("OTEL_SERVICE_NAME") or "unknown_service",
    )


def configure_logger(log_level: LogLevelOption) -> None:
    level = _LOG_LEVELS.get(log_level, logging.WARNING)
    logger.setLevel(level)
    if log_level != "silent" and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
        logger.addHandler(handler)


def report_error(
    on_error: Callable[[BaseException], None] | None,
    message: str,
    exc: BaseException,
) -> None:
    """Route an internal SDK error to the on_error hook + leveled logger; never raises."""
    if on_error is not None:
        try:
            on_error(exc)
        except BaseException:
            logger.debug("telemetry-dev on_error hook raised", exc_info=True)
    logger.error("telemetry-dev: %s", message, exc_info=exc)
