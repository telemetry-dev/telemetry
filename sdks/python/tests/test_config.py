from __future__ import annotations

from telemetry_dev._config import resolve_config


def test_defaults_without_env() -> None:
    config = resolve_config()
    assert config.api_key is None
    assert config.base_url == "https://ingest.telemetry.dev"
    assert config.environment == "production"
    assert config.service_name == "unknown_service"
