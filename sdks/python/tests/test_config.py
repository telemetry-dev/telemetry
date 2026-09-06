from __future__ import annotations

import pytest

from telemetry_dev._config import resolve_config


def test_defaults_without_env() -> None:
    config = resolve_config()
    assert config.api_key is None
    assert config.base_url == "https://ingest.telemetry.dev"
    assert config.environment == "production"
    assert config.service_name == "unknown_service"


def test_env_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEMETRY_DEV_API_KEY", "td_live_env")
    monkeypatch.setenv("TELEMETRY_DEV_BASE_URL", "https://example.test")
    monkeypatch.setenv("TELEMETRY_DEV_ENVIRONMENT", "staging")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "svc")
    config = resolve_config()
    assert config.api_key == "td_live_env"
    assert config.base_url == "https://example.test"
    assert config.environment == "staging"
    assert config.service_name == "svc"


def test_arguments_take_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEMETRY_DEV_API_KEY", "td_live_env")
    monkeypatch.setenv("TELEMETRY_DEV_BASE_URL", "https://env.test")
    monkeypatch.setenv("TELEMETRY_DEV_ENVIRONMENT", "staging")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "env-svc")
    config = resolve_config(
        api_key="td_live_arg",
        base_url="https://arg.test",
        environment="dev",
        service_name="arg-svc",
    )
    assert config.api_key == "td_live_arg"
    assert config.base_url == "https://arg.test"
    assert config.environment == "dev"
    assert config.service_name == "arg-svc"


def test_base_url_trailing_slashes_stripped() -> None:
    assert resolve_config(base_url="https://x.test/").base_url == "https://x.test"
    assert resolve_config(base_url="https://x.test///").base_url == "https://x.test"


def test_base_url_env_trailing_slash_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEMETRY_DEV_BASE_URL", "https://env.test//")
    assert resolve_config().base_url == "https://env.test"
