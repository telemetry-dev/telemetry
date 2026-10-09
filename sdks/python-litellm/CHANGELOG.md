# Changelog

## [0.1.5](https://github.com/telemetry-dev/telemetry/compare/python-litellm-v0.1.4...python-litellm-v0.1.5) (2026-10-09)


### Bug Fixes

* **sdk-litellm:** don't report prefix usage for truncated streams ([#43](https://github.com/telemetry-dev/telemetry/issues/43)) ([fa03361](https://github.com/telemetry-dev/telemetry/commit/fa03361318289141d6db25bbf02bbe6cf6c9d28f)), closes [#37](https://github.com/telemetry-dev/telemetry/issues/37)
* **sdk-litellm:** omit zero usage from litellm streams with output ([#39](https://github.com/telemetry-dev/telemetry/issues/39)) ([c1933e9](https://github.com/telemetry-dev/telemetry/commit/c1933e9fae6c8e553d8a4fb5cf1636cec8cd9938))


### Documentation

* **sdk:** note that video generation isn't traced in openrouter and litellm ([#47](https://github.com/telemetry-dev/telemetry/issues/47)) ([d050cbd](https://github.com/telemetry-dev/telemetry/commit/d050cbd9d187709cfc8132fc744a8575865bcbba))

## [0.1.4](https://github.com/telemetry-dev/telemetry/compare/python-litellm-v0.1.3...python-litellm-v0.1.4) (2026-09-30)


### Bug Fixes

* require OpenTelemetry 1.39 or newer to match telemetry-dev core ([#22](https://github.com/telemetry-dev/telemetry/issues/22)) ([cee7079](https://github.com/telemetry-dev/telemetry/commit/cee707948779667c3fc5f3709f752b7ee3d4ec9b))

## [0.1.3](https://github.com/telemetry-dev/telemetry/compare/python-litellm-v0.1.2...python-litellm-v0.1.3) (2026-09-27)


### Features

* **sdk:** trace media, realtime, rerank, and modality token usage ([#17](https://github.com/telemetry-dev/telemetry/issues/17)) ([c489503](https://github.com/telemetry-dev/telemetry/commit/c4895031a27a30004e21dc58f678d0f1ce19c072))

## [0.1.2](https://github.com/telemetry-dev/telemetry/compare/python-litellm-v0.1.1...python-litellm-v0.1.2) (2026-09-12)


### Bug Fixes

* correct package repository metadata ([#7](https://github.com/telemetry-dev/telemetry/issues/7)) ([23f74ea](https://github.com/telemetry-dev/telemetry/commit/23f74ea1480d02f4d3e34380dfb61ce149f12c86))

## [0.1.1](https://github.com/telemetry-dev/telemetry/compare/python-litellm-v0.1.0...python-litellm-v0.1.1) (2026-09-12)


### Features

* **sdk:** add streaming latency metrics ([#5](https://github.com/telemetry-dev/telemetry/issues/5)) ([f1a7261](https://github.com/telemetry-dev/telemetry/commit/f1a72617ed1f802d68ee71baf767cec934827f98))
