# Changelog

## [0.2.0](https://github.com/telemetry-dev/telemetry/compare/tanstack-ai-v0.1.10...tanstack-ai-v0.2.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **sdk:** the default base URL for telemetry export and getPrompt/get_prompt is now https://api.telemetry.dev. https://ingest.telemetry.dev reaches the same service and stays supported; networks that allow only it should set TELEMETRY_DEV_BASE_URL to it.

### Features

* **sdk:** send to api.telemetry.dev by default ([#109](https://github.com/telemetry-dev/telemetry/issues/109)) ([ad8ccf3](https://github.com/telemetry-dev/telemetry/commit/ad8ccf3c41355683deb3b38dab4b604ecaa25a7a))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * @telemetry-dev/otel bumped to 0.2.0

## [0.1.10](https://github.com/telemetry-dev/telemetry/compare/tanstack-ai-v0.1.9...tanstack-ai-v0.1.10) (2026-10-27)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * @telemetry-dev/otel bumped to 0.1.7

## [0.1.9](https://github.com/telemetry-dev/telemetry/compare/tanstack-ai-v0.1.8...tanstack-ai-v0.1.9) (2026-10-19)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * @telemetry-dev/otel bumped to 0.1.6

## [0.1.8](https://github.com/telemetry-dev/telemetry/compare/tanstack-ai-v0.1.7...tanstack-ai-v0.1.8) (2026-10-14)


### Features

* **sdk-tanstack-ai:** trace generation activities besides chat and evaluate ([#50](https://github.com/telemetry-dev/telemetry/issues/50)) ([bba49a4](https://github.com/telemetry-dev/telemetry/commit/bba49a47b7069cf4348773c04dc52c21b303506f))


### Bug Fixes

* **sdk-tanstack-ai:** record modality token usage on chat spans ([#54](https://github.com/telemetry-dev/telemetry/issues/54)) ([1976ace](https://github.com/telemetry-dev/telemetry/commit/1976aceccc0b277987106a08976050e98e8ced54))

## [0.1.7](https://github.com/telemetry-dev/telemetry/compare/tanstack-ai-v0.1.6...tanstack-ai-v0.1.7) (2026-09-30)


### Bug Fixes

* **sdk:** include cache tokens in input totals ([#26](https://github.com/telemetry-dev/telemetry/issues/26)) ([fa2e833](https://github.com/telemetry-dev/telemetry/commit/fa2e83309e5e67d36abfeaa047c8a5e62da3befe))

## [0.1.6](https://github.com/telemetry-dev/telemetry/compare/tanstack-ai-v0.1.5...tanstack-ai-v0.1.6) (2026-09-27)

### Features

- **sdk:** add evaluation telemetry ([#14](https://github.com/telemetry-dev/telemetry/issues/14)) ([b1b5218](https://github.com/telemetry-dev/telemetry/commit/b1b521866763b4f090b083530671714823a2384f))

## [0.1.5](https://github.com/telemetry-dev/telemetry/compare/tanstack-ai-v0.1.4...tanstack-ai-v0.1.5) (2026-09-12)

### Bug Fixes

- correct package repository metadata ([#7](https://github.com/telemetry-dev/telemetry/issues/7)) ([23f74ea](https://github.com/telemetry-dev/telemetry/commit/23f74ea1480d02f4d3e34380dfb61ce149f12c86))

### Dependencies

- The following workspace dependencies were updated
  - dependencies
    - @telemetry-dev/otel bumped to 0.1.5

## [0.1.4](https://github.com/telemetry-dev/telemetry/compare/tanstack-ai-v0.1.3...tanstack-ai-v0.1.4) (2026-09-12)

### Features

- **sdk:** add streaming latency metrics ([#5](https://github.com/telemetry-dev/telemetry/issues/5)) ([f1a7261](https://github.com/telemetry-dev/telemetry/commit/f1a72617ed1f802d68ee71baf767cec934827f98))

### Dependencies

- The following workspace dependencies were updated
  - dependencies
    - @telemetry-dev/otel bumped to 0.1.4

## [0.1.3](https://github.com/telemetry-dev/sdks/compare/tanstack-ai-v0.1.2...tanstack-ai-v0.1.3) (2026-09-07)

### Bug Fixes

- include shared instrumentation and capture refactors in published SDKs ([7c82386](https://github.com/telemetry-dev/sdks/commit/7c82386e139303a49f4cb3b6a39affac85763e88))

### Dependencies

- The following workspace dependencies were updated
  - dependencies
    - @telemetry-dev/otel bumped to 0.1.3

## [0.1.2](https://github.com/telemetry-dev/sdks/compare/tanstack-ai-v0.1.1...tanstack-ai-v0.1.2) (2026-09-05)

### Features

- one trace per session, with turn-by-turn session replay ([#226](https://github.com/telemetry-dev/sdks/issues/226)) ([44e2e75](https://github.com/telemetry-dev/sdks/commit/44e2e7510bc5073df89a0135cac9e986aeade5c6))

### Bug Fixes

- **sdk:** share the otel transport with the ai and tanstack packages ([#208](https://github.com/telemetry-dev/sdks/issues/208)) ([6611e79](https://github.com/telemetry-dev/sdks/commit/6611e79299d59b285ea0832757e11ad16abab2cf))

### Dependencies

- The following workspace dependencies were updated
  - dependencies
    - @telemetry-dev/otel bumped to 0.1.2

## [0.1.1](https://github.com/telemetry-dev/sdks/compare/tanstack-ai-v0.1.0...tanstack-ai-v0.1.1) (2026-09-02)

### Features

- **sdk:** add telemetry.dev request header ([#181](https://github.com/telemetry-dev/sdks/issues/181)) ([23d5141](https://github.com/telemetry-dev/sdks/commit/23d5141764b9e7b1640f7ea6f836686146a24822))
