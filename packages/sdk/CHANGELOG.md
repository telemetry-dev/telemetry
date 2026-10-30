# Changelog

## [0.2.1](https://github.com/telemetry-dev/telemetry/compare/sdk-v0.2.0...sdk-v0.2.1) (2026-10-30)


### Bug Fixes

* **sdk:** bound prompt fetches and time them by elapsed time ([#117](https://github.com/telemetry-dev/telemetry/issues/117)) ([0741322](https://github.com/telemetry-dev/telemetry/commit/0741322071d2648cbb771750763847e3483c8fa0))

## [0.2.0](https://github.com/telemetry-dev/telemetry/compare/sdk-v0.1.12...sdk-v0.2.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **sdk:** the default base URL for telemetry export and getPrompt/get_prompt is now https://api.telemetry.dev. https://ingest.telemetry.dev reaches the same service and stays supported; networks that allow only it should set TELEMETRY_DEV_BASE_URL to it.

### Features

* **sdk:** send to api.telemetry.dev by default ([#109](https://github.com/telemetry-dev/telemetry/issues/109)) ([ad8ccf3](https://github.com/telemetry-dev/telemetry/commit/ad8ccf3c41355683deb3b38dab4b604ecaa25a7a))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * @telemetry-dev/otel bumped to 0.2.0

## [0.1.12](https://github.com/telemetry-dev/telemetry/compare/sdk-v0.1.11...sdk-v0.1.12) (2026-10-29)


### Bug Fixes

* **sdk:** honor retry-after for prompts and keep python task variables ([#102](https://github.com/telemetry-dev/telemetry/issues/102)) ([6ad4a0c](https://github.com/telemetry-dev/telemetry/commit/6ad4a0c1714d8849f9a5561016bd84f3959d534c))

## [0.1.11](https://github.com/telemetry-dev/telemetry/compare/sdk-v0.1.10...sdk-v0.1.11) (2026-10-29)


### Features

* **sdk:** add getPrompt and withPrompt in TypeScript and Python ([#99](https://github.com/telemetry-dev/telemetry/issues/99)) ([b1a33e4](https://github.com/telemetry-dev/telemetry/commit/b1a33e446864f02c2fd050e754c81c875c7f9864))

## [0.1.10](https://github.com/telemetry-dev/telemetry/compare/sdk-v0.1.9...sdk-v0.1.10) (2026-10-27)


### Bug Fixes

* **sdk:** keep exception text behind capture flags ([#88](https://github.com/telemetry-dev/telemetry/issues/88)) ([e9e49c3](https://github.com/telemetry-dev/telemetry/commit/e9e49c33dfcff28c74f7a6e16126054dc32b0d90))
* **sdk:** raise opentelemetry floor past baggage advisory ([#80](https://github.com/telemetry-dev/telemetry/issues/80)) ([29029fa](https://github.com/telemetry-dev/telemetry/commit/29029fa1050ee905306633c5d60d4bc75cf61074))
* **sdk:** report export outages once and keep errors off agent TUIs ([d52effd](https://github.com/telemetry-dev/telemetry/commit/d52effd6f28b532a30d1be85cc5d36059e254604))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * @telemetry-dev/otel bumped to 0.1.7

## [0.1.9](https://github.com/telemetry-dev/telemetry/compare/sdk-v0.1.8...sdk-v0.1.9) (2026-10-19)


### Features

* **sdk:** run the configured mask on integration content and in the ai package ([#73](https://github.com/telemetry-dev/telemetry/issues/73)) ([365b6e7](https://github.com/telemetry-dev/telemetry/commit/365b6e76ebbccd3bb63803d0abb1757b7d16573d))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * @telemetry-dev/otel bumped to 0.1.6

## [0.1.8](https://github.com/telemetry-dev/telemetry/compare/sdk-v0.1.7...sdk-v0.1.8) (2026-10-14)


### Bug Fixes

* **sdk-python:** charge json escapes when CaptureBudget measures strings ([#48](https://github.com/telemetry-dev/telemetry/issues/48)) ([3d5daf5](https://github.com/telemetry-dev/telemetry/commit/3d5daf578bbcdfe4113bea62b55fceb88dd10418))
* **sdk:** keep non-streamed captures cut to the retention bound from masks ([#53](https://github.com/telemetry-dev/telemetry/issues/53)) ([60fdeb3](https://github.com/telemetry-dev/telemetry/commit/60fdeb33c564f9f80435714e8dc9382a6dc20083))

## [0.1.7](https://github.com/telemetry-dev/telemetry/compare/sdk-v0.1.6...sdk-v0.1.7) (2026-10-01)


### Bug Fixes

* **sdk:** flag incomplete stream capture and keep it from masks ([#33](https://github.com/telemetry-dev/telemetry/issues/33)) ([9454284](https://github.com/telemetry-dev/telemetry/commit/9454284555a2b704c2616620311d36d9db7a9d82))

## [0.1.6](https://github.com/telemetry-dev/telemetry/compare/sdk-v0.1.5...sdk-v0.1.6) (2026-09-27)

### Features

- **sdk:** add evaluation telemetry ([#14](https://github.com/telemetry-dev/telemetry/issues/14)) ([b1b5218](https://github.com/telemetry-dev/telemetry/commit/b1b521866763b4f090b083530671714823a2384f))
- **sdk:** trace media, realtime, rerank, and modality token usage ([#17](https://github.com/telemetry-dev/telemetry/issues/17)) ([c489503](https://github.com/telemetry-dev/telemetry/commit/c4895031a27a30004e21dc58f678d0f1ce19c072))

## [0.1.5](https://github.com/telemetry-dev/telemetry/compare/sdk-v0.1.4...sdk-v0.1.5) (2026-09-12)

### Bug Fixes

- correct package repository metadata ([#7](https://github.com/telemetry-dev/telemetry/issues/7)) ([23f74ea](https://github.com/telemetry-dev/telemetry/commit/23f74ea1480d02f4d3e34380dfb61ce149f12c86))

### Dependencies

- The following workspace dependencies were updated
  - dependencies
    - @telemetry-dev/otel bumped to 0.1.5

## [0.1.4](https://github.com/telemetry-dev/telemetry/compare/sdk-v0.1.3...sdk-v0.1.4) (2026-09-12)

### Features

- **sdk:** add streaming latency metrics ([#5](https://github.com/telemetry-dev/telemetry/issues/5)) ([f1a7261](https://github.com/telemetry-dev/telemetry/commit/f1a72617ed1f802d68ee71baf767cec934827f98))

### Dependencies

- The following workspace dependencies were updated
  - dependencies
    - @telemetry-dev/otel bumped to 0.1.4

## [0.1.3](https://github.com/telemetry-dev/sdks/compare/sdk-v0.1.2...sdk-v0.1.3) (2026-09-07)

### Dependencies

- The following workspace dependencies were updated
  - dependencies
    - @telemetry-dev/otel bumped to 0.1.3

## [0.1.2](https://github.com/telemetry-dev/sdks/compare/sdk-v0.1.1...sdk-v0.1.2) (2026-09-05)

### Features

- one trace per session, with turn-by-turn session replay ([#226](https://github.com/telemetry-dev/sdks/issues/226)) ([44e2e75](https://github.com/telemetry-dev/sdks/commit/44e2e7510bc5073df89a0135cac9e986aeade5c6))

### Dependencies

- The following workspace dependencies were updated
  - dependencies
    - @telemetry-dev/otel bumped to 0.1.2

## [0.1.1](https://github.com/telemetry-dev/sdks/compare/sdk-v0.1.0...sdk-v0.1.1) (2026-09-02)

### Features

- **mcp:** add transport instrumentation ([#197](https://github.com/telemetry-dev/sdks/issues/197)) ([5ec7f51](https://github.com/telemetry-dev/sdks/commit/5ec7f51c85477fbc1c3359bd1296475c55b569ba))
- **sdk:** add telemetry.dev request header ([#181](https://github.com/telemetry-dev/sdks/issues/181)) ([23d5141](https://github.com/telemetry-dev/sdks/commit/23d5141764b9e7b1640f7ea6f836686146a24822))

### Dependencies

- The following workspace dependencies were updated
  - dependencies
    - @telemetry-dev/otel bumped to 0.1.1
