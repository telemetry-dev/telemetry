# Changelog

## [0.2.10](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.9...python-v0.2.10) (2026-10-19)


### Features

* **sdk:** run the configured mask on integration content and in the ai package ([#73](https://github.com/telemetry-dev/telemetry/issues/73)) ([365b6e7](https://github.com/telemetry-dev/telemetry/commit/365b6e76ebbccd3bb63803d0abb1757b7d16573d))


### Documentation

* document native claude sdk spans ([#61](https://github.com/telemetry-dev/telemetry/issues/61)) ([e280bcc](https://github.com/telemetry-dev/telemetry/commit/e280bccab3960f1e8388d6cddb5361b0f947e081))
* link each package readme to its docs and fix stale setup ([#76](https://github.com/telemetry-dev/telemetry/issues/76)) ([dde0b0c](https://github.com/telemetry-dev/telemetry/commit/dde0b0cb0639b1e2dcde31a8a3098ad349f71d23))

## [0.2.9](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.8...python-v0.2.9) (2026-10-14)


### Bug Fixes

* **sdk-python:** charge json escapes when CaptureBudget measures strings ([#48](https://github.com/telemetry-dev/telemetry/issues/48)) ([3d5daf5](https://github.com/telemetry-dev/telemetry/commit/3d5daf578bbcdfe4113bea62b55fceb88dd10418))

## [0.2.8](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.7...python-v0.2.8) (2026-10-01)


### Bug Fixes

* **sdk:** flag incomplete stream capture and keep it from masks ([#33](https://github.com/telemetry-dev/telemetry/issues/33)) ([9454284](https://github.com/telemetry-dev/telemetry/commit/9454284555a2b704c2616620311d36d9db7a9d82))

## [0.2.7](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.6...python-v0.2.7) (2026-09-30)


### Bug Fixes

* require OpenTelemetry 1.39 or newer and drop SDK-internal metric-reader telemetry ([#22](https://github.com/telemetry-dev/telemetry/issues/22)) ([cee7079](https://github.com/telemetry-dev/telemetry/commit/cee707948779667c3fc5f3709f752b7ee3d4ec9b))

## [0.2.6](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.5...python-v0.2.6) (2026-09-27)


### Features

* **sdk:** trace media, realtime, rerank, and modality token usage ([#17](https://github.com/telemetry-dev/telemetry/issues/17)) ([c489503](https://github.com/telemetry-dev/telemetry/commit/c4895031a27a30004e21dc58f678d0f1ce19c072))

## [0.2.5](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.4...python-v0.2.5) (2026-09-12)


### Bug Fixes

* correct package repository metadata ([#7](https://github.com/telemetry-dev/telemetry/issues/7)) ([23f74ea](https://github.com/telemetry-dev/telemetry/commit/23f74ea1480d02f4d3e34380dfb61ce149f12c86))

## [0.2.4](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.3...python-v0.2.4) (2026-09-12)


### Features

* **sdk:** add streaming latency metrics ([#5](https://github.com/telemetry-dev/telemetry/issues/5)) ([f1a7261](https://github.com/telemetry-dev/telemetry/commit/f1a72617ed1f802d68ee71baf767cec934827f98))

## [0.2.3](https://github.com/telemetry-dev/sdks/compare/python-v0.2.2...python-v0.2.3) (2026-09-07)


### Bug Fixes

* include shared instrumentation and capture refactors in published SDKs ([7c82386](https://github.com/telemetry-dev/sdks/commit/7c82386e139303a49f4cb3b6a39affac85763e88))

## [0.2.2](https://github.com/telemetry-dev/sdks/compare/python-v0.2.1...python-v0.2.2) (2026-09-05)


### Features

* one trace per session, with turn-by-turn session replay ([#226](https://github.com/telemetry-dev/sdks/issues/226)) ([44e2e75](https://github.com/telemetry-dev/sdks/commit/44e2e7510bc5073df89a0135cac9e986aeade5c6))

## [0.2.1](https://github.com/telemetry-dev/sdks/compare/python-v0.2.0...python-v0.2.1) (2026-09-02)


### Features

* **sdk:** add telemetry.dev request header ([#181](https://github.com/telemetry-dev/sdks/issues/181)) ([23d5141](https://github.com/telemetry-dev/sdks/commit/23d5141764b9e7b1640f7ea6f836686146a24822))
