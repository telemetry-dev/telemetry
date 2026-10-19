# Changelog

## [0.1.8](https://github.com/telemetry-dev/telemetry/compare/python-openai-v0.1.7...python-openai-v0.1.8) (2026-10-19)


### Documentation

* link each package readme to its docs and fix stale setup ([#76](https://github.com/telemetry-dev/telemetry/issues/76)) ([dde0b0c](https://github.com/telemetry-dev/telemetry/commit/dde0b0cb0639b1e2dcde31a8a3098ad349f71d23))

## [0.1.7](https://github.com/telemetry-dev/telemetry/compare/python-openai-v0.1.6...python-openai-v0.1.7) (2026-10-14)


### Bug Fixes

* **python-anthropic:** keep stream capture accounting consistent with serialization ([751e5df](https://github.com/telemetry-dev/telemetry/commit/751e5df16fa016e361f05e5e24dd61acb9be3351))
* **python-openai:** derive the remaining chat structure costs from the core ([a13a9d1](https://github.com/telemetry-dev/telemetry/commit/a13a9d19bcfa7b7cd1d9c2e0accacdd3a9b8c08e)), closes [#32](https://github.com/telemetry-dev/telemetry/issues/32)
* **python-openai:** measure captured strings one way ([6666f47](https://github.com/telemetry-dev/telemetry/commit/6666f472bcf988d4646ca4dbfc0eaed4e4c2c8e9)), closes [#32](https://github.com/telemetry-dev/telemetry/issues/32)
* **sdk-python:** charge json escapes when CaptureBudget measures strings ([#48](https://github.com/telemetry-dev/telemetry/issues/48)) ([3d5daf5](https://github.com/telemetry-dev/telemetry/commit/3d5daf578bbcdfe4113bea62b55fceb88dd10418))


### Documentation

* **sdk-openai:** note the openai videos api shutdown ([#44](https://github.com/telemetry-dev/telemetry/issues/44)) ([3df0536](https://github.com/telemetry-dev/telemetry/commit/3df0536c572a04566e22ee79e09075c6c768974b))

## [0.1.6](https://github.com/telemetry-dev/telemetry/compare/python-openai-v0.1.5...python-openai-v0.1.6) (2026-10-01)


### Bug Fixes

* **sdk:** flag incomplete stream capture and keep it from masks ([#33](https://github.com/telemetry-dev/telemetry/issues/33)) ([9454284](https://github.com/telemetry-dev/telemetry/commit/9454284555a2b704c2616620311d36d9db7a9d82))

## [0.1.5](https://github.com/telemetry-dev/telemetry/compare/python-openai-v0.1.4...python-openai-v0.1.5) (2026-09-30)


### Bug Fixes

* **sdk:** bound streamed capture and preserve provider compatibility ([#27](https://github.com/telemetry-dev/telemetry/issues/27)) ([2f490d4](https://github.com/telemetry-dev/telemetry/commit/2f490d43a5da48d3097af4116118f60625965530))
* **sdk:** support anthropic 1.x and openai 3.x in the Python integrations ([#22](https://github.com/telemetry-dev/telemetry/issues/22)) ([cee7079](https://github.com/telemetry-dev/telemetry/commit/cee707948779667c3fc5f3709f752b7ee3d4ec9b))

## [0.1.4](https://github.com/telemetry-dev/telemetry/compare/python-openai-v0.1.3...python-openai-v0.1.4) (2026-09-27)


### Features

* **sdk:** trace media, realtime, rerank, and modality token usage ([#17](https://github.com/telemetry-dev/telemetry/issues/17)) ([c489503](https://github.com/telemetry-dev/telemetry/commit/c4895031a27a30004e21dc58f678d0f1ce19c072))

## [0.1.3](https://github.com/telemetry-dev/telemetry/compare/python-openai-v0.1.2...python-openai-v0.1.3) (2026-09-12)


### Bug Fixes

* correct package repository metadata ([#7](https://github.com/telemetry-dev/telemetry/issues/7)) ([23f74ea](https://github.com/telemetry-dev/telemetry/commit/23f74ea1480d02f4d3e34380dfb61ce149f12c86))

## [0.1.2](https://github.com/telemetry-dev/telemetry/compare/python-openai-v0.1.1...python-openai-v0.1.2) (2026-09-12)


### Features

* **sdk:** add streaming latency metrics ([#5](https://github.com/telemetry-dev/telemetry/issues/5)) ([f1a7261](https://github.com/telemetry-dev/telemetry/commit/f1a72617ed1f802d68ee71baf767cec934827f98))

## [0.1.1](https://github.com/telemetry-dev/sdks/compare/python-openai-v0.1.0...python-openai-v0.1.1) (2026-09-05)


### Features

* one trace per session, with turn-by-turn session replay ([#226](https://github.com/telemetry-dev/sdks/issues/226)) ([44e2e75](https://github.com/telemetry-dev/sdks/commit/44e2e7510bc5073df89a0135cac9e986aeade5c6))
