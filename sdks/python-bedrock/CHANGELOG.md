# Changelog

## [0.2.2](https://github.com/telemetry-dev/telemetry/compare/python-bedrock-v0.2.1...python-bedrock-v0.2.2) (2026-10-30)


### Features

* **sdk:** record the prompt of compiled messages outside withPrompt ([6435fb6](https://github.com/telemetry-dev/telemetry/commit/6435fb6d2d7488c968a6ce9f9c6437f2cf426456))

## [0.2.1](https://github.com/telemetry-dev/telemetry/compare/python-bedrock-v0.2.0...python-bedrock-v0.2.1) (2026-10-30)


### Bug Fixes

* **sdk-python:** require telemetry-dev 0.3.0 in the python integrations ([#122](https://github.com/telemetry-dev/telemetry/issues/122)) ([fb6fe58](https://github.com/telemetry-dev/telemetry/commit/fb6fe58926c4078cc19af7b562dfefd833cdfed7))

## [0.2.0](https://github.com/telemetry-dev/telemetry/compare/python-bedrock-v0.1.7...python-bedrock-v0.2.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **sdk:** the default base URL for telemetry export and getPrompt/get_prompt is now https://api.telemetry.dev. https://ingest.telemetry.dev reaches the same service and stays supported; networks that allow only it should set TELEMETRY_DEV_BASE_URL to it.

### Features

* **sdk:** send to api.telemetry.dev by default ([#109](https://github.com/telemetry-dev/telemetry/issues/109)) ([ad8ccf3](https://github.com/telemetry-dev/telemetry/commit/ad8ccf3c41355683deb3b38dab4b604ecaa25a7a))

## [0.1.7](https://github.com/telemetry-dev/telemetry/compare/python-bedrock-v0.1.6...python-bedrock-v0.1.7) (2026-10-27)


### Bug Fixes

* **bedrock:** count compaction usage on invoke model spans ([#77](https://github.com/telemetry-dev/telemetry/issues/77)) ([895456c](https://github.com/telemetry-dev/telemetry/commit/895456cab14e8f9fa1cf7fe26c36403cfbd91ae5))
* **sdk:** align captured JSON and provider usage ([#79](https://github.com/telemetry-dev/telemetry/issues/79)) ([1ec98d5](https://github.com/telemetry-dev/telemetry/commit/1ec98d5014ac810116402736cc7db420745eb9ef))

## [0.1.6](https://github.com/telemetry-dev/telemetry/compare/python-bedrock-v0.1.5...python-bedrock-v0.1.6) (2026-10-19)


### Bug Fixes

* **sdk-bedrock:** keep agent trace content out of spans when it is not captured ([#68](https://github.com/telemetry-dev/telemetry/issues/68)) ([0bc2adf](https://github.com/telemetry-dev/telemetry/commit/0bc2adf2ef43a0b195982e41a3ca7927880ed208))


### Documentation

* link each package readme to its docs and fix stale setup ([#76](https://github.com/telemetry-dev/telemetry/issues/76)) ([dde0b0c](https://github.com/telemetry-dev/telemetry/commit/dde0b0cb0639b1e2dcde31a8a3098ad349f71d23))

## [0.1.5](https://github.com/telemetry-dev/telemetry/compare/python-bedrock-v0.1.4...python-bedrock-v0.1.5) (2026-10-14)


### Bug Fixes

* **sdk:** bound per-index stream state and bedrock returnControl ([#55](https://github.com/telemetry-dev/telemetry/issues/55)) ([d4dd31d](https://github.com/telemetry-dev/telemetry/commit/d4dd31de6742173a2ac179e2bf8d18ce62d27ed8))
* **sdk:** keep stream retention independent of max_attribute_length ([#49](https://github.com/telemetry-dev/telemetry/issues/49)) ([e4f9dec](https://github.com/telemetry-dev/telemetry/commit/e4f9dec739832d35697f64ec705f5b51d02e0904))

## [0.1.4](https://github.com/telemetry-dev/telemetry/compare/python-bedrock-v0.1.3...python-bedrock-v0.1.4) (2026-09-30)


### Bug Fixes

* **sdk:** include cache tokens in input totals ([#26](https://github.com/telemetry-dev/telemetry/issues/26)) ([fa2e833](https://github.com/telemetry-dev/telemetry/commit/fa2e83309e5e67d36abfeaa047c8a5e62da3befe))
* expose `StreamingBody.readinto` only when botocore supports it ([#22](https://github.com/telemetry-dev/telemetry/issues/22)) ([cee7079](https://github.com/telemetry-dev/telemetry/commit/cee707948779667c3fc5f3709f752b7ee3d4ec9b))

## [0.1.3](https://github.com/telemetry-dev/telemetry/compare/python-bedrock-v0.1.2...python-bedrock-v0.1.3) (2026-09-27)


### Features

* **sdk:** trace media, realtime, rerank, and modality token usage ([#17](https://github.com/telemetry-dev/telemetry/issues/17)) ([c489503](https://github.com/telemetry-dev/telemetry/commit/c4895031a27a30004e21dc58f678d0f1ce19c072))

## [0.1.2](https://github.com/telemetry-dev/telemetry/compare/python-bedrock-v0.1.1...python-bedrock-v0.1.2) (2026-09-12)


### Bug Fixes

* correct package repository metadata ([#7](https://github.com/telemetry-dev/telemetry/issues/7)) ([23f74ea](https://github.com/telemetry-dev/telemetry/commit/23f74ea1480d02f4d3e34380dfb61ce149f12c86))

## [0.1.1](https://github.com/telemetry-dev/telemetry/compare/python-bedrock-v0.1.0...python-bedrock-v0.1.1) (2026-09-12)


### Features

* **sdk:** add streaming latency metrics ([#5](https://github.com/telemetry-dev/telemetry/issues/5)) ([f1a7261](https://github.com/telemetry-dev/telemetry/commit/f1a72617ed1f802d68ee71baf767cec934827f98))
