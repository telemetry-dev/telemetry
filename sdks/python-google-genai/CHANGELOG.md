# Changelog

## [0.1.7](https://github.com/telemetry-dev/telemetry/compare/python-google-genai-v0.1.6...python-google-genai-v0.1.7) (2026-10-19)


### Bug Fixes

* **sdk-google-genai:** keep block messages and safety rating text out of spans when output is not captured ([#71](https://github.com/telemetry-dev/telemetry/issues/71)) ([fe2f520](https://github.com/telemetry-dev/telemetry/commit/fe2f520339fae8ccc18ae12d7b18271a111d7961))
* **sdk-google-genai:** keep grounding content out of spans when output is not captured ([#66](https://github.com/telemetry-dev/telemetry/issues/66)) ([7bfacc1](https://github.com/telemetry-dev/telemetry/commit/7bfacc1f9ec2fae3e13032e3f362fd12ff85495d))
* **sdk:** cap the length of retained stream finish reasons ([#58](https://github.com/telemetry-dev/telemetry/issues/58)) ([f423cb4](https://github.com/telemetry-dev/telemetry/commit/f423cb4f602ff032128faf9fadff572d644458a7))


### Documentation

* link each package readme to its docs and fix stale setup ([#76](https://github.com/telemetry-dev/telemetry/issues/76)) ([dde0b0c](https://github.com/telemetry-dev/telemetry/commit/dde0b0cb0639b1e2dcde31a8a3098ad349f71d23))

## [0.1.6](https://github.com/telemetry-dev/telemetry/compare/python-google-genai-v0.1.5...python-google-genai-v0.1.6) (2026-10-14)


### Bug Fixes

* **sdk:** bound per-index stream state and bedrock returnControl ([#55](https://github.com/telemetry-dev/telemetry/issues/55)) ([d4dd31d](https://github.com/telemetry-dev/telemetry/commit/d4dd31de6742173a2ac179e2bf8d18ce62d27ed8))
* **sdk:** keep stream retention independent of max_attribute_length ([#49](https://github.com/telemetry-dev/telemetry/issues/49)) ([e4f9dec](https://github.com/telemetry-dev/telemetry/commit/e4f9dec739832d35697f64ec705f5b51d02e0904))

## [0.1.5](https://github.com/telemetry-dev/telemetry/compare/python-google-genai-v0.1.4...python-google-genai-v0.1.5) (2026-10-01)


### Bug Fixes

* **sdk:** flag incomplete stream capture and keep it from masks ([#33](https://github.com/telemetry-dev/telemetry/issues/33)) ([9454284](https://github.com/telemetry-dev/telemetry/commit/9454284555a2b704c2616620311d36d9db7a9d82))

## [0.1.4](https://github.com/telemetry-dev/telemetry/compare/python-google-genai-v0.1.3...python-google-genai-v0.1.4) (2026-09-30)


### Maintenance

* refresh compatibility coverage for google-genai 2.25; published dependency ranges are unchanged ([#22](https://github.com/telemetry-dev/telemetry/issues/22)) ([cee7079](https://github.com/telemetry-dev/telemetry/commit/cee707948779667c3fc5f3709f752b7ee3d4ec9b))

## [0.1.3](https://github.com/telemetry-dev/telemetry/compare/python-google-genai-v0.1.2...python-google-genai-v0.1.3) (2026-09-27)


### Features

* **sdk:** trace media, realtime, rerank, and modality token usage ([#17](https://github.com/telemetry-dev/telemetry/issues/17)) ([c489503](https://github.com/telemetry-dev/telemetry/commit/c4895031a27a30004e21dc58f678d0f1ce19c072))

## [0.1.2](https://github.com/telemetry-dev/telemetry/compare/python-google-genai-v0.1.1...python-google-genai-v0.1.2) (2026-09-12)


### Bug Fixes

* correct package repository metadata ([#7](https://github.com/telemetry-dev/telemetry/issues/7)) ([23f74ea](https://github.com/telemetry-dev/telemetry/commit/23f74ea1480d02f4d3e34380dfb61ce149f12c86))

## [0.1.1](https://github.com/telemetry-dev/telemetry/compare/python-google-genai-v0.1.0...python-google-genai-v0.1.1) (2026-09-12)


### Features

* **sdk:** add streaming latency metrics ([#5](https://github.com/telemetry-dev/telemetry/issues/5)) ([f1a7261](https://github.com/telemetry-dev/telemetry/commit/f1a72617ed1f802d68ee71baf767cec934827f98))
