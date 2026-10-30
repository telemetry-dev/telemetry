# Changelog

## [0.2.2](https://github.com/telemetry-dev/telemetry/compare/python-anthropic-v0.2.1...python-anthropic-v0.2.2) (2026-10-30)


### Features

* **sdk:** record the prompt of compiled messages outside withPrompt ([6435fb6](https://github.com/telemetry-dev/telemetry/commit/6435fb6d2d7488c968a6ce9f9c6437f2cf426456))

## [0.2.1](https://github.com/telemetry-dev/telemetry/compare/python-anthropic-v0.2.0...python-anthropic-v0.2.1) (2026-10-30)


### Bug Fixes

* **sdk-python:** require telemetry-dev 0.3.0 in the python integrations ([#122](https://github.com/telemetry-dev/telemetry/issues/122)) ([fb6fe58](https://github.com/telemetry-dev/telemetry/commit/fb6fe58926c4078cc19af7b562dfefd833cdfed7))

## [0.2.0](https://github.com/telemetry-dev/telemetry/compare/python-anthropic-v0.1.8...python-anthropic-v0.2.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **sdk:** the default base URL for telemetry export and getPrompt/get_prompt is now https://api.telemetry.dev. https://ingest.telemetry.dev reaches the same service and stays supported; networks that allow only it should set TELEMETRY_DEV_BASE_URL to it.

### Features

* **sdk:** send to api.telemetry.dev by default ([#109](https://github.com/telemetry-dev/telemetry/issues/109)) ([ad8ccf3](https://github.com/telemetry-dev/telemetry/commit/ad8ccf3c41355683deb3b38dab4b604ecaa25a7a))

## [0.1.8](https://github.com/telemetry-dev/telemetry/compare/python-anthropic-v0.1.7...python-anthropic-v0.1.8) (2026-10-28)


### Features

* **sdk-anthropic:** record request tool names as tool definitions ([#95](https://github.com/telemetry-dev/telemetry/issues/95)) ([f075e8c](https://github.com/telemetry-dev/telemetry/commit/f075e8cf5db664ddaf0fc4a9e690a1a4582a216e))

## [0.1.7](https://github.com/telemetry-dev/telemetry/compare/python-anthropic-v0.1.6...python-anthropic-v0.1.7) (2026-10-27)


### Bug Fixes

* **sdk-anthropic:** bound stream field alias inspection ([3b45e21](https://github.com/telemetry-dev/telemetry/commit/3b45e2173dbbecb0d06cbf9a401aaad18aec397f))
* **sdk-anthropic:** record fallback blocks under their wire keys ([cb37f76](https://github.com/telemetry-dev/telemetry/commit/cb37f7671973ca08be464e0ea65296a3f5cb5796))
* **sdk:** align captured JSON and provider usage ([#79](https://github.com/telemetry-dev/telemetry/issues/79)) ([1ec98d5](https://github.com/telemetry-dev/telemetry/commit/1ec98d5014ac810116402736cc7db420745eb9ef))
* **sdk:** keep exception text behind capture flags ([#88](https://github.com/telemetry-dev/telemetry/issues/88)) ([e9e49c3](https://github.com/telemetry-dev/telemetry/commit/e9e49c33dfcff28c74f7a6e16126054dc32b0d90))

## [0.1.6](https://github.com/telemetry-dev/telemetry/compare/python-anthropic-v0.1.5...python-anthropic-v0.1.6) (2026-10-19)


### Bug Fixes

* **anthropic:** count compaction usage and one-hour cache writes ([#64](https://github.com/telemetry-dev/telemetry/issues/64)) ([59cafca](https://github.com/telemetry-dev/telemetry/commit/59cafca68884a912f8c8da92a027c4afea03c549))


### Documentation

* document native claude sdk spans ([#61](https://github.com/telemetry-dev/telemetry/issues/61)) ([e280bcc](https://github.com/telemetry-dev/telemetry/commit/e280bccab3960f1e8388d6cddb5361b0f947e081))
* link each package readme to its docs and fix stale setup ([#76](https://github.com/telemetry-dev/telemetry/issues/76)) ([dde0b0c](https://github.com/telemetry-dev/telemetry/commit/dde0b0cb0639b1e2dcde31a8a3098ad349f71d23))

## [0.1.5](https://github.com/telemetry-dev/telemetry/compare/python-anthropic-v0.1.4...python-anthropic-v0.1.5) (2026-10-14)


### Bug Fixes

* **python-anthropic:** keep raw tool input charged after a block settles ([159554d](https://github.com/telemetry-dev/telemetry/commit/159554d4ed4c6c3897678eaa8f0d2a7fc8d47af4))
* **python-anthropic:** keep retained stream text encodable and settle against the final check ([40e850d](https://github.com/telemetry-dev/telemetry/commit/40e850d10878aac9ea4a87078d6a172d3d96cdaf)), closes [#32](https://github.com/telemetry-dev/telemetry/issues/32)
* **python-anthropic:** keep stream capture accounting consistent with serialization ([751e5df](https://github.com/telemetry-dev/telemetry/commit/751e5df16fa016e361f05e5e24dd61acb9be3351))
* **python-anthropic:** keep stream capture accounting consistent with serialization ([23e7f6a](https://github.com/telemetry-dev/telemetry/commit/23e7f6a5e4e408b70434719c39d9fdc23f384ba3)), closes [#32](https://github.com/telemetry-dev/telemetry/issues/32)
* **python-anthropic:** keep the settled measure when a tool block is replaced ([284330e](https://github.com/telemetry-dev/telemetry/commit/284330e36b56af225d8a7b60c034b4e6caa0fb02)), closes [#32](https://github.com/telemetry-dev/telemetry/issues/32)
* **python-anthropic:** settle tool blocks the way the final check counts them ([17479bd](https://github.com/telemetry-dev/telemetry/commit/17479bd0e575140db14cae3a618501726738be01)), closes [#32](https://github.com/telemetry-dev/telemetry/issues/32)
* **sdk-python:** charge json escapes when CaptureBudget measures strings ([#48](https://github.com/telemetry-dev/telemetry/issues/48)) ([3d5daf5](https://github.com/telemetry-dev/telemetry/commit/3d5daf578bbcdfe4113bea62b55fceb88dd10418))


### Performance Improvements

* **python-anthropic:** scan stream strings for surrogates with a regex ([#51](https://github.com/telemetry-dev/telemetry/issues/51)) ([8cc0bb0](https://github.com/telemetry-dev/telemetry/commit/8cc0bb03f684643f5bf3e40e593b87147d2d3aaa))

## [0.1.4](https://github.com/telemetry-dev/telemetry/compare/python-anthropic-v0.1.3...python-anthropic-v0.1.4) (2026-10-01)


### Bug Fixes

* **sdk:** flag incomplete stream capture and keep it from masks ([#33](https://github.com/telemetry-dev/telemetry/issues/33)) ([9454284](https://github.com/telemetry-dev/telemetry/commit/9454284555a2b704c2616620311d36d9db7a9d82))

## [0.1.3](https://github.com/telemetry-dev/telemetry/compare/python-anthropic-v0.1.2...python-anthropic-v0.1.3) (2026-09-30)


### Features

* **anthropic:** trace beta.messages and messages.parse ([#15](https://github.com/telemetry-dev/telemetry/issues/15)) ([d1dd58c](https://github.com/telemetry-dev/telemetry/commit/d1dd58ca4f3294a97011e3fcbf02413a0a76099e))


### Bug Fixes

* **sdk-anthropic:** account for tool input when replacing streamed blocks ([#21](https://github.com/telemetry-dev/telemetry/issues/21)) ([6741f18](https://github.com/telemetry-dev/telemetry/commit/6741f187d3347b88dc07c29195d5f157e892e789))
* **sdk-anthropic:** attribute Bedrock Mantle clients to aws.bedrock ([#23](https://github.com/telemetry-dev/telemetry/issues/23)) ([356fe49](https://github.com/telemetry-dev/telemetry/commit/356fe49672206cb4b2f3f1b2357c31250632fed0))
* **sdk:** bound streamed capture and preserve provider compatibility ([#27](https://github.com/telemetry-dev/telemetry/issues/27)) ([2f490d4](https://github.com/telemetry-dev/telemetry/commit/2f490d43a5da48d3097af4116118f60625965530))
* **sdk:** include cache tokens in input totals ([#26](https://github.com/telemetry-dev/telemetry/issues/26)) ([fa2e833](https://github.com/telemetry-dev/telemetry/commit/fa2e83309e5e67d36abfeaa047c8a5e62da3befe))
* **sdk:** support anthropic 1.x and openai 3.x in the Python integrations ([#22](https://github.com/telemetry-dev/telemetry/issues/22)) ([cee7079](https://github.com/telemetry-dev/telemetry/commit/cee707948779667c3fc5f3709f752b7ee3d4ec9b))

## [0.1.2](https://github.com/telemetry-dev/telemetry/compare/python-anthropic-v0.1.1...python-anthropic-v0.1.2) (2026-09-12)


### Bug Fixes

* correct package repository metadata ([#7](https://github.com/telemetry-dev/telemetry/issues/7)) ([23f74ea](https://github.com/telemetry-dev/telemetry/commit/23f74ea1480d02f4d3e34380dfb61ce149f12c86))

## [0.1.1](https://github.com/telemetry-dev/telemetry/compare/python-anthropic-v0.1.0...python-anthropic-v0.1.1) (2026-09-12)


### Features

* **sdk:** add streaming latency metrics ([#5](https://github.com/telemetry-dev/telemetry/issues/5)) ([f1a7261](https://github.com/telemetry-dev/telemetry/commit/f1a72617ed1f802d68ee71baf767cec934827f98))
