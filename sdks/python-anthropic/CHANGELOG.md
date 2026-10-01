# Changelog

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
