# Changelog

## [0.2.2](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.2.1...anthropic-v0.2.2) (2026-10-30)


### Features

* **sdk:** record the prompt of compiled messages outside withPrompt ([6435fb6](https://github.com/telemetry-dev/telemetry/commit/6435fb6d2d7488c968a6ce9f9c6437f2cf426456))
* **sdk:** record the prompt on model calls outside withPrompt ([339eb0c](https://github.com/telemetry-dev/telemetry/commit/339eb0c01c5754191eb7d1b5ec37db21ebd28855)), closes [#97](https://github.com/telemetry-dev/telemetry/issues/97)


### Bug Fixes

* **sdk:** find the prompt by message identity, not by text ([ae47605](https://github.com/telemetry-dev/telemetry/commit/ae47605555ef9f5dfed1d6f762ac56823e692edb))
* **sdk:** give the base url to getPrompt in the prompt tests ([abb8d10](https://github.com/telemetry-dev/telemetry/commit/abb8d10f680d1fa7ace6913a1d53b43a3e0db703)), closes [#97](https://github.com/telemetry-dev/telemetry/issues/97)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.2.2
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.2.2

## [0.2.1](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.2.0...anthropic-v0.2.1) (2026-10-30)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.2.1
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.2.1

## [0.2.0](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.1.15...anthropic-v0.2.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **sdk:** the default base URL for telemetry export and getPrompt/get_prompt is now https://api.telemetry.dev. https://ingest.telemetry.dev reaches the same service and stays supported; networks that allow only it should set TELEMETRY_DEV_BASE_URL to it.

### Features

* **sdk:** send to api.telemetry.dev by default ([#109](https://github.com/telemetry-dev/telemetry/issues/109)) ([ad8ccf3](https://github.com/telemetry-dev/telemetry/commit/ad8ccf3c41355683deb3b38dab4b604ecaa25a7a))


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.2.0
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.2.0

## [0.1.15](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.1.14...anthropic-v0.1.15) (2026-10-29)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.12
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.12

## [0.1.14](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.1.13...anthropic-v0.1.14) (2026-10-29)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.11
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.11

## [0.1.13](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.1.12...anthropic-v0.1.13) (2026-10-28)


### Features

* **sdk-anthropic:** record request tool names as tool definitions ([#95](https://github.com/telemetry-dev/telemetry/issues/95)) ([f075e8c](https://github.com/telemetry-dev/telemetry/commit/f075e8cf5db664ddaf0fc4a9e690a1a4582a216e))

## [0.1.12](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.1.11...anthropic-v0.1.12) (2026-10-27)


### Bug Fixes

* **sdk:** align captured JSON and provider usage ([#79](https://github.com/telemetry-dev/telemetry/issues/79)) ([1ec98d5](https://github.com/telemetry-dev/telemetry/commit/1ec98d5014ac810116402736cc7db420745eb9ef))
* **sdk:** keep exception text behind capture flags ([#88](https://github.com/telemetry-dev/telemetry/issues/88)) ([e9e49c3](https://github.com/telemetry-dev/telemetry/commit/e9e49c33dfcff28c74f7a6e16126054dc32b0d90))


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.10
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.10

## [0.1.11](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.1.10...anthropic-v0.1.11) (2026-10-19)


### Bug Fixes

* **anthropic:** count compaction usage and one-hour cache writes ([#64](https://github.com/telemetry-dev/telemetry/issues/64)) ([59cafca](https://github.com/telemetry-dev/telemetry/commit/59cafca68884a912f8c8da92a027c4afea03c549))


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.9
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.9

## [0.1.10](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.1.9...anthropic-v0.1.10) (2026-10-14)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.8
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.8

## [0.1.9](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.1.8...anthropic-v0.1.9) (2026-10-01)


### Bug Fixes

* **sdk:** flag incomplete stream capture and keep it from masks ([#33](https://github.com/telemetry-dev/telemetry/issues/33)) ([9454284](https://github.com/telemetry-dev/telemetry/commit/9454284555a2b704c2616620311d36d9db7a9d82))


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.7
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.7

## [0.1.8](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.1.7...anthropic-v0.1.8) (2026-09-30)


### Features

* **anthropic:** trace beta.messages and messages.parse ([#15](https://github.com/telemetry-dev/telemetry/issues/15)) ([d1dd58c](https://github.com/telemetry-dev/telemetry/commit/d1dd58ca4f3294a97011e3fcbf02413a0a76099e))


### Bug Fixes

* **sdk-anthropic:** account for tool input when replacing streamed blocks ([#21](https://github.com/telemetry-dev/telemetry/issues/21)) ([6741f18](https://github.com/telemetry-dev/telemetry/commit/6741f187d3347b88dc07c29195d5f157e892e789))
* **sdk-anthropic:** attribute Bedrock Mantle clients to aws.bedrock ([#23](https://github.com/telemetry-dev/telemetry/issues/23)) ([356fe49](https://github.com/telemetry-dev/telemetry/commit/356fe49672206cb4b2f3f1b2357c31250632fed0))
* **sdk:** bound streamed capture and preserve provider compatibility ([#27](https://github.com/telemetry-dev/telemetry/issues/27)) ([2f490d4](https://github.com/telemetry-dev/telemetry/commit/2f490d43a5da48d3097af4116118f60625965530))
* **sdk:** include cache tokens in input totals ([#26](https://github.com/telemetry-dev/telemetry/issues/26)) ([fa2e833](https://github.com/telemetry-dev/telemetry/commit/fa2e83309e5e67d36abfeaa047c8a5e62da3befe))

## [0.1.7](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.1.6...anthropic-v0.1.7) (2026-09-27)

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.6
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.6

## [0.1.6](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.1.5...anthropic-v0.1.6) (2026-09-20)

### Features

- **ai:** support current provider SDKs ([#10](https://github.com/telemetry-dev/telemetry/issues/10)) ([ed9e664](https://github.com/telemetry-dev/telemetry/commit/ed9e664aaf78d0725f3e86b11a1d818a0785bacb))

## [0.1.5](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.1.4...anthropic-v0.1.5) (2026-09-12)

### Bug Fixes

- correct package repository metadata ([#7](https://github.com/telemetry-dev/telemetry/issues/7)) ([23f74ea](https://github.com/telemetry-dev/telemetry/commit/23f74ea1480d02f4d3e34380dfb61ce149f12c86))

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.5
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.5

## [0.1.4](https://github.com/telemetry-dev/telemetry/compare/anthropic-v0.1.3...anthropic-v0.1.4) (2026-09-12)

### Features

- **sdk:** add streaming latency metrics ([#5](https://github.com/telemetry-dev/telemetry/issues/5)) ([f1a7261](https://github.com/telemetry-dev/telemetry/commit/f1a72617ed1f802d68ee71baf767cec934827f98))

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.4
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.4

## [0.1.3](https://github.com/telemetry-dev/sdks/compare/anthropic-v0.1.2...anthropic-v0.1.3) (2026-09-07)

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.3
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.3

## [0.1.2](https://github.com/telemetry-dev/sdks/compare/anthropic-v0.1.1...anthropic-v0.1.2) (2026-09-05)

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.2
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.2

## [0.1.1](https://github.com/telemetry-dev/sdks/compare/anthropic-v0.1.0...anthropic-v0.1.1) (2026-09-02)

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.1
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.1
