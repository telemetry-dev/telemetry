# Changelog

## [0.2.2](https://github.com/telemetry-dev/telemetry/compare/openai-v0.2.1...openai-v0.2.2) (2026-10-30)


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

## [0.2.1](https://github.com/telemetry-dev/telemetry/compare/openai-v0.2.0...openai-v0.2.1) (2026-10-30)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.2.1
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.2.1

## [0.2.0](https://github.com/telemetry-dev/telemetry/compare/openai-v0.1.14...openai-v0.2.0) (2026-10-29)


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

## [0.1.14](https://github.com/telemetry-dev/telemetry/compare/openai-v0.1.13...openai-v0.1.14) (2026-10-29)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.12
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.12

## [0.1.13](https://github.com/telemetry-dev/telemetry/compare/openai-v0.1.12...openai-v0.1.13) (2026-10-29)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.11
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.11

## [0.1.12](https://github.com/telemetry-dev/telemetry/compare/openai-v0.1.11...openai-v0.1.12) (2026-10-27)


### Bug Fixes

* **sdk:** keep exception text behind capture flags ([#88](https://github.com/telemetry-dev/telemetry/issues/88)) ([e9e49c3](https://github.com/telemetry-dev/telemetry/commit/e9e49c33dfcff28c74f7a6e16126054dc32b0d90))


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.10
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.10

## [0.1.11](https://github.com/telemetry-dev/telemetry/compare/openai-v0.1.10...openai-v0.1.11) (2026-10-19)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.9
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.9

## [0.1.10](https://github.com/telemetry-dev/telemetry/compare/openai-v0.1.9...openai-v0.1.10) (2026-10-14)


### Bug Fixes

* **sdk:** keep non-streamed captures cut to the retention bound from masks ([#53](https://github.com/telemetry-dev/telemetry/issues/53)) ([60fdeb3](https://github.com/telemetry-dev/telemetry/commit/60fdeb33c564f9f80435714e8dc9382a6dc20083))


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.8
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.8

## [0.1.9](https://github.com/telemetry-dev/telemetry/compare/openai-v0.1.8...openai-v0.1.9) (2026-10-01)


### Bug Fixes

* **sdk:** flag incomplete stream capture and keep it from masks ([#33](https://github.com/telemetry-dev/telemetry/issues/33)) ([9454284](https://github.com/telemetry-dev/telemetry/commit/9454284555a2b704c2616620311d36d9db7a9d82))


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.7
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.7

## [0.1.8](https://github.com/telemetry-dev/telemetry/compare/openai-v0.1.7...openai-v0.1.8) (2026-09-30)


### Bug Fixes

* preserve CommonJS stream constructors and detect Azure clients without ESM class identity ([#27](https://github.com/telemetry-dev/telemetry/issues/27)) ([2f490d4](https://github.com/telemetry-dev/telemetry/commit/2f490d43a5da48d3097af4116118f60625965530))
* recognize shell-call command deltas as streamed output ([#22](https://github.com/telemetry-dev/telemetry/issues/22)) ([cee7079](https://github.com/telemetry-dev/telemetry/commit/cee707948779667c3fc5f3709f752b7ee3d4ec9b))

## [0.1.7](https://github.com/telemetry-dev/telemetry/compare/openai-v0.1.6...openai-v0.1.7) (2026-09-27)

### Features

- **sdk:** trace media, realtime, rerank, and modality token usage ([#17](https://github.com/telemetry-dev/telemetry/issues/17)) ([c489503](https://github.com/telemetry-dev/telemetry/commit/c4895031a27a30004e21dc58f678d0f1ce19c072))

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.6
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.6

## [0.1.6](https://github.com/telemetry-dev/telemetry/compare/openai-v0.1.5...openai-v0.1.6) (2026-09-20)

### Features

- **ai:** support current provider SDKs ([#10](https://github.com/telemetry-dev/telemetry/issues/10)) ([ed9e664](https://github.com/telemetry-dev/telemetry/commit/ed9e664aaf78d0725f3e86b11a1d818a0785bacb))

## [0.1.5](https://github.com/telemetry-dev/telemetry/compare/openai-v0.1.4...openai-v0.1.5) (2026-09-12)

### Bug Fixes

- correct package repository metadata ([#7](https://github.com/telemetry-dev/telemetry/issues/7)) ([23f74ea](https://github.com/telemetry-dev/telemetry/commit/23f74ea1480d02f4d3e34380dfb61ce149f12c86))

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.5
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.5

## [0.1.4](https://github.com/telemetry-dev/telemetry/compare/openai-v0.1.3...openai-v0.1.4) (2026-09-12)

### Features

- **sdk:** add streaming latency metrics ([#5](https://github.com/telemetry-dev/telemetry/issues/5)) ([f1a7261](https://github.com/telemetry-dev/telemetry/commit/f1a72617ed1f802d68ee71baf767cec934827f98))

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.4
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.4

## [0.1.3](https://github.com/telemetry-dev/sdks/compare/openai-v0.1.2...openai-v0.1.3) (2026-09-07)

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.3
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.3

## [0.1.2](https://github.com/telemetry-dev/sdks/compare/openai-v0.1.1...openai-v0.1.2) (2026-09-05)

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.2
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.2

## [0.1.1](https://github.com/telemetry-dev/sdks/compare/openai-v0.1.0...openai-v0.1.1) (2026-09-02)

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.1
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.1
