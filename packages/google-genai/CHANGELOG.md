# Changelog

## [0.1.13](https://github.com/telemetry-dev/telemetry/compare/google-genai-v0.1.12...google-genai-v0.1.13) (2026-10-29)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.12
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.12

## [0.1.12](https://github.com/telemetry-dev/telemetry/compare/google-genai-v0.1.11...google-genai-v0.1.12) (2026-10-29)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.11
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.11

## [0.1.11](https://github.com/telemetry-dev/telemetry/compare/google-genai-v0.1.10...google-genai-v0.1.11) (2026-10-27)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.10
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.10

## [0.1.10](https://github.com/telemetry-dev/telemetry/compare/google-genai-v0.1.9...google-genai-v0.1.10) (2026-10-19)


### Bug Fixes

* **sdk-google-genai:** keep block messages and safety rating text out of spans when output is not captured ([#71](https://github.com/telemetry-dev/telemetry/issues/71)) ([fe2f520](https://github.com/telemetry-dev/telemetry/commit/fe2f520339fae8ccc18ae12d7b18271a111d7961))
* **sdk-google-genai:** keep grounding content out of spans when output is not captured ([#66](https://github.com/telemetry-dev/telemetry/issues/66)) ([7bfacc1](https://github.com/telemetry-dev/telemetry/commit/7bfacc1f9ec2fae3e13032e3f362fd12ff85495d))
* **sdk:** cap the length of retained stream finish reasons ([#58](https://github.com/telemetry-dev/telemetry/issues/58)) ([f423cb4](https://github.com/telemetry-dev/telemetry/commit/f423cb4f602ff032128faf9fadff572d644458a7))


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.9
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.9

## [0.1.9](https://github.com/telemetry-dev/telemetry/compare/google-genai-v0.1.8...google-genai-v0.1.9) (2026-10-14)


### Bug Fixes

* **sdk:** bound per-index stream state and bedrock returnControl ([#55](https://github.com/telemetry-dev/telemetry/issues/55)) ([d4dd31d](https://github.com/telemetry-dev/telemetry/commit/d4dd31de6742173a2ac179e2bf8d18ce62d27ed8))
* **sdk:** keep non-streamed captures cut to the retention bound from masks ([#53](https://github.com/telemetry-dev/telemetry/issues/53)) ([60fdeb3](https://github.com/telemetry-dev/telemetry/commit/60fdeb33c564f9f80435714e8dc9382a6dc20083))
* **sdk:** keep stream retention independent of max_attribute_length ([#49](https://github.com/telemetry-dev/telemetry/issues/49)) ([e4f9dec](https://github.com/telemetry-dev/telemetry/commit/e4f9dec739832d35697f64ec705f5b51d02e0904))


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.8
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.8

## [0.1.8](https://github.com/telemetry-dev/telemetry/compare/google-genai-v0.1.7...google-genai-v0.1.8) (2026-10-01)


### Bug Fixes

* **sdk:** flag incomplete stream capture and keep it from masks ([#33](https://github.com/telemetry-dev/telemetry/issues/33)) ([9454284](https://github.com/telemetry-dev/telemetry/commit/9454284555a2b704c2616620311d36d9db7a9d82))


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.7
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.7

## [0.1.7](https://github.com/telemetry-dev/telemetry/compare/google-genai-v0.1.6...google-genai-v0.1.7) (2026-09-27)

### Features

- **sdk:** trace media, realtime, rerank, and modality token usage ([#17](https://github.com/telemetry-dev/telemetry/issues/17)) ([c489503](https://github.com/telemetry-dev/telemetry/commit/c4895031a27a30004e21dc58f678d0f1ce19c072))

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.6
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.6

## [0.1.6](https://github.com/telemetry-dev/telemetry/compare/google-genai-v0.1.5...google-genai-v0.1.6) (2026-09-20)

### Features

- **ai:** support current provider SDKs ([#10](https://github.com/telemetry-dev/telemetry/issues/10)) ([ed9e664](https://github.com/telemetry-dev/telemetry/commit/ed9e664aaf78d0725f3e86b11a1d818a0785bacb))

## [0.1.5](https://github.com/telemetry-dev/telemetry/compare/google-genai-v0.1.4...google-genai-v0.1.5) (2026-09-12)

### Bug Fixes

- correct package repository metadata ([#7](https://github.com/telemetry-dev/telemetry/issues/7)) ([23f74ea](https://github.com/telemetry-dev/telemetry/commit/23f74ea1480d02f4d3e34380dfb61ce149f12c86))

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.5
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.5

## [0.1.4](https://github.com/telemetry-dev/telemetry/compare/google-genai-v0.1.3...google-genai-v0.1.4) (2026-09-12)

### Features

- **sdk:** add streaming latency metrics ([#5](https://github.com/telemetry-dev/telemetry/issues/5)) ([f1a7261](https://github.com/telemetry-dev/telemetry/commit/f1a72617ed1f802d68ee71baf767cec934827f98))

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.4
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.4

## [0.1.3](https://github.com/telemetry-dev/sdks/compare/google-genai-v0.1.2...google-genai-v0.1.3) (2026-09-07)

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.3
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.3

## [0.1.2](https://github.com/telemetry-dev/sdks/compare/google-genai-v0.1.1...google-genai-v0.1.2) (2026-09-05)

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.2
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.2

## [0.1.1](https://github.com/telemetry-dev/sdks/compare/google-genai-v0.1.0...google-genai-v0.1.1) (2026-09-02)

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.1
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.1
