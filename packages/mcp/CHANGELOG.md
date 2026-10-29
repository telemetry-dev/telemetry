# Changelog

## [0.2.0](https://github.com/telemetry-dev/telemetry/compare/mcp-v0.1.12...mcp-v0.2.0) (2026-10-29)


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

## [0.1.12](https://github.com/telemetry-dev/telemetry/compare/mcp-v0.1.11...mcp-v0.1.12) (2026-10-29)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.12
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.12

## [0.1.11](https://github.com/telemetry-dev/telemetry/compare/mcp-v0.1.10...mcp-v0.1.11) (2026-10-29)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.11
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.11

## [0.1.10](https://github.com/telemetry-dev/telemetry/compare/mcp-v0.1.9...mcp-v0.1.10) (2026-10-27)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.10
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.10

## [0.1.9](https://github.com/telemetry-dev/telemetry/compare/mcp-v0.1.8...mcp-v0.1.9) (2026-10-19)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.9
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.9

## [0.1.8](https://github.com/telemetry-dev/telemetry/compare/mcp-v0.1.7...mcp-v0.1.8) (2026-10-14)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.8
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.8

## [0.1.7](https://github.com/telemetry-dev/telemetry/compare/mcp-v0.1.6...mcp-v0.1.7) (2026-10-01)


### Dependencies

* The following workspace dependencies were updated
  * devDependencies
    * @telemetry-dev/sdk bumped to 0.1.7
  * peerDependencies
    * @telemetry-dev/sdk bumped to 0.1.7

## [0.1.6](https://github.com/telemetry-dev/telemetry/compare/mcp-v0.1.5...mcp-v0.1.6) (2026-09-27)

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.6
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.6

## [0.1.5](https://github.com/telemetry-dev/telemetry/compare/mcp-v0.1.4...mcp-v0.1.5) (2026-09-12)

### Bug Fixes

- correct package repository metadata ([#7](https://github.com/telemetry-dev/telemetry/issues/7)) ([23f74ea](https://github.com/telemetry-dev/telemetry/commit/23f74ea1480d02f4d3e34380dfb61ce149f12c86))

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.5
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.5

## [0.1.4](https://github.com/telemetry-dev/telemetry/compare/mcp-v0.1.3...mcp-v0.1.4) (2026-09-12)

### Features

- **sdk:** add streaming latency metrics ([#5](https://github.com/telemetry-dev/telemetry/issues/5)) ([f1a7261](https://github.com/telemetry-dev/telemetry/commit/f1a72617ed1f802d68ee71baf767cec934827f98))

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.4
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.4

## [0.1.3](https://github.com/telemetry-dev/sdks/compare/mcp-v0.1.2...mcp-v0.1.3) (2026-09-07)

### Bug Fixes

- include shared instrumentation and capture refactors in published SDKs ([7c82386](https://github.com/telemetry-dev/sdks/commit/7c82386e139303a49f4cb3b6a39affac85763e88))

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.3
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.3

## [0.1.2](https://github.com/telemetry-dev/sdks/compare/mcp-v0.1.1...mcp-v0.1.2) (2026-09-05)

### Bug Fixes

- **mcp:** complete responses before transport close ([#227](https://github.com/telemetry-dev/sdks/issues/227)) ([1739b40](https://github.com/telemetry-dev/sdks/commit/1739b40cb50ef4d8b2ff82618723780285ce5a51))

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.2
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.2

## [0.1.1](https://github.com/telemetry-dev/sdks/compare/mcp-v0.1.0...mcp-v0.1.1) (2026-09-02)

### Features

- **mcp:** add transport instrumentation ([#197](https://github.com/telemetry-dev/sdks/issues/197)) ([5ec7f51](https://github.com/telemetry-dev/sdks/commit/5ec7f51c85477fbc1c3359bd1296475c55b569ba))

### Dependencies

- The following workspace dependencies were updated
  - devDependencies
    - @telemetry-dev/sdk bumped to 0.1.1
  - peerDependencies
    - @telemetry-dev/sdk bumped to 0.1.1
