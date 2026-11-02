# Changelog

## [0.1.7](https://github.com/telemetry-dev/telemetry/compare/cli-v0.1.6...cli-v0.1.7) (2026-11-02)


### Performance Improvements

* **cli:** shorten deep failure paths without measuring each drop ([#147](https://github.com/telemetry-dev/telemetry/issues/147)) ([ee260e5](https://github.com/telemetry-dev/telemetry/commit/ee260e5f9741964ea63813538799fb819bed87cf))

## [0.1.6](https://github.com/telemetry-dev/telemetry/compare/cli-v0.1.5...cli-v0.1.6) (2026-10-31)


### Features

* **cli:** filter traces by agent and tool ([#143](https://github.com/telemetry-dev/telemetry/issues/143)) ([8651565](https://github.com/telemetry-dev/telemetry/commit/8651565c68c520405d917614e0aa673566e68426))

## [0.1.5](https://github.com/telemetry-dev/telemetry/compare/cli-v0.1.4...cli-v0.1.5) (2026-10-30)


### Bug Fixes

* **cli:** say how sessions orders its results ([#140](https://github.com/telemetry-dev/telemetry/issues/140)) ([ebf7fdf](https://github.com/telemetry-dev/telemetry/commit/ebf7fdf904cee4ab6b38a1925926f69d899ecc73))

## [0.1.4](https://github.com/telemetry-dev/telemetry/compare/cli-v0.1.3...cli-v0.1.4) (2026-10-30)


### Features

* **cli:** add --since, --until and trace filters to traces and sessions ([#137](https://github.com/telemetry-dev/telemetry/issues/137)) ([4aa30a4](https://github.com/telemetry-dev/telemetry/commit/4aa30a4cf26a795f53c368248560f60b02f91b01))
* **cli:** link a trace to its dashboard page ([#136](https://github.com/telemetry-dev/telemetry/issues/136)) ([7848f15](https://github.com/telemetry-dev/telemetry/commit/7848f157c1ddc53192a8ceabf3e8eb179c14820c))
* **cli:** page logs and give it --since and --until ([#138](https://github.com/telemetry-dev/telemetry/issues/138)) ([82c8d1b](https://github.com/telemetry-dev/telemetry/commit/82c8d1b49ee69223a818a55b624219fe3e5524d5))

## [0.1.3](https://github.com/telemetry-dev/telemetry/compare/cli-v0.1.2...cli-v0.1.3) (2026-10-30)


### Bug Fixes

* **cli:** break the sign-in lock only when its holder has exited ([#124](https://github.com/telemetry-dev/telemetry/issues/124)) ([bb3369d](https://github.com/telemetry-dev/telemetry/commit/bb3369d276b287b09aaf2b749c7f991321d9508f))

## [0.1.2](https://github.com/telemetry-dev/telemetry/compare/cli-v0.1.1...cli-v0.1.2) (2026-10-30)


### Features

* **cli:** add --json to login and logout ([#131](https://github.com/telemetry-dev/telemetry/issues/131)) ([505309b](https://github.com/telemetry-dev/telemetry/commit/505309bf84c00edf3036488348086e10a5870510))
* **cli:** map telemetry.dev's tool error codes to cli error codes ([#134](https://github.com/telemetry-dev/telemetry/issues/134)) ([b639de9](https://github.com/telemetry-dev/telemetry/commit/b639de9f388ed3ce42335b82cff8728c61c2d322))

## [0.1.1](https://github.com/telemetry-dev/telemetry/compare/cli-v0.1.0...cli-v0.1.1) (2026-10-30)


### Features

* **cli:** report errors as one JSON line with a stable code under --json ([#127](https://github.com/telemetry-dev/telemetry/issues/127)) ([dcad14b](https://github.com/telemetry-dev/telemetry/commit/dcad14b08926862409cfaf94c529ecfab8df26e5))
* **cli:** summarize a trace's failures and repeated calls, and suggest next commands ([#128](https://github.com/telemetry-dev/telemetry/issues/128)) ([02f63f5](https://github.com/telemetry-dev/telemetry/commit/02f63f5bb3ac5fc3c3c28d8198c0605961768077))

## 0.1.0 (2026-10-30)


### Features

* **cli:** add the telemetry command line ([#111](https://github.com/telemetry-dev/telemetry/issues/111)) ([a3d3379](https://github.com/telemetry-dev/telemetry/commit/a3d337911e538b1046afc85561fe3bc5e2792212))
* **cli:** sign in with telemetry.dev's client id metadata document ([#119](https://github.com/telemetry-dev/telemetry/issues/119)) ([52135ef](https://github.com/telemetry-dev/telemetry/commit/52135ef4f6da4f77a728cfffe855df80f7fd554f))


### Bug Fixes

* **cli:** revoke the sign-in that a new login replaces ([#118](https://github.com/telemetry-dev/telemetry/issues/118)) ([9150f57](https://github.com/telemetry-dev/telemetry/commit/9150f571ffcbcfc7d225b1cc4c1e46e6f64337e6))
