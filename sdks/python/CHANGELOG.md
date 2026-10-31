# Changelog

## [0.3.3](https://github.com/telemetry-dev/telemetry/compare/python-v0.3.2...python-v0.3.3) (2026-10-31)


### Bug Fixes

* **sdk-python:** recover prompt fetches and the cache lock in forked children ([#159](https://github.com/telemetry-dev/telemetry/issues/159)) ([1cfb936](https://github.com/telemetry-dev/telemetry/commit/1cfb936b24d8b38afe62b9699679bc600cb01069))
* **sdk-python:** run a flush or shutdown from on_error after the export ([#152](https://github.com/telemetry-dev/telemetry/issues/152)) ([19ee0b1](https://github.com/telemetry-dev/telemetry/commit/19ee0b1aa36973b30db652d8e98396f358a8611b))
* **sdk-python:** wait for prompt fetches on per-call locks so forks cannot hang ([#167](https://github.com/telemetry-dev/telemetry/issues/167)) ([9ec46de](https://github.com/telemetry-dev/telemetry/commit/9ec46ded237d982d6d5514180f4632faab979901)), closes [#158](https://github.com/telemetry-dev/telemetry/issues/158)
* **sdk:** find the comment of the action on the old pages of a long pull request ([#157](https://github.com/telemetry-dev/telemetry/issues/157)) ([4791ba4](https://github.com/telemetry-dev/telemetry/commit/4791ba4e0779e8dcf826a43718e29a5937586095))
* **sdk:** keep every prompt variable when their JSON is too long ([#142](https://github.com/telemetry-dev/telemetry/issues/142)) ([ca0b86c](https://github.com/telemetry-dev/telemetry/commit/ca0b86c8592807071c7c97942a2b28f7e0a6ae8e))
* **sdk:** name failed runs in the experiment result and suggest spanId for feedback ([#163](https://github.com/telemetry-dev/telemetry/issues/163)) ([3046a72](https://github.com/telemetry-dev/telemetry/commit/3046a72db1a27461388f9c24e1ad269897c5cada))
* **sdk:** send each captured value whole ([#148](https://github.com/telemetry-dev/telemetry/issues/148)) ([b9e689a](https://github.com/telemetry-dev/telemetry/commit/b9e689a438aecc72436495d4d5d75b9ac847f90e))


### Documentation

* **sdk:** correct the READMEs that no longer match the SDKs ([#164](https://github.com/telemetry-dev/telemetry/issues/164)) ([acb45f5](https://github.com/telemetry-dev/telemetry/commit/acb45f501bf60dd5ecefaff6a8f5769bf0cd51fa))

## [0.3.2](https://github.com/telemetry-dev/telemetry/compare/python-v0.3.1...python-v0.3.2) (2026-10-30)


### Features

* **sdk:** record the prompt of compiled messages outside withPrompt ([6435fb6](https://github.com/telemetry-dev/telemetry/commit/6435fb6d2d7488c968a6ce9f9c6437f2cf426456))
* **sdk:** record the prompt on model calls outside withPrompt ([339eb0c](https://github.com/telemetry-dev/telemetry/commit/339eb0c01c5754191eb7d1b5ec37db21ebd28855)), closes [#97](https://github.com/telemetry-dev/telemetry/issues/97)
* **sdk:** run experiments on datasets and send scores ([fe00efa](https://github.com/telemetry-dev/telemetry/commit/fe00efa1326797f8fda647d665b388621a961d39))


### Bug Fixes

* **sdk-python:** fetch prompts again in a forked process ([#105](https://github.com/telemetry-dev/telemetry/issues/105)) ([c6b599e](https://github.com/telemetry-dev/telemetry/commit/c6b599ed1d0311ff2c367c0e64e47bd7c2b3b402))
* **sdk:** find the prompt by message identity, not by text ([ae47605](https://github.com/telemetry-dev/telemetry/commit/ae47605555ef9f5dfed1d6f762ac56823e692edb))
* **sdk:** give the base url to getPrompt in the prompt tests ([abb8d10](https://github.com/telemetry-dev/telemetry/commit/abb8d10f680d1fa7ace6913a1d53b43a3e0db703)), closes [#97](https://github.com/telemetry-dev/telemetry/issues/97)
* **sdk:** keep compiled variables for 5 minutes, not two turns ([1b09916](https://github.com/telemetry-dev/telemetry/commit/1b099167bbd8acaf5008848ad9ba2a2ab8e35b0d)), closes [#97](https://github.com/telemetry-dev/telemetry/issues/97)
* **sdk:** keep the variables of recent compiles only ([3bc6f53](https://github.com/telemetry-dev/telemetry/commit/3bc6f53a0afb0ad8b9fe19f8eaf7e5d812a93b0a)), closes [#97](https://github.com/telemetry-dev/telemetry/issues/97)
* **sdk:** keep the variables of the last two chat turns ([f6aa1c1](https://github.com/telemetry-dev/telemetry/commit/f6aa1c159742123600d6499653916c4dbf7b2ab7)), closes [#97](https://github.com/telemetry-dev/telemetry/issues/97)
* **sdk:** name the newest compile, not the last tagged message ([a6b98fe](https://github.com/telemetry-dev/telemetry/commit/a6b98fe916d5ed4b37bddd7972b07507182f2d2d)), closes [#97](https://github.com/telemetry-dev/telemetry/issues/97)


### Documentation

* **sdk:** name the read prompts access of the prompt api key ([#103](https://github.com/telemetry-dev/telemetry/issues/103)) ([1195b11](https://github.com/telemetry-dev/telemetry/commit/1195b11581dc729e346e888cf8406ab3dd917d96))
* **sdk:** tell that a call with no compiled message records its input ([0c2f2b6](https://github.com/telemetry-dev/telemetry/commit/0c2f2b6e17a534d403179dd277db9136ab17f815)), closes [#97](https://github.com/telemetry-dev/telemetry/issues/97)

## [0.3.1](https://github.com/telemetry-dev/telemetry/compare/python-v0.3.0...python-v0.3.1) (2026-10-30)


### Bug Fixes

* **sdk:** bound prompt fetches and time them by elapsed time ([#117](https://github.com/telemetry-dev/telemetry/issues/117)) ([0741322](https://github.com/telemetry-dev/telemetry/commit/0741322071d2648cbb771750763847e3483c8fa0))

## [0.3.0](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.13...python-v0.3.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **sdk:** the default base URL for telemetry export and getPrompt/get_prompt is now https://api.telemetry.dev. https://ingest.telemetry.dev reaches the same service and stays supported; networks that allow only it should set TELEMETRY_DEV_BASE_URL to it.

### Features

* **sdk:** send to api.telemetry.dev by default ([#109](https://github.com/telemetry-dev/telemetry/issues/109)) ([ad8ccf3](https://github.com/telemetry-dev/telemetry/commit/ad8ccf3c41355683deb3b38dab4b604ecaa25a7a))


### Bug Fixes

* **sdk-python:** send a user agent with prompt fetches ([#108](https://github.com/telemetry-dev/telemetry/issues/108)) ([cbd7173](https://github.com/telemetry-dev/telemetry/commit/cbd7173db89f315ac48c8013df1d5f0a8059569c))

## [0.2.13](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.12...python-v0.2.13) (2026-10-29)


### Bug Fixes

* **sdk:** honor retry-after for prompts and keep python task variables ([#102](https://github.com/telemetry-dev/telemetry/issues/102)) ([6ad4a0c](https://github.com/telemetry-dev/telemetry/commit/6ad4a0c1714d8849f9a5561016bd84f3959d534c))

## [0.2.12](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.11...python-v0.2.12) (2026-10-29)


### Features

* **sdk:** add getPrompt and withPrompt in TypeScript and Python ([#99](https://github.com/telemetry-dev/telemetry/issues/99)) ([b1a33e4](https://github.com/telemetry-dev/telemetry/commit/b1a33e446864f02c2fd050e754c81c875c7f9864))

## [0.2.11](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.10...python-v0.2.11) (2026-10-27)


### Bug Fixes

* **sdk-python:** keep the end of truncated tracebacks ([#89](https://github.com/telemetry-dev/telemetry/issues/89)) ([2404ae4](https://github.com/telemetry-dev/telemetry/commit/2404ae49677a8de4c5c64ea99866167c5431b1e1))
* **sdk-python:** report a series of failed exports once ([#87](https://github.com/telemetry-dev/telemetry/issues/87)) ([06b75e9](https://github.com/telemetry-dev/telemetry/commit/06b75e905c640c2ca28b16ed16724f1457090eaf))
* **sdk:** align captured JSON and provider usage ([#79](https://github.com/telemetry-dev/telemetry/issues/79)) ([1ec98d5](https://github.com/telemetry-dev/telemetry/commit/1ec98d5014ac810116402736cc7db420745eb9ef))
* **sdk:** keep exception text behind capture flags ([#88](https://github.com/telemetry-dev/telemetry/issues/88)) ([e9e49c3](https://github.com/telemetry-dev/telemetry/commit/e9e49c33dfcff28c74f7a6e16126054dc32b0d90))

## [0.2.10](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.9...python-v0.2.10) (2026-10-19)


### Features

* **sdk:** run the configured mask on integration content and in the ai package ([#73](https://github.com/telemetry-dev/telemetry/issues/73)) ([365b6e7](https://github.com/telemetry-dev/telemetry/commit/365b6e76ebbccd3bb63803d0abb1757b7d16573d))


### Documentation

* document native claude sdk spans ([#61](https://github.com/telemetry-dev/telemetry/issues/61)) ([e280bcc](https://github.com/telemetry-dev/telemetry/commit/e280bccab3960f1e8388d6cddb5361b0f947e081))
* link each package readme to its docs and fix stale setup ([#76](https://github.com/telemetry-dev/telemetry/issues/76)) ([dde0b0c](https://github.com/telemetry-dev/telemetry/commit/dde0b0cb0639b1e2dcde31a8a3098ad349f71d23))

## [0.2.9](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.8...python-v0.2.9) (2026-10-14)


### Bug Fixes

* **sdk-python:** charge json escapes when CaptureBudget measures strings ([#48](https://github.com/telemetry-dev/telemetry/issues/48)) ([3d5daf5](https://github.com/telemetry-dev/telemetry/commit/3d5daf578bbcdfe4113bea62b55fceb88dd10418))

## [0.2.8](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.7...python-v0.2.8) (2026-10-01)


### Bug Fixes

* **sdk:** flag incomplete stream capture and keep it from masks ([#33](https://github.com/telemetry-dev/telemetry/issues/33)) ([9454284](https://github.com/telemetry-dev/telemetry/commit/9454284555a2b704c2616620311d36d9db7a9d82))

## [0.2.7](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.6...python-v0.2.7) (2026-09-30)


### Bug Fixes

* require OpenTelemetry 1.39 or newer and drop SDK-internal metric-reader telemetry ([#22](https://github.com/telemetry-dev/telemetry/issues/22)) ([cee7079](https://github.com/telemetry-dev/telemetry/commit/cee707948779667c3fc5f3709f752b7ee3d4ec9b))

## [0.2.6](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.5...python-v0.2.6) (2026-09-27)


### Features

* **sdk:** trace media, realtime, rerank, and modality token usage ([#17](https://github.com/telemetry-dev/telemetry/issues/17)) ([c489503](https://github.com/telemetry-dev/telemetry/commit/c4895031a27a30004e21dc58f678d0f1ce19c072))

## [0.2.5](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.4...python-v0.2.5) (2026-09-12)


### Bug Fixes

* correct package repository metadata ([#7](https://github.com/telemetry-dev/telemetry/issues/7)) ([23f74ea](https://github.com/telemetry-dev/telemetry/commit/23f74ea1480d02f4d3e34380dfb61ce149f12c86))

## [0.2.4](https://github.com/telemetry-dev/telemetry/compare/python-v0.2.3...python-v0.2.4) (2026-09-12)


### Features

* **sdk:** add streaming latency metrics ([#5](https://github.com/telemetry-dev/telemetry/issues/5)) ([f1a7261](https://github.com/telemetry-dev/telemetry/commit/f1a72617ed1f802d68ee71baf767cec934827f98))

## [0.2.3](https://github.com/telemetry-dev/sdks/compare/python-v0.2.2...python-v0.2.3) (2026-09-07)


### Bug Fixes

* include shared instrumentation and capture refactors in published SDKs ([7c82386](https://github.com/telemetry-dev/sdks/commit/7c82386e139303a49f4cb3b6a39affac85763e88))

## [0.2.2](https://github.com/telemetry-dev/sdks/compare/python-v0.2.1...python-v0.2.2) (2026-09-05)


### Features

* one trace per session, with turn-by-turn session replay ([#226](https://github.com/telemetry-dev/sdks/issues/226)) ([44e2e75](https://github.com/telemetry-dev/sdks/commit/44e2e7510bc5073df89a0135cac9e986aeade5c6))

## [0.2.1](https://github.com/telemetry-dev/sdks/compare/python-v0.2.0...python-v0.2.1) (2026-09-02)


### Features

* **sdk:** add telemetry.dev request header ([#181](https://github.com/telemetry-dev/sdks/issues/181)) ([23d5141](https://github.com/telemetry-dev/sdks/commit/23d5141764b9e7b1640f7ea6f836686146a24822))
