# Changelog

## [0.4.2](https://github.com/CAIIVS/raitap/compare/v0.4.1...v0.4.2) (2026-05-06)


### Bug fixes

* **deps:** pin triton-xpu&gt;=3.0.0 to skip yanked 0.0.2 ([#111](https://github.com/CAIIVS/raitap/issues/111)) ([42f1932](https://github.com/CAIIVS/raitap/commit/42f193204210a0cba334551d4ace7f104c5d4119))

## [0.4.1](https://github.com/CAIIVS/raitap/compare/v0.4.0...v0.4.1) (2026-05-06)


### Bug fixes

* **data:** nested ImageFolder layouts blocked input_metadata inference ([#107](https://github.com/CAIIVS/raitap/issues/107)) ([87fa454](https://github.com/CAIIVS/raitap/commit/87fa4542ab8e588aaa89541e136b6381dde9f900))
* **data:** suggest id_strategy in label-alignment error messages ([#106](https://github.com/CAIIVS/raitap/issues/106)) ([da2875e](https://github.com/CAIIVS/raitap/commit/da2875eca65c53dc8be8b15d2c97f8d1877fd38c))


### Documentation

* **transparency:** surface required raitap.input_metadata config ([#105](https://github.com/CAIIVS/raitap/issues/105)) ([a2ae60f](https://github.com/CAIIVS/raitap/commit/a2ae60f1f913d7d566b7604d88a48aa7f96f4b21))

## [0.4.0](https://github.com/CAIIVS/raitap/compare/v0.3.0...v0.4.0) (2026-05-06)


### ⚠ BREAKING CHANGES

* **model:** state_dict + TorchScript loaders; deprecate pickled nn.Module ([#104](https://github.com/CAIIVS/raitap/issues/104))
* **data:** path-based labels.csv resolution + recursive image walk ([#103](https://github.com/CAIIVS/raitap/issues/103))

### Features

* **data:** path-based labels.csv resolution + recursive image walk ([#103](https://github.com/CAIIVS/raitap/issues/103)) ([8bc55e2](https://github.com/CAIIVS/raitap/commit/8bc55e2d91f7c4a1965820de93d4de343b87c252))
* **model:** state_dict + TorchScript loaders; deprecate pickled nn.Module ([#104](https://github.com/CAIIVS/raitap/issues/104)) ([6511b98](https://github.com/CAIIVS/raitap/commit/6511b982b898e37e0a72867004a14b5c515922be))


### Documentation

* badge link in README ([4264a58](https://github.com/CAIIVS/raitap/commit/4264a5892c6479f2196968eb4a2d4e93d0e4d692))

## [0.3.0](https://github.com/CAIIVS/raitap/compare/v0.2.1...v0.3.0) (2026-05-05)


### ⚠ BREAKING CHANGES

* **transparency:** overhaul transparency to support many explainer categories, scopes and families in a typed robust way ([#98](https://github.com/CAIIVS/raitap/issues/98))
* remove alibi support for the moment, due to deps conflicts
* **reporting:** add compact global/local reports and multirun merging ([#89](https://github.com/CAIIVS/raitap/issues/89))
* **transparency:** split RAITAP and library kwargs in transparency config ([#86](https://github.com/CAIIVS/raitap/issues/86))
* harden python version support and document legacy CUDA compatibility guidance ([#85](https://github.com/CAIIVS/raitap/issues/85))

### Features

* **reporting:** add compact global/local reports and multirun merging ([#89](https://github.com/CAIIVS/raitap/issues/89)) ([e691e70](https://github.com/CAIIVS/raitap/commit/e691e707a24f1af0e29f3c6cac31b1be29a76750))


### Bug fixes

* harden python version support and document legacy CUDA compatibility guidance ([#85](https://github.com/CAIIVS/raitap/issues/85)) ([5f0cad6](https://github.com/CAIIVS/raitap/commit/5f0cad65b984069a1f094988826ef6fdd8afd3a6))
* remove alibi support for the moment, due to deps conflicts ([4a9d6d4](https://github.com/CAIIVS/raitap/commit/4a9d6d4214c837bf7f3aed620921ae20edf3ca5c))


### Dependencies

* **actions:** bump dorny/paths-filter from 3 to 4 ([#84](https://github.com/CAIIVS/raitap/issues/84)) ([b215d4a](https://github.com/CAIIVS/raitap/commit/b215d4a64234d72e6b8b1dfb7abea422171d7c6c))
* **actions:** bump googleapis/release-please-action from 4 to 5 ([#91](https://github.com/CAIIVS/raitap/issues/91)) ([106bc89](https://github.com/CAIIVS/raitap/commit/106bc89ead1438e8b08c8ab43b89822db22608bf))


### Documentation

* badges ([098f0b5](https://github.com/CAIIVS/raitap/commit/098f0b5d0e66b7c7af88b3f597d0d07947cfa18c))
* delete mkdocs.yaml ([21087d1](https://github.com/CAIIVS/raitap/commit/21087d1249f6f70944db9781ec2acf093ada7492))


### Internal code refactoring

* improve type-safety, remove dead code, remove memory and data leaks ([#80](https://github.com/CAIIVS/raitap/issues/80)) ([d50a79c](https://github.com/CAIIVS/raitap/commit/d50a79c4f7ac5e3c5601428113b84b5aa613e2b7))
* **transparency:** overhaul transparency to support many explainer categories, scopes and families in a typed robust way ([#98](https://github.com/CAIIVS/raitap/issues/98)) ([f0d5718](https://github.com/CAIIVS/raitap/commit/f0d5718da13adf45f5063082c4044a44054b5a77))
* **transparency:** split RAITAP and library kwargs in transparency config ([#86](https://github.com/CAIIVS/raitap/issues/86)) ([663f42e](https://github.com/CAIIVS/raitap/commit/663f42e196647fb2d5400f62288948258f270299))

## [0.2.1](https://github.com/CAIIVS/raitap/compare/v0.2.0...v0.2.1) (2026-04-14)


### Dependencies

* **dev:** update commitizen requirement from &gt;=3.0.0 to &gt;=4.13.9 ([#47](https://github.com/CAIIVS/raitap/issues/47)) ([846633b](https://github.com/CAIIVS/raitap/commit/846633b42fdfde949e1af600a5eebc01908dae0b))
* **dev:** update furo requirement from &gt;=2024.8.6 to &gt;=2025.12.19 ([#42](https://github.com/CAIIVS/raitap/issues/42)) ([b89e8ab](https://github.com/CAIIVS/raitap/commit/b89e8ab41697b30cdc1379151d6c245c18019a45))
* **dev:** update myst-parser requirement from &gt;=4.0.0 to &gt;=5.0.0 ([#45](https://github.com/CAIIVS/raitap/issues/45)) ([4c726d0](https://github.com/CAIIVS/raitap/commit/4c726d0d17fa4cb6ef6b80db0247859c89508750))
* **dev:** update pytest requirement from &gt;=8.0.0 to &gt;=9.0.3 ([#49](https://github.com/CAIIVS/raitap/issues/49)) ([22745d4](https://github.com/CAIIVS/raitap/commit/22745d41435a0510476eab7fde3d9048735ba416))
* **dev:** update pytest-cov requirement from &gt;=4.1.0 to &gt;=7.1.0 ([#41](https://github.com/CAIIVS/raitap/issues/41)) ([29ef0ee](https://github.com/CAIIVS/raitap/commit/29ef0ee5d65c6f771b26f608190b0e68d2fdf825))
* **dev:** update sphinx requirement from &gt;=8.0.0 to &gt;=9.1.0 ([#46](https://github.com/CAIIVS/raitap/issues/46)) ([6ecbc20](https://github.com/CAIIVS/raitap/commit/6ecbc20a7156fe6dcc21c8c4012727c0fc4b8d07))
* **dev:** update sphinx-autobuild requirement from &gt;=2024.10.3 to &gt;=2025.8.25 ([#48](https://github.com/CAIIVS/raitap/issues/48)) ([c313c3f](https://github.com/CAIIVS/raitap/commit/c313c3ff2131885351f88889b264ecd28c3b2359))
* update pillow requirement from &gt;=10.0.0 to &gt;=12.2.0 ([#43](https://github.com/CAIIVS/raitap/issues/43)) ([34b79ad](https://github.com/CAIIVS/raitap/commit/34b79adfaa5ab909bf5e64b0c36bef9794a6e1e0))


### Code Refactoring

* **transparency:** truly lib agnostic and added alibi ([#69](https://github.com/CAIIVS/raitap/issues/69)) ([1447c61](https://github.com/CAIIVS/raitap/commit/1447c61db83f0d8db17a73b051a7b01dc9d29fc6))

## [0.2.0](https://github.com/CAIIVS/raitap/compare/v0.1.0...v0.2.0) (2026-04-13)


### ⚠ BREAKING CHANGES

* issues discovered during cluster tests ([#51](https://github.com/CAIIVS/raitap/issues/51))

### Bug Fixes

* issues discovered during cluster tests ([#51](https://github.com/CAIIVS/raitap/issues/51)) ([64e964c](https://github.com/CAIIVS/raitap/commit/64e964c88f5e44ad2b092781d2e607c15664e8c6))


### Documentation

* fix order ([13e5ae3](https://github.com/CAIIVS/raitap/commit/13e5ae398ee54b8d22c07a25f998415dc6c9cc99))

## [0.1.0](https://github.com/CAIIVS/raitap/compare/v0.0.1...v0.1.0) (2026-04-12)


### Features

* pypi publishing ([#64](https://github.com/CAIIVS/raitap/issues/64)) ([a1cadb0](https://github.com/CAIIVS/raitap/commit/a1cadb06383951d2bc99ace4067d1bd6a29406f0))
