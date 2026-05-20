# Changelog

## [0.9.2](https://github.com/CAIIVS/raitap/compare/v0.9.1...v0.9.2) (2026-05-20)


### Documentation

* consolidate CLI/python/env knobs into Flags reference table ([#189](https://github.com/CAIIVS/raitap/issues/189)) ([1477f17](https://github.com/CAIIVS/raitap/commit/1477f17f49595bdc9f025b1836106d9ade40dc47))
* document RAITAP_* environment variables ([#187](https://github.com/CAIIVS/raitap/issues/187)) ([aee3895](https://github.com/CAIIVS/raitap/commit/aee3895a83553f446034d6fd189da77a6803d36f))

## [0.9.1](https://github.com/CAIIVS/raitap/compare/v0.9.0...v0.9.1) (2026-05-19)


### Bug fixes

* **transparency:** auto-upsample low-res attributions for blended_heat_map/heat_map ([#184](https://github.com/CAIIVS/raitap/issues/184)) ([a6d876e](https://github.com/CAIIVS/raitap/commit/a6d876e94fe8ee22e9d8d7be8295d4673c5379b0))

## [0.9.0](https://github.com/CAIIVS/raitap/compare/v0.8.0...v0.9.0) (2026-05-19)


### ⚠ BREAKING CHANGES

* **transparency:** detection task-family + Phase 5 contributor config + E2E ([#176](https://github.com/CAIIVS/raitap/issues/176))

### Features

* **transparency:** detection task-family + Phase 5 contributor config + E2E ([#176](https://github.com/CAIIVS/raitap/issues/176)) ([55d6fdb](https://github.com/CAIIVS/raitap/commit/55d6fdbf8771e97fffd7628b588e4ca1659e6a59))

## [0.8.0](https://github.com/CAIIVS/raitap/compare/v0.7.1...v0.8.0) (2026-05-18)


### ⚠ BREAKING CHANGES

* **data:** configurable image preprocessing and fix bootstrapping ([#172](https://github.com/CAIIVS/raitap/issues/172))
* **metrics:** typed schemas, per-task classification adapters, nested iou config ([#179](https://github.com/CAIIVS/raitap/issues/179))

### Features

* **metrics:** typed schemas, per-task classification adapters, nested iou config ([#179](https://github.com/CAIIVS/raitap/issues/179)) ([0871ad3](https://github.com/CAIIVS/raitap/commit/0871ad3c048596c7412dbcf19c9f58ca8d3ca4ce))


### Bug fixes

* **data:** configurable image preprocessing and fix bootstrapping ([#172](https://github.com/CAIIVS/raitap/issues/172)) ([4cc36b8](https://github.com/CAIIVS/raitap/commit/4cc36b87348a8de9347119ca0138ee61b1a0420c))
* validate sample_names length ([#97](https://github.com/CAIIVS/raitap/issues/97)) ([#177](https://github.com/CAIIVS/raitap/issues/177)) ([9a4afc4](https://github.com/CAIIVS/raitap/commit/9a4afc48d5d3b9dd3e726ef139d1b06b786552b3))

## [0.7.1](https://github.com/CAIIVS/raitap/compare/v0.7.0...v0.7.1) (2026-05-17)


### Features

* programmatic Python API + hydra-zen + typed schemas + docs tabs ([#168](https://github.com/CAIIVS/raitap/issues/168)) ([14a816a](https://github.com/CAIIVS/raitap/commit/14a816abf5b28b001180f12653c124d42fc22438))


### Dependencies

* **actions:** bump actions/download-artifact from 6 to 8 ([#169](https://github.com/CAIIVS/raitap/issues/169)) ([c593f54](https://github.com/CAIIVS/raitap/commit/c593f54e718d59d706edfb54b18c2761fefc30de))


### Internal code refactoring

* annotation system + python api fix + llm docs ([#174](https://github.com/CAIIVS/raitap/issues/174)) ([6b2ae75](https://github.com/CAIIVS/raitap/commit/6b2ae75c9aad9d68f4dd6987a8da9834bc7dc66c))

## [0.7.0](https://github.com/CAIIVS/raitap/compare/v0.6.2...v0.7.0) (2026-05-14)


### ⚠ BREAKING CHANGES

* **config:** prune presets, add --demo flag ([#165](https://github.com/CAIIVS/raitap/issues/165))

### Internal code refactoring

* **config:** prune presets, add --demo flag ([#165](https://github.com/CAIIVS/raitap/issues/165)) ([b1107cc](https://github.com/CAIIVS/raitap/commit/b1107cc5dc322bf1d5eb60946fc39d0be27c68c1))
* rename Base/Abstract classes and registry kwarg ([#167](https://github.com/CAIIVS/raitap/issues/167)) ([b4de997](https://github.com/CAIIVS/raitap/commit/b4de9976a9f8a32e613cfa2eda8e38c2f8f347bc))

## [0.6.2](https://github.com/CAIIVS/raitap/compare/v0.6.1...v0.6.2) (2026-05-13)


### Bug fixes

* **docs:** light-mode brand color, drop logo margin; rename reporting extras ([#162](https://github.com/CAIIVS/raitap/issues/162)) ([d7270db](https://github.com/CAIIVS/raitap/commit/d7270db1a2b1cc17142282e4d47f2751f4f08de6))

## [0.6.1](https://github.com/CAIIVS/raitap/compare/v0.6.0...v0.6.1) (2026-05-13)


### Features

* **docs:** brand color, landing logo, sidebar alignment ([#159](https://github.com/CAIIVS/raitap/issues/159)) ([#160](https://github.com/CAIIVS/raitap/issues/160)) ([7f2d014](https://github.com/CAIIVS/raitap/commit/7f2d0142d5da4f2aa8581999fe8acde924851eaf))

## [0.6.0](https://github.com/CAIIVS/raitap/compare/v0.5.0...v0.6.0) (2026-05-13)


### ⚠ BREAKING CHANGES

* **model:** refuse unsafe pickle checkpoints unless opted in ([#157](https://github.com/CAIIVS/raitap/issues/157))
* **model:** adapt model input shape in backends ([#152](https://github.com/CAIIVS/raitap/issues/152))
* **reporting:** redesign report and implement html reporter ([#139](https://github.com/CAIIVS/raitap/issues/139))

### Features

* **infra:** chips for every panel-producing log ([#149](https://github.com/CAIIVS/raitap/issues/149)) ([#150](https://github.com/CAIIVS/raitap/issues/150)) ([d75b7a9](https://github.com/CAIIVS/raitap/commit/d75b7a9e9b94204f99b0ab3ef45974e7796509fc))
* **infra:** error-rethrow layer for wrapped third-party libs ([#140](https://github.com/CAIIVS/raitap/issues/140)) ([1bdd2b6](https://github.com/CAIIVS/raitap/commit/1bdd2b666936878265b44d0fb2810c8807a0a9c3))
* **infra:** raitap-deps — auto-detect uv extras from Hydra config ([#100](https://github.com/CAIIVS/raitap/issues/100)) ([#155](https://github.com/CAIIVS/raitap/issues/155)) ([6d4cb71](https://github.com/CAIIVS/raitap/commit/6d4cb716d5b1c9aa3c54fc82980bdcabe3a12824))
* **model:** adapt model input shape in backends ([#152](https://github.com/CAIIVS/raitap/issues/152)) ([d04bab7](https://github.com/CAIIVS/raitap/commit/d04bab70d1599f5286ff6bf5d06f592befe83f27))
* **reporting:** redesign report and implement html reporter ([#139](https://github.com/CAIIVS/raitap/issues/139)) ([f80df50](https://github.com/CAIIVS/raitap/commit/f80df501ac0ffcc9dc3c8351cb005cd89c27e16a))
* **robustness:** per-logit output bounds for MarabouAssessor ([#142](https://github.com/CAIIVS/raitap/issues/142)) ([aa11125](https://github.com/CAIIVS/raitap/commit/aa11125a5898478e118f6d04d4445a908fa3197f))
* **robustness:** visualise MarabouAssessor per-logit output bounds ([#145](https://github.com/CAIIVS/raitap/issues/145)) ([5dc9ec1](https://github.com/CAIIVS/raitap/commit/5dc9ec1d9cbe0a89bfbf561a0e4aed20579025ea))


### Bug fixes

* **model:** refuse unsafe pickle checkpoints unless opted in ([#157](https://github.com/CAIIVS/raitap/issues/157)) ([1f0cb66](https://github.com/CAIIVS/raitap/commit/1f0cb6696637fa83593eedcee5e388df712b9f6a))

## [0.5.0](https://github.com/CAIIVS/raitap/compare/v0.4.2...v0.5.0) (2026-05-10)


### ⚠ BREAKING CHANGES

* **tracking:** switch mlflow from deprecated mlruns to sqllite db ([#122](https://github.com/CAIIVS/raitap/issues/122))
* **deps:** lower Python floor to 3.11 ([#133](https://github.com/CAIIVS/raitap/issues/133))
* **data:** unify data.source and data.labels.source resolvers ([#128](https://github.com/CAIIVS/raitap/issues/128))

### Features

* **data:** unify data.source and data.labels.source resolvers ([#128](https://github.com/CAIIVS/raitap/issues/128)) ([7fccefe](https://github.com/CAIIVS/raitap/commit/7fccefea8a96da3e4236c7f6fa728a21a776d362))
* **infra:** diagnostics utils, better terminal warnings ([#124](https://github.com/CAIIVS/raitap/issues/124)) ([11ef2f5](https://github.com/CAIIVS/raitap/commit/11ef2f5f31f1016986d860e1bf5d3d69cbf6b2ae))
* **misc:** prettify terminal output with rich panels and themed progress ([#121](https://github.com/CAIIVS/raitap/issues/121)) ([6a9fb61](https://github.com/CAIIVS/raitap/commit/6a9fb61551337da384569670f436990b1e935cc5))
* **reporting:** allow users to pin report samples for explainer visualisations ([#118](https://github.com/CAIIVS/raitap/issues/118)) ([58deebe](https://github.com/CAIIVS/raitap/commit/58deebee1c2ed205b0cc4e4fb3c05a3dc2f47a44))
* **robustness:** Marabou formal-verification adapter + SemanticallyDescribable refactor ([#134](https://github.com/CAIIVS/raitap/issues/134)) ([99968f2](https://github.com/CAIIVS/raitap/commit/99968f2ac1e5ca15c609526504a813c39044a145))
* **robustness:** rebuild module with typed contracts and attack adapters ([#119](https://github.com/CAIIVS/raitap/issues/119)) ([ed3f845](https://github.com/CAIIVS/raitap/commit/ed3f84582448d6688c5c50a021a9bc89f19b03d8))


### Bug fixes

* **tracking:** switch mlflow from deprecated mlruns to sqllite db ([#122](https://github.com/CAIIVS/raitap/issues/122)) ([73b1a64](https://github.com/CAIIVS/raitap/commit/73b1a64e7ecac19c18aa62fecb1f961ae523dca3))


### Documentation

* add Context7 widget script to Sphinx docs ([deea214](https://github.com/CAIIVS/raitap/commit/deea214a406fa4c11a6151a485186c6135a3164c))
* context7.json with URL and public key ([73c30c3](https://github.com/CAIIVS/raitap/commit/73c30c35b0338d9e0ba8fad00ad7ffa9926bc496))
* update README with logo and project details ([a4da263](https://github.com/CAIIVS/raitap/commit/a4da263eb4c0bbef1bb64a152012ad11a6f46986))


### Build system

* **deps:** lower Python floor to 3.11 ([#133](https://github.com/CAIIVS/raitap/issues/133)) ([da5ad39](https://github.com/CAIIVS/raitap/commit/da5ad3914cf3ed623acf08ba8692aab7cee101f9))

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
