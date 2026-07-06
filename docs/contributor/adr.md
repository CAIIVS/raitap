---
title: "Architecture decisions"
description: "A log of RAITAP's durable design decisions: the fork each faced, the option chosen, and what it costs or unlocks."
myst:
  html_meta:
    "description": "A log of RAITAP's durable design decisions: the fork each faced, the option chosen, and what it costs or unlocks."
---

# Architecture decisions

The *why* behind the layout in {doc}`architecture`. Each record has the same four fields, in the same order:

- **Context** the forces in play.
- **Options** what we weighed.
- **Decision** what we chose.
- **Consequences** what it costs or unlocks, plus any condition that would reopen it.

## Adapters self-register via facade decorators

**Context:** Every family (explainer, assessor, metric, reporter, tracker, visualiser, backend) needs its concrete types discoverable by Hydra config, and contributors must add one without touching core.

**Options:** A central registry table edited per adapter; a base class the pipeline scans by subclass; a decorator that self-registers.

**Decision:** Decorator. `@adapters.<family>` / `@visualisers.<family>` / `@backends.register` in `<module>/registration.py` delegate to `_adapters._register_core`, which builds the hydra-zen builder, registers it with the `ConfigStore`, and populates `_BUILDERS` / `ADAPTER_EXTRAS` / `THIRD_PARTY_LIBS`.

**Consequences:** Adding an adapter is one decorated class, no core edit. `AdapterDecoratorOptions` (TypedDict + `Unpack`) gives call-site type checking. Resolution is lazy via each family `__init__`'s `__getattr__`, which is what keeps the deps layer torch-free.

## Deps layer stays torch-free

**Context:** `pip install raitap` then `raitap --demo` must work before any backend is installed, since the deps layer's job is to infer and sync the right extras.

**Options:** Import backends eagerly and require a heavy install up front; keep the pre-pipeline path free of backend imports.

**Decision:** Torch-free `deps/`. Every adapter family `__init__` stays free of top-level backend-lib imports (`utils/lazy`), so dep inference runs on a bare install.

**Consequences:** Bootstrap-from-zero works. The cost is a standing constraint: a top-level `import torch` anywhere in a family `__init__` silently breaks it. Guarded by `deps/tests/test_bootstrap_from_zero.py`.

## Compatibility gates stay separate

**Context:** RAITAP has three "declared-property vs accepted-set" gates: backend (`required ⊆ provided`, subset, flat `Capability` enum), visualiser-semantic (`candidate ∩ supported ≠ ∅` over typed `ExplanationSemantics` axes), and adapter `supported_tasks` (declared, no runtime raising gate).

**Options:** Unify behind a shared `CompatibilityGate` protocol / predicate helper; keep them separate, sharing only a typed error.

**Decision:** Keep separate. Revisited three times (from the error-type, predicate-helper, and interface angles) and held each time.

**Consequences:** Three different set predicates and two different type systems stay legible; a shared `check(declared, accepted)` would hide which predicate runs and erase the visualiser typing. Reopens only when the plugin system lets third parties register their own gates, at which point a `CompatibilityGate` protocol is justified as an extension seam, not internal DRY.

## Error types only at boundaries

**Context:** The status frame (`print_failure_panel` / `RaitapRichHandler`) already renders any raitap-raised error, typed or not, with scope and `file:line`.

**Options:** Subclass `RaitapError` broadly for nice rendering; raise raw `ValueError`/`TypeError` and type only where needed.

**Decision:** Type only at a boundary: something `except`s it for control flow, a test asserts that specific failure, or it crosses the public facade.

**Consequences:** "Make it render nicely" is not a reason to subclass. The repo runs on ~190 raw raises, 3 `RaitapError` subclasses, and 1 production catch (`AssessorBackendIncompatibilityError`). Type lazily when a boundary appears, never as a sweep. See {doc}`logging`.

## Non-uniform algorithms use the invoker seam

**Context:** Most explainers and assessors share one construct-and-call lifecycle, but some (SHAP legacy vs modern, foolbox `DatasetAttack`) need a different call shape.

**Options:** Subclass the adapter per odd algorithm; branch inside the adapter on algorithm name; carry a per-algorithm invoker.

**Decision:** Per-algorithm `invoker` (#266). The registry record (`ExplainerAlgorithmSpec` / `AssessorAlgorithmSpec`) carries an optional `invoker`; the chokepoint runs `(spec.invoker or self._default_invoke)(ctx)`, where `ctx` is a frozen per-family dataclass carrying the adapter instance.

**Consequences:** A non-uniform algorithm is a module-level function plus `invoker=` on its entry, reusing shared helpers through `ctx`. No adapter-base changes. Per-algorithm hints (norm/threat/stochastic) must be verified against the installed library, not assumed.

## Tree backends stamp OutputKind at the forward site

**Context:** Tree backends emit calibrated probabilities; torch/ONNX backends emit logits. Downstream classification must know which to skip softmax correctly.

**Options:** Read `PREDICT_PROBA in provides` deep inside the task family where it is needed; stamp the fact once at the forward pass.

**Decision:** Stamp `OutputKind` (`LOGITS` / `PROBABILITIES`) on `ForwardOutput` at the forward-pass site, which reads `provides` there.

**Consequences:** Task families read a plain field, not backend internals. General rule: stamp a derived fact at its source site, do not re-derive it deep in consumers.

## Label-format adapters raise plain ValueError

**Context:** Ingesting COCO/YOLO/VOC labels (#338) needs an error for unsupported (format, task) pairs. The typed compatibility error already exists.

**Options:** Reuse the typed compat error; raise plain `ValueError`.

**Decision:** Plain `ValueError`. An unsupported label format is not a capability mismatch, so the compat error is semantically wrong here.

**Consequences:** Consistent with the boundary rule above. COCO/YOLO category ids also pass through unchanged (no dense remap) to preserve the model's label space.

## Text tokenisation is model-owned

**Context:** Text input (#340) needs token ids plus an attention mask threaded to both metrics and the explainer. Where the tokeniser lives sets whether the design overfits to classification.

**Options:** A data-layer `adapt_loaded_inputs` tokenisation seam; the model owns tokenisation via its Hugging Face tokenizer.

**Decision:** Model-owned. The model loads its HF tokenizer; the mask threads the metrics forward (`ForwardContext.extras`) and the explainer (`additional_forward_args`). Token attribution uses `LayerIntegratedGradients` over embeddings because token ids are discrete.

**Consequences:** `InputModality.text` carries no task assumption, so seq2seq (#285) must add its own task family without editing the text loader. That litmus is how we keep #340 from overfitting.

## The tree extra bundles scikit-learn and CPU torch

**Context:** A `.ubj` config infers the `xgboost` extra, but `XGBoostBackend` loads through the sklearn-API `XGBClassifier` and the pipeline is torch-based (`torch.from_numpy`).

**Options:** Ship only `xgboost` in the extra; bundle scikit-learn and CPU torch alongside it.

**Decision:** Bundle scikit-learn and CPU torch, with torch routed to the CPU index and conflicting against the cuda/xpu torch extras.

**Consequences:** Tree torch is always CPU plumbing, so the extra stays hardware-variant-free. `infer_extras` needs no change: it still returns bare `xgboost`, and the extra's package list carries the rest.

## Third-party plugins load at bootstrap, default-allow

**Context:** External packages should register their own adapters without a RAITAP release, but a bad plugin must not sink a run.

**Options:** Opt-in allowlist; default-allow with an opt-out.

**Decision:** Default-allow (#173) over the `raitap.adapters` entry-point group. `discover_third_party_adapters()` fires plugin decorators at bootstrap, before config composition, with `RAITAP_DISABLE_PLUGINS=1` to opt out.

**Consequences:** Zero-config for plugin authors. Each plugin loads under its own try/except so one failure logs and continues. The version check reuses the plugin's pip `Requires-Dist` raitap specifier against `raitap.__about__`; a mismatch warns and skips. See {doc}`writing-a-plugin`.

## Cross-cutting principles

The threads that recur across the records above:

- Extend through registries and one-class-drop, not core edits.
- Prefer inline, obvious code over premature abstraction. Revisit, but do not unify.
- Type and gate lazily at real boundaries, never as a sweep.
- Stamp derived facts at their source site.
- Pre-1.0 clean breaks. Docs under `docs/modules/**` and the contributor docs update with every config or behaviour change.
