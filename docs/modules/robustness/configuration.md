```{config-page}
:intro: This page describes how to configure the robustness module that probes
  how the model behaves under input perturbations.

  Inside the `robustness` key, you can configure one or more named assessors.
  See {ref}`modules-robustness-configuration-examples` for the config shape.

  See {doc}`frameworks-and-libraries` for the backend behaviour behind
  `_target_`, `algorithm`, and visualiser compatibility.

:option: _target_
:allowed: "TorchattacksAssessor", "FoolboxAssessor"
:default: null
:description: Hydra target for the assessor class.

:option: algorithm
:allowed: See {doc}`frameworks-and-libraries`
:default: null
:description: Name of the underlying attack algorithm to use. The exact class is
  resolved by the selected assessor backend.

:option: constructor
:allowed: dict
:default: {}
:description: Keyword arguments forwarded when constructing the assessor /
  underlying library object. **Torchattacks** consumes the perturbation budget
  (`eps`, `alpha`, `steps`, ...) here, since the adapter does
  `attack_class(model, **constructor)` once.

:option: call
:allowed: dict
:default: {}
:description: Keyword arguments forwarded verbatim to the underlying library
  at call time. **Foolbox** consumes the perturbation budget (`eps`,
  `epsilons`) here, since foolbox attacks read it at `attack(...)` time. Any
  nested dict with a `source` key is treated as a runtime data source.

:option: raitap
:allowed: dict
:default: {}
:description: RAITAP-owned runtime options such as batching, progress display,
  and sample-name metadata. These keys are not forwarded to the underlying
  library.

:option: raitap.batch_size
:allowed: int
:default: None
:description: Batch size for generating adversarial examples. If unset, the
  assessor processes the full input batch in a single call to the attack
  library. Set this for memory-bound attacks (Square, CW, ...) on large
  batches.

:option: raitap.show_progress
:allowed: bool
:default: True
:description: Whether to show a progress bar across attack batches.

:option: raitap.progress_desc
:allowed: str
:default: null
:description: Description used by the progress bar.

:option: raitap.sample_names
:allowed: list[str]
:default: null
:description: Optional per-sample names for downstream visualisers. Injected at
  runtime from the data pipeline. Runtime sample names take precedence over
  `raitap.sample_names` from config.

:option: raitap.show_sample_names
:allowed: bool
:default: False
:description: Default toggle for showing sample names in visualiser titles. Set
  the assessor-level default here under `raitap:`. If a specific visualiser
  needs different behaviour, override it with
  `visualisers[].call.show_sample_names`.

:option: raitap.input_metadata
:allowed: dict
:default: null
:description: Input modality + layout hints. Used by image visualisers to
  refuse non-image results and by the budget norm to size per-sample distance.
  Auto-inferred from `data.source` for the default loaders. This
  per-assessor metadata is scoped to visualiser/budget semantics; backend
  input reshape is controlled by `data.input_metadata.shape` instead — see
  {doc}`../data/configuration`.

:option: visualisers
:allowed: list[dict]
:default: [ImagePairVisualiser]
:description: Visualiser definitions. Each entry must include at least
  `_target_`. Each visualiser can also define its own `constructor` and `call`
  blocks. Visualisers declare which `MethodKind` (`empirical_attack` /
  `formal_verification`) they support; the factory rejects mismatches at parse
  time.

:yaml:
robustness:
  pgd:
    _target_: "TorchattacksAssessor"
    algorithm: "PGD"
    constructor:
      eps: 0.03
      alpha: 0.0078
      steps: 10
    visualisers:
      - _target_: "ImagePairVisualiser"
  linf_pgd:
    _target_: "FoolboxAssessor"
    algorithm: "LinfPGD"
    constructor:
      rel_stepsize: 0.025
      steps: 40
    call:
      eps: 0.03
    visualisers:
      - _target_: "ImagePairVisualiser"
      - _target_: "PerturbationHeatmapVisualiser"

:cli: +robustness=torchattacks robustness.torchattacks.algorithm=PGD robustness.torchattacks.constructor.eps=0.05

:python:
from raitap.api import foolbox, torchattacks

robustness = {
    "pgd": torchattacks(
        algorithm="PGD",
        constructor={"eps": 0.03, "alpha": 0.0078, "steps": 10},
        visualisers=[{"_target_": "ImagePairVisualiser"}],
    ),
    "linf_pgd": foolbox(
        algorithm="LinfPGD",
        constructor={"rel_stepsize": 0.025, "steps": 40},
        call={"eps": 0.03},
        visualisers=[
            {"_target_": "ImagePairVisualiser"},
            {"_target_": "PerturbationHeatmapVisualiser"},
        ],
    ),
}
```
