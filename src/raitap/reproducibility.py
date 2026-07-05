"""Reproducibility partition for stochastic methods (issues #251, #339).

A run is bit-reproducible only if every method it ran is deterministic, or its
RNG source is covered by a pinned global seed. Methods declare their RNG
source (:data:`Seeding`) per algorithm in their ``algorithm_registry``; it
flows onto each result's ``semantics.seeding``. This module partitions the
stochastic methods in a finished :class:`~raitap.pipeline.outputs.RunOutputs`
against a (maybe-unset) seed into ``reproducible`` (``global_rng`` methods
covered by the pinned seed) and ``warned`` (``self_seeded`` methods, always;
``global_rng`` methods when no seed is pinned), and renders the caveat for the
three surfaces (report banner, ``REPRODUCIBILITY.md``, CLI warning). It is
pure: callers own the side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.pipeline.outputs import RunOutputs

REPRODUCIBILITY_FILENAME = "REPRODUCIBILITY.md"

Seeding = Literal["deterministic", "global_rng", "self_seeded"]
"""Per-algorithm RNG-source classification (issue #251 follow-up).

``deterministic`` no RNG; ``global_rng`` draws from the process-global
torch/numpy/random RNG (covered by :func:`pin_global_seed`); ``self_seeded``
owns a seed param that time-defaults (a global seed does not reach it).
"""


def pin_global_seed(seed: int) -> None:
    """Pin the process-global torch / numpy / random RNGs to ``seed``.

    Covers every ``global_rng`` method in the run. Does not reach ``self_seeded``
    methods (they own their seed param). Imports are local so the reproducibility
    module stays import-light for callers that never seed.
    """
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass(frozen=True)
class StochasticMethod:
    """A method in a run whose result is not bit-reproducible."""

    module: str  # "transparency" | "robustness"
    name: str  # user-facing label (config name, falling back to algorithm)
    algorithm: str
    seeding: Seeding  # "global_rng" | "self_seeded" (never "deterministic")


@dataclass(frozen=True)
class ReproducibilityReport:
    """Partition of a run's stochastic methods against a (maybe-unset) seed."""

    seed: int | None
    reproducible: list[StochasticMethod]  # global_rng, covered by the pinned seed
    warned: list[StochasticMethod]  # emit the caveat for these


def stochastic_methods(outputs: RunOutputs) -> list[StochasticMethod]:
    """Every method in this run whose semantics mark it non-deterministic.

    Reads ``result.semantics.seeding`` (set from the per-algorithm registry),
    not live adapter hints — results carry their semantics. Order is stable:
    transparency artefacts first, then robustness, each in run order.
    """
    found: list[StochasticMethod] = []
    for module, results in (
        ("transparency", outputs.transparency),
        ("robustness", outputs.robustness),
    ):
        for result in results:
            semantics = getattr(result, "semantics", None)
            seeding = cast("Seeding", getattr(semantics, "seeding", "deterministic"))
            if seeding == "deterministic":
                continue
            algorithm = str(getattr(result, "algorithm", "") or "")
            name = str(getattr(result, "name", None) or algorithm or "<unnamed>")
            found.append(
                StochasticMethod(module=module, name=name, algorithm=algorithm, seeding=seeding)
            )
    return found


def assess_reproducibility(outputs: RunOutputs, seed: int | None) -> ReproducibilityReport:
    """Partition this run's stochastic methods against ``seed``.

    ``global_rng`` methods are reproducible only when ``seed`` is set (pinning
    the global seed covers them); ``self_seeded`` methods are always warned
    because a global seed does not reach their own seed parameter.
    """
    reproducible: list[StochasticMethod] = []
    warned: list[StochasticMethod] = []
    for method in stochastic_methods(outputs):
        if method.seeding == "global_rng" and seed is not None:
            reproducible.append(method)
        else:
            warned.append(method)
    return ReproducibilityReport(seed=seed, reproducible=reproducible, warned=warned)


def reproducibility_caveat(report: ReproducibilityReport) -> str | None:
    """One-line caveat naming the warned methods, or ``None`` when there are none."""
    if not report.warned:
        return None
    warned_names = ", ".join(sorted({m.name for m in report.warned}))
    if report.seed is None:
        return (
            f"This run includes stochastic methods ({warned_names}); results are "
            "not bit-reproducible unless a seed is pinned (config `seed`)."
        )
    self_seeded = ", ".join(sorted({m.name for m in report.warned if m.seeding == "self_seeded"}))
    covered = ", ".join(sorted({m.name for m in report.reproducible}))
    prefix = f"Run seed={report.seed}."
    if covered:
        prefix += f" Reproducible under this seed: {covered}."
    return (
        f"{prefix} NOT reproducible ({self_seeded}); these self-seed, pass each "
        "method's own seed parameter."
    )


def write_reproducibility_md(run_dir: Path, report: ReproducibilityReport) -> Path:
    """Write ``REPRODUCIBILITY.md`` into ``run_dir``. Returns the written path.

    Callers (the orchestrator) decide when to invoke this: when ``report.warned``
    is non-empty, or when ``report.seed is not None``. A fully-reproducible run
    still records the seed run-wide, since this file is the run-level
    reproducibility artefact — the seed is not duplicated into per-module
    metadata.
    """
    seed_line = f"Seed: {report.seed}" if report.seed is not None else "Seed: not pinned"
    lines = ["# Reproducibility", "", seed_line, ""]
    caveat = reproducibility_caveat(report)
    if caveat is not None:
        lines.append(caveat)
        lines.append("")
    lines.append("## Stochastic artefacts")
    lines.append("")
    lines.extend(
        f"- `{method.name}` ({method.module}, algorithm `{method.algorithm}`)"
        for method in sorted(report.warned, key=lambda item: (item.module, item.name))
    )
    lines.append("")
    if report.seed is not None:
        lines.append("## Reproducible under this seed")
        lines.append("")
        lines.extend(
            f"- `{method.name}` ({method.module}, algorithm `{method.algorithm}`)"
            for method in sorted(report.reproducible, key=lambda item: (item.module, item.name))
        )
        lines.append("")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / REPRODUCIBILITY_FILENAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
