"""Reproducibility caveat for stochastic methods (issue #251).

A run is bit-reproducible only if every method it ran is deterministic. Methods
declare ``stochastic`` per algorithm in their ``algorithm_registry``; the flag
flows onto each result's ``semantics``. This module derives, from a finished
:class:`~raitap.pipeline.outputs.RunOutputs`, the list of stochastic methods and
renders the caveat for the three surfaces (report banner, ``REPRODUCIBILITY.md``,
CLI warning). It is pure: callers own the side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence
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


def stochastic_methods(outputs: RunOutputs) -> list[StochasticMethod]:
    """Every method in this run whose semantics mark it stochastic.

    Reads ``result.semantics.stochastic`` (set from the per-algorithm registry),
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
            if not getattr(semantics, "stochastic", False):
                continue
            algorithm = str(getattr(result, "algorithm", "") or "")
            name = str(getattr(result, "name", None) or algorithm or "<unnamed>")
            found.append(StochasticMethod(module=module, name=name, algorithm=algorithm))
    return found


def reproducibility_caveat(methods: Sequence[StochasticMethod]) -> str:
    """One-line caveat naming the stochastic methods."""
    names = ", ".join(sorted({method.name for method in methods}))
    return (
        f"This run includes stochastic methods ({names}); results are not "
        "bit-reproducible unless seeds are pinned."
    )


def write_reproducibility_md(run_dir: Path, methods: Sequence[StochasticMethod]) -> Path:
    """Write ``REPRODUCIBILITY.md`` into ``run_dir`` listing the stochastic artefacts.

    Returns the written path. Callers should only invoke this when ``methods`` is
    non-empty (a fully-deterministic run writes no file).
    """
    lines = [
        "# Reproducibility",
        "",
        reproducibility_caveat(methods),
        "",
        "## Stochastic artefacts",
        "",
    ]
    lines.extend(
        f"- `{method.name}` ({method.module}, algorithm `{method.algorithm}`)"
        for method in sorted(methods, key=lambda item: (item.module, item.name))
    )
    lines.append("")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / REPRODUCIBILITY_FILENAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
