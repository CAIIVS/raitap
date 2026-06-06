"""Tests for the reproducibility caveat derivation (issue #251)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from raitap.reproducibility import (
    REPRODUCIBILITY_FILENAME,
    StochasticMethod,
    reproducibility_caveat,
    stochastic_methods,
    write_reproducibility_md,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from raitap.pipeline.outputs import RunOutputs


def _result(name: str | None, algorithm: str, *, stochastic: bool) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        algorithm=algorithm,
        semantics=SimpleNamespace(stochastic=stochastic),
    )


def _outputs(transparency: Sequence[object] = (), robustness: Sequence[object] = ()) -> RunOutputs:
    # ``stochastic_methods`` reads results structurally (getattr); a namespace
    # stand-in is enough and avoids constructing a full RunOutputs.
    return cast(
        "RunOutputs",
        SimpleNamespace(transparency=list(transparency), robustness=list(robustness)),
    )


def test_stochastic_methods_filters_on_semantics() -> None:
    outputs = _outputs(
        transparency=[
            _result("sal", "Saliency", stochastic=False),
            _result("grad", "GradientExplainer", stochastic=True),
        ],
        robustness=[_result("pgd", "PGD", stochastic=True)],
    )

    got = stochastic_methods(outputs)

    assert {m.name for m in got} == {"grad", "pgd"}
    assert {m.module for m in got} == {"transparency", "robustness"}


def test_stochastic_methods_empty_when_all_deterministic() -> None:
    outputs = _outputs(transparency=[_result("sal", "Saliency", stochastic=False)])
    assert stochastic_methods(outputs) == []


def test_name_falls_back_to_algorithm_when_unnamed() -> None:
    outputs = _outputs(robustness=[_result(None, "PGD", stochastic=True)])
    assert stochastic_methods(outputs)[0].name == "PGD"


def test_caveat_names_methods() -> None:
    msg = reproducibility_caveat([StochasticMethod("robustness", "PGD", "PGD")])
    assert "PGD" in msg
    assert "not" in msg
    assert "bit-reproducible" in msg


def test_write_reproducibility_md(tmp_path: Path) -> None:
    methods = [
        StochasticMethod("robustness", "PGD", "PGD"),
        StochasticMethod("transparency", "grad", "GradientExplainer"),
    ]

    path = write_reproducibility_md(tmp_path, methods)

    assert path.name == REPRODUCIBILITY_FILENAME
    text = path.read_text(encoding="utf-8")
    assert "Stochastic artefacts" in text
    assert "PGD" in text
    assert "GradientExplainer" in text
