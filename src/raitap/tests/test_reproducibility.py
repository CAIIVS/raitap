"""Tests for the reproducibility partition and caveat derivation (#251, #339)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from raitap.reproducibility import (
    REPRODUCIBILITY_FILENAME,
    ReproducibilityReport,
    StochasticMethod,
    assess_reproducibility,
    reproducibility_caveat,
    stochastic_methods,
    write_reproducibility_md,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from raitap.pipeline.outputs import RunOutputs


def _result(name: str | None, algorithm: str, *, seeding: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        algorithm=algorithm,
        semantics=SimpleNamespace(seeding=seeding, stochastic=seeding != "deterministic"),
    )


def _outputs(transparency: Sequence[object] = (), robustness: Sequence[object] = ()) -> RunOutputs:
    # ``stochastic_methods`` reads results structurally (getattr); a namespace
    # stand-in is enough and avoids constructing a full RunOutputs.
    return cast(
        "RunOutputs",
        SimpleNamespace(transparency=list(transparency), robustness=list(robustness)),
    )


def _named_outputs(*seedings: str) -> RunOutputs:
    # Matches the brief's fixture: m0, m1, ... named by position, all under
    # "transparency", each carrying only ``semantics.seeding``/``.stochastic``.
    results = [
        SimpleNamespace(
            name=f"m{i}",
            algorithm=f"algo{i}",
            semantics=SimpleNamespace(seeding=s, stochastic=s != "deterministic"),
        )
        for i, s in enumerate(seedings)
    ]
    return cast("RunOutputs", SimpleNamespace(transparency=results, robustness=[]))


def test_stochastic_methods_filters_on_semantics() -> None:
    outputs = _outputs(
        transparency=[
            _result("sal", "Saliency", seeding="deterministic"),
            _result("grad", "GradientExplainer", seeding="global_rng"),
        ],
        robustness=[_result("pgd", "PGD", seeding="self_seeded")],
    )

    got = stochastic_methods(outputs)

    assert {m.name for m in got} == {"grad", "pgd"}
    assert {m.module for m in got} == {"transparency", "robustness"}
    assert {m.name: m.seeding for m in got} == {"grad": "global_rng", "pgd": "self_seeded"}


def test_stochastic_methods_empty_when_all_deterministic() -> None:
    outputs = _outputs(transparency=[_result("sal", "Saliency", seeding="deterministic")])
    assert stochastic_methods(outputs) == []


def test_name_falls_back_to_algorithm_when_unnamed() -> None:
    outputs = _outputs(robustness=[_result(None, "PGD", seeding="self_seeded")])
    assert stochastic_methods(outputs)[0].name == "PGD"


def test_caveat_names_methods() -> None:
    report = ReproducibilityReport(
        seed=None,
        reproducible=[],
        warned=[StochasticMethod("robustness", "PGD", "PGD", "self_seeded")],
    )
    msg = reproducibility_caveat(report)
    assert msg is not None
    assert "PGD" in msg
    assert "not" in msg
    assert "bit-reproducible" in msg


def test_write_reproducibility_md(tmp_path: Path) -> None:
    warned = [
        StochasticMethod("robustness", "PGD", "PGD", "self_seeded"),
        StochasticMethod("transparency", "grad", "GradientExplainer", "self_seeded"),
    ]
    report = ReproducibilityReport(seed=None, reproducible=[], warned=warned)

    path = write_reproducibility_md(tmp_path, report)

    assert path.name == REPRODUCIBILITY_FILENAME
    text = path.read_text(encoding="utf-8")
    assert "Stochastic artefacts" in text
    assert "PGD" in text
    assert "GradientExplainer" in text


def test_pin_global_seed_makes_torch_deterministic() -> None:
    import torch

    from raitap.reproducibility import pin_global_seed

    pin_global_seed(1234)
    first = torch.rand(3)
    pin_global_seed(1234)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_pin_global_seed_sets_numpy_and_random() -> None:
    import random

    import numpy as np

    from raitap.reproducibility import pin_global_seed

    pin_global_seed(7)
    np_first, py_first = np.random.rand(3).tolist(), [random.random() for _ in range(3)]
    pin_global_seed(7)
    np_second, py_second = np.random.rand(3).tolist(), [random.random() for _ in range(3)]
    assert np_first == np_second
    assert py_first == py_second


# --- Seed-aware partition (issue #339) --------------------------------------


def test_global_rng_reproducible_when_seed_set() -> None:
    report = assess_reproducibility(_named_outputs("global_rng"), seed=42)
    assert [m.name for m in report.reproducible] == ["m0"]
    assert report.warned == []
    assert reproducibility_caveat(report) is None


def test_self_seeded_always_warned() -> None:
    report = assess_reproducibility(_named_outputs("self_seeded"), seed=42)
    assert [m.name for m in report.warned] == ["m0"]
    caveat = reproducibility_caveat(report)
    assert caveat is not None
    assert "m0" in caveat


def test_seed_unset_warns_all_stochastic() -> None:
    report = assess_reproducibility(_named_outputs("global_rng", "self_seeded"), seed=None)
    assert {m.name for m in report.warned} == {"m0", "m1"}


def test_deterministic_run_no_caveat() -> None:
    report = assess_reproducibility(_named_outputs("deterministic"), seed=None)
    assert report.warned == [] and report.reproducible == []
    assert reproducibility_caveat(report) is None


def test_reproducibility_md_records_seed_when_fully_reproducible(tmp_path: Path) -> None:
    report = assess_reproducibility(_named_outputs("global_rng"), seed=42)  # no warned methods
    path = write_reproducibility_md(tmp_path, report)
    body = path.read_text(encoding="utf-8")
    assert "42" in body  # the seed is recorded run-wide even with nothing warned
    assert "m0" in body  # listed as reproducible under the seed
