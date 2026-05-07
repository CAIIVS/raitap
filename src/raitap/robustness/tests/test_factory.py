from __future__ import annotations

from typing import Any

import pytest

from raitap.robustness.assessors import TorchattacksAssessor
from raitap.robustness.contracts import MethodKind
from raitap.robustness.exceptions import MethodKindVisualiserIncompatibilityError
from raitap.robustness.factory import (
    _parse_assessor_config,
    _resolve_call_data_sources,
    check_assessor_visualiser_compat,
)
from raitap.robustness.results import ConfiguredRobustnessVisualiser
from raitap.robustness.visualisers.base_visualiser import BaseRobustnessVisualiser


class _OnlyFormalVisualiser(BaseRobustnessVisualiser):
    supported_method_kinds = frozenset({MethodKind.FORMAL_VERIFICATION})

    def visualise(self, result, *, context, **kwargs) -> Any:  # noqa: ANN001
        raise NotImplementedError


def test_parse_validates_top_level_keys() -> None:
    with pytest.raises(ValueError, match="Unknown robustness assessor config keys"):
        _parse_assessor_config(
            {
                "_target_": "TorchattacksAssessor",
                "algorithm": "PGD",
                "wibble": True,  # unknown key
            }
        )


def test_parse_migrates_misplaced_raitap_keys() -> None:
    parsed = _parse_assessor_config(
        {
            "_target_": "TorchattacksAssessor",
            "algorithm": "PGD",
            "call": {"eps": 0.03, "batch_size": 8},
        }
    )
    # batch_size is RAITAP-owned; factory migrates it from `call` to `raitap`.
    assert "batch_size" not in parsed.call
    assert parsed.raitap["batch_size"] == 8


def test_resolve_call_data_sources_passes_through_non_source_dicts() -> None:
    out = _resolve_call_data_sources({"target_labels": [0, 1]})
    assert out == {"target_labels": [0, 1]}


def test_check_visualiser_compat_raises_on_method_kind_mismatch() -> None:
    assessor = TorchattacksAssessor(algorithm="PGD")  # EMPIRICAL_ATTACK
    visualiser = _OnlyFormalVisualiser()
    configured = [ConfiguredRobustnessVisualiser(visualiser=visualiser)]
    with pytest.raises(MethodKindVisualiserIncompatibilityError):
        check_assessor_visualiser_compat(
            assessor,
            "raitap.robustness.assessors.TorchattacksAssessor",
            configured,
        )
