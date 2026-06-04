import importlib

import pytest

from raitap.types import Capability
from raitap.utils.errors import BackendIncompatibilityError


class _NoCapBackend:
    provides = frozenset()


class _AutogradBackend:
    provides = frozenset({Capability.AUTOGRAD})


@pytest.mark.parametrize(
    ("module_path", "cls_name", "algorithm"),
    [
        ("raitap.robustness.assessors.torchattacks_assessor", "TorchattacksAssessor", "PGD"),
        ("raitap.robustness.assessors.auto_lirpa_assessor", "AutoLiRPAAssessor", "crown"),
        (
            "raitap.transparency.explainers.captum_explainer",
            "CaptumExplainer",
            "IntegratedGradients",
        ),
    ],
)
def test_known_gradient_algorithm_is_rejected_on_no_capability_backend(
    module_path: str, cls_name: str, algorithm: str
) -> None:
    cls = getattr(importlib.import_module(module_path), cls_name)
    adapter = cls(algorithm=algorithm)
    with pytest.raises(BackendIncompatibilityError, match="autograd"):
        adapter.check_backend_compat(_NoCapBackend())
    adapter.check_backend_compat(_AutogradBackend())  # gradient backend -> no raise


def test_model_agnostic_algorithm_runs_on_no_capability_backend() -> None:
    from raitap.transparency.explainers.captum_explainer import CaptumExplainer

    CaptumExplainer(algorithm="Occlusion").check_backend_compat(_NoCapBackend())  # no raise
