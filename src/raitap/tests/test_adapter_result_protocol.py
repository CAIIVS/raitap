"""Every per-adapter result must satisfy the ``AdapterResult`` contract.

This is the thesis guardrail (esp. ``semantics``) made explicit. The contract is
*also* enforced statically by the ``run_adapters`` TypeVar bound, so a future
module that drops an envelope field fails pyright; this test documents it and
catches it at runtime too. Metrics is a singleton — deliberately NOT an
``AdapterResult`` — so it is excluded.
"""

from __future__ import annotations

import importlib

import pytest

from raitap.pipeline.outputs import AdapterResult

_ENVELOPE = ("name", "adapter_target", "algorithm", "semantics", "run_dir", "visualisations")


@pytest.mark.parametrize(
    "result_path",
    [
        "raitap.transparency.results.ExplanationResult",
        "raitap.robustness.results.RobustnessResult",
    ],
)
def test_result_declares_adapter_envelope(result_path: str) -> None:
    module_path, _, cls_name = result_path.rpartition(".")
    cls = getattr(importlib.import_module(module_path), cls_name)
    fields = set(cls.__dataclass_fields__)
    missing = [attr for attr in _ENVELOPE if attr not in fields]
    assert not missing, f"{cls_name} is missing AdapterResult envelope field(s): {missing}"
    assert hasattr(cls, "_visualise"), f"{cls_name} must expose _visualise()"


def test_adapter_result_is_runtime_checkable() -> None:
    assert getattr(AdapterResult, "_is_runtime_protocol", False)
