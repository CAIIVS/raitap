"""Nested transparency.<name>.evaluation._target_ must also infer an extra (#341)."""

from __future__ import annotations

from raitap.deps.inference import infer_extras
from raitap.types import ResolvedHardware


def test_nested_evaluation_target_infers_quantus() -> None:
    cfg = {
        "model": {"source": "foo.pt"},
        "transparency": {
            "ig": {
                "_target_": "raitap.transparency.CaptumExplainer",
                "evaluation": {"_target_": "raitap.transparency.QuantusEvaluator"},
            }
        },
    }
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "quantus" in extras
