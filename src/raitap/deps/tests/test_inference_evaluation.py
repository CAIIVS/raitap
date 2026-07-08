"""Nested transparency.<name>.evaluation.use must also infer an extra (#341)."""

from __future__ import annotations

from raitap.deps.inference import infer_extras
from raitap.types import ResolvedHardware


def test_nested_evaluation_use_infers_quantus() -> None:
    cfg = {
        "model": {"source": "foo.pt"},
        "transparency": {
            "ig": {
                "use": "captum",
                "evaluation": {"use": "quantus"},
            }
        },
    }
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "quantus" in extras
