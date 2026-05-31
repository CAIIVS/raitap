"""Decorator integration test: a decorated stub adapter must land in _BUILDERS
under the transparency group and pass the algorithm/payload metadata through."""

from __future__ import annotations

import torch
import torch.nn as nn

from raitap import adapters
from raitap.transparency.contracts import (
    ExplainerSemanticsHints,
    ExplanationPayloadKind,
    MethodFamily,
)
from raitap.transparency.explainers.base_explainer import AttributionOnlyExplainer


def test_transparency_adapter_registers_and_assigns_classvars() -> None:
    @adapters.transparency(
        registry_name="_stub_xai",
        extra="_stub_extra",
        library="_stub_lib",
        algorithm_registry={"alg": ExplainerSemanticsHints(frozenset({MethodFamily.GRADIENT}))},
    )
    class _StubExplainer(AttributionOnlyExplainer):
        def __init__(self, algorithm: str):
            super().__init__()
            self.algorithm = algorithm

        def check_backend_compat(self, backend: object) -> None:
            del backend

        def compute_attributions(
            self,
            model: nn.Module,
            inputs: torch.Tensor,
            backend: object | None = None,
            **kw: object,
        ) -> torch.Tensor:
            del model, inputs, backend, kw
            return torch.zeros(0)

    from raitap._adapters import _BUILDERS, ADAPTER_EXTRAS

    assert "_stub_xai" in _BUILDERS["transparency"]
    assert ADAPTER_EXTRAS["_StubExplainer"] == "_stub_extra"
    # output_payload_kind defaults to ATTRIBUTIONS when the decorator kwarg is omitted.
    assert _StubExplainer.output_payload_kind is ExplanationPayloadKind.ATTRIBUTIONS
    assert _StubExplainer.algorithm_registry == {
        "alg": ExplainerSemanticsHints(frozenset({MethodFamily.GRADIENT}))
    }
