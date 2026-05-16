"""Decorator integration test: a decorated stub adapter must land in _BUILDERS
under the transparency group and pass the algorithm/payload metadata through."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn

from raitap.transparency.contracts import ExplanationPayloadKind, MethodFamily
from raitap.transparency.explainers.base_explainer import AttributionOnlyExplainer
from raitap.transparency.explainers.registration import register_transparency_adapter


def test_register_transparency_adapter_registers_and_assigns_classvars() -> None:
    @register_transparency_adapter(
        registry_name="_stub_xai",
        extra="_stub_extra",
        library="_stub_lib",
    )
    class _StubExplainer(AttributionOnlyExplainer):
        # Family-required class-body attrs — validated at decoration time by
        # _register_core against TRANSPARENCY.required_classvars. The decorator
        # itself only carries cross-family kwargs (registry_name/extra/library/...).
        output_payload_kind: ClassVar[ExplanationPayloadKind] = (
            ExplanationPayloadKind.ATTRIBUTIONS
        )
        algorithm_registry: ClassVar[dict[str, frozenset[MethodFamily]]] = {
            "alg": frozenset({MethodFamily.GRADIENT}),
        }

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

    from raitap._adapters import ADAPTER_EXTRAS, _BUILDERS

    assert "_stub_xai" in _BUILDERS["transparency"]
    assert ADAPTER_EXTRAS["_StubExplainer"] == "_stub_extra"
    assert _StubExplainer.output_payload_kind is ExplanationPayloadKind.ATTRIBUTIONS
    assert _StubExplainer.algorithm_registry == {"alg": frozenset({MethodFamily.GRADIENT})}
