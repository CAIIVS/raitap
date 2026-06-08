"""Tests for the structured-payload data model (#101)."""

from __future__ import annotations

import torch

from raitap.transparency.contracts import (
    ExplainerAlgorithmSpec,
    MethodFamily,
    StructuredOutputSpec,
    StructuredPayload,
    StructuredPayloadKind,
)


def test_structured_payload_holds_name_kind_data() -> None:
    data = torch.zeros(4)
    payload = StructuredPayload(
        name="convergence_delta",
        kind=StructuredPayloadKind.CONVERGENCE_DELTA,
        data=data,
    )
    assert payload.name == "convergence_delta"
    assert payload.kind is StructuredPayloadKind.CONVERGENCE_DELTA
    assert payload.data is data


def test_structured_output_spec_defaults_per_sample_true() -> None:
    spec = StructuredOutputSpec(
        name="convergence_delta",
        kind=StructuredPayloadKind.CONVERGENCE_DELTA,
    )
    assert spec.per_sample is True


def test_algorithm_spec_extra_outputs_default_empty() -> None:
    spec = ExplainerAlgorithmSpec({MethodFamily.GRADIENT})
    assert spec.extra_outputs == ()


def test_algorithm_spec_carries_extra_outputs() -> None:
    out = StructuredOutputSpec("convergence_delta", StructuredPayloadKind.CONVERGENCE_DELTA)
    spec = ExplainerAlgorithmSpec({MethodFamily.GRADIENT}, extra_outputs=(out,))
    assert spec.extra_outputs == (out,)
