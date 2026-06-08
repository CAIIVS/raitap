"""Persistence + metadata for structured payloads on ExplanationResult (#101)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    ExplanationSemantics,
    MethodFamily,
    OutputSpaceSpec,
    ScopeDefinitionStep,
    StructuredPayload,
    StructuredPayloadKind,
)
from raitap.transparency.results import ExplanationResult


def _semantics() -> ExplanationSemantics:
    return ExplanationSemantics(
        scope=ExplanationScope.LOCAL,
        scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
        payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
        method_families=frozenset({MethodFamily.GRADIENT}),
        target=None,
        sample_selection=None,
        input_spec=None,
        output_space=OutputSpaceSpec(
            space=ExplanationOutputSpace.INPUT_FEATURES, shape=(2, 3), layout="(B, F)"
        ),
    )


def _result(tmp_path: Path, payloads: list[StructuredPayload]) -> ExplanationResult:
    return ExplanationResult(
        attributions=torch.zeros(2, 3),
        inputs=torch.zeros(2, 3),
        run_dir=tmp_path,
        experiment_name="exp",
        adapter_target="X",
        algorithm="IntegratedGradients",
        structured_payloads=payloads,
        semantics=_semantics(),
    )


def test_payload_tensor_detached_to_cpu_in_post_init(tmp_path: Path) -> None:
    payload = StructuredPayload(
        "convergence_delta", StructuredPayloadKind.CONVERGENCE_DELTA, torch.zeros(2)
    )
    result = _result(tmp_path, [payload])
    stored = result.structured_payloads[0].data
    assert stored.device.type == "cpu"


def test_write_artifacts_persists_payload_tensor(tmp_path: Path) -> None:
    payload = StructuredPayload(
        "convergence_delta", StructuredPayloadKind.CONVERGENCE_DELTA, torch.tensor([1.0, 2.0])
    )
    result = _result(tmp_path, [payload])
    result.write_artifacts()

    saved = tmp_path / "payloads" / "convergence_delta.pt"
    assert saved.is_file()
    assert torch.equal(torch.load(saved), torch.tensor([1.0, 2.0]))


def test_metadata_describes_structured_payloads(tmp_path: Path) -> None:
    payload = StructuredPayload(
        "convergence_delta", StructuredPayloadKind.CONVERGENCE_DELTA, torch.zeros(2)
    )
    result = _result(tmp_path, [payload])
    result.write_artifacts()

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["structured_payloads"] == [
        {
            "name": "convergence_delta",
            "kind": "convergence_delta",
            "storage": "tensor",
            "file": "payloads/convergence_delta.pt",
            "shape": [2],
            "dtype": "torch.float32",
        }
    ]


def test_no_payloads_writes_no_payloads_dir_and_no_block(tmp_path: Path) -> None:
    result = _result(tmp_path, [])
    result.write_artifacts()
    assert not (tmp_path / "payloads").exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert "structured_payloads" not in metadata
