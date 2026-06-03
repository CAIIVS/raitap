"""Tests for the detection explain phase."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch
from torch import nn

from raitap.models.backend import TorchBackend
from raitap.pipeline.outputs import ForwardOutput
from raitap.transparency.contracts import (
    DetectionBox,
    ExplainerSemanticsHints,
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    ExplanationSemantics,
    InputKind,
    InputSpec,
    OutputSpaceSpec,
    ScopeDefinitionStep,
    TensorLayout,
)
from raitap.transparency.explain_detection import explain_detection
from raitap.transparency.results import ExplanationResult
from raitap.types import TaskKind
from raitap.utils.errors import RaitapError


class _FakeDetector(nn.Module):
    """Stand-in detector. The phase reads predictions from ForwardOutput, so the
    detector itself is only consulted indirectly via ScalarDetectionWrapper at
    explain time. We return a single-box list[dict] for any inputs."""

    def forward(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        bs = int(images.shape[0])
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
            for _ in range(bs)
        ]


class _RecordingExplainer:
    """Test double — captures every explain() invocation."""

    algorithm = "IntegratedGradients"
    algorithm_registry: ClassVar[dict[str, Any]] = {
        "IntegratedGradients": ExplainerSemanticsHints(frozenset())
    }
    output_payload_kind = ExplanationPayloadKind.ATTRIBUTIONS
    output_scope = ExplanationScope.LOCAL

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def check_backend_compat(self, backend: Any) -> None:
        del backend

    def explain(
        self,
        model: Any,
        inputs: torch.Tensor,
        *,
        backend: Any = None,
        run_dir: Any = None,
        experiment_name: str | None = None,
        explainer_target: str | None = None,
        explainer_name: str | None = None,
        visualisers: list[Any] | None = None,
        raitap_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ExplanationResult:
        self.calls.append(
            {
                "model": model,
                "inputs_shape": tuple(inputs.shape),
                "run_dir": run_dir,
                "call_target": kwargs.get("target"),
                "call_provenance": kwargs.get("call_provenance"),
            }
        )
        semantics = ExplanationSemantics(
            scope=ExplanationScope.LOCAL,
            scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
            payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
            method_families=frozenset(),
            target=None,
            sample_selection=None,
            input_spec=InputSpec(
                kind=InputKind.IMAGE,
                shape=tuple(int(d) for d in inputs.shape),
                layout=TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
            ),
            output_space=OutputSpaceSpec(
                space=ExplanationOutputSpace.DETECTION_BOXES,
                shape=tuple(int(d) for d in inputs.shape),
                layout=TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
            ),
        )
        assert run_dir is not None
        return ExplanationResult(
            attributions=torch.zeros_like(inputs),
            inputs=inputs,
            run_dir=Path(run_dir),
            experiment_name=experiment_name,
            adapter_target=explainer_target or "x",
            algorithm=self.algorithm,
            name=explainer_name,
            visualisers=[],
            semantics=semantics,
        )


@pytest.fixture
def detection_forward_output() -> ForwardOutput:
    """Two-sample forward output. Sample 0 has 3 boxes (0.9 / 0.4 / 0.7);
    sample 1 has zero."""
    return ForwardOutput(
        task_kind=TaskKind.detection,
        batch_size=2,
        detection_predictions=[
            {
                "boxes": torch.tensor(
                    [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0], [5.0, 5.0, 15.0, 15.0]]
                ),
                "scores": torch.tensor([0.9, 0.4, 0.7]),
                "labels": torch.tensor([1, 2, 1], dtype=torch.int64),
            },
            {
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.int64),
            },
        ],
    )


def test_explain_detection_filters_below_threshold_and_caps_at_max_boxes(
    detection_forward_output: ForwardOutput, tmp_path: Path
) -> None:
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    inputs = torch.zeros(2, 3, 8, 8)
    explainer = _RecordingExplainer()

    results = list(
        explain_detection(
            inputs=inputs,
            forward_output=detection_forward_output,
            backend=backend,
            explainer=explainer,
            explainer_target="t",
            explainer_name="x",
            visualisers=[],
            base_run_dir=tmp_path,
            raitap_kwargs={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            call_kwargs={},
        )
    )

    # Sample 0: keep 0.9 and 0.7 (raw indices 0 and 2). Sample 1: empty.
    assert len(results) == 2
    boxes = [r.detection_box for r in results]
    assert boxes[0] is not None and boxes[0].display_index == 0 and boxes[0].raw_index == 0
    assert boxes[0].score == pytest.approx(0.9)
    assert boxes[1] is not None and boxes[1].display_index == 1 and boxes[1].raw_index == 2
    assert boxes[1].score == pytest.approx(0.7)


def test_explain_detection_passes_target_zero_overriding_call_kwargs(
    detection_forward_output: ForwardOutput, tmp_path: Path
) -> None:
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    explainer = _RecordingExplainer()

    list(
        explain_detection(
            inputs=torch.zeros(2, 3, 8, 8),
            forward_output=detection_forward_output,
            backend=backend,
            explainer=explainer,
            explainer_target="t",
            explainer_name="x",
            visualisers=[],
            base_run_dir=tmp_path,
            raitap_kwargs={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            call_kwargs={"target": 7},
        )
    )

    for call in explainer.calls:
        assert call["call_target"] == 0


def test_explain_detection_rejects_auto_pred(
    detection_forward_output: ForwardOutput, tmp_path: Path
) -> None:
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    explainer = _RecordingExplainer()

    with pytest.raises(RaitapError, match="auto_pred"):
        list(
            explain_detection(
                inputs=torch.zeros(2, 3, 8, 8),
                forward_output=detection_forward_output,
                backend=backend,
                explainer=explainer,
                explainer_target="t",
                explainer_name="x",
                visualisers=[],
                base_run_dir=tmp_path,
                raitap_kwargs={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
                call_kwargs={"target": "auto_pred"},
            )
        )


def test_explain_detection_skips_samples_with_no_passing_boxes(
    detection_forward_output: ForwardOutput, tmp_path: Path
) -> None:
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    explainer = _RecordingExplainer()

    # Threshold above any score → 0 results.
    results = list(
        explain_detection(
            inputs=torch.zeros(2, 3, 8, 8),
            forward_output=detection_forward_output,
            backend=backend,
            explainer=explainer,
            explainer_target="t",
            explainer_name="x",
            visualisers=[],
            base_run_dir=tmp_path,
            raitap_kwargs={"detection": {"score_threshold": 0.99, "max_boxes": 5}},
            call_kwargs={},
        )
    )
    assert results == []


def test_explain_detection_per_box_run_dir_layout(
    detection_forward_output: ForwardOutput, tmp_path: Path
) -> None:
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    explainer = _RecordingExplainer()

    list(
        explain_detection(
            inputs=torch.zeros(2, 3, 8, 8),
            forward_output=detection_forward_output,
            backend=backend,
            explainer=explainer,
            explainer_target="t",
            explainer_name="my_explainer",
            visualisers=[],
            base_run_dir=tmp_path,
            raitap_kwargs={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            call_kwargs={},
        )
    )

    expected_dirs = {
        tmp_path / "sample_0" / "box_0",
        tmp_path / "sample_0" / "box_2",
    }
    actual_dirs = {Path(call["run_dir"]) for call in explainer.calls}
    assert actual_dirs == expected_dirs


def test_explain_detection_attaches_detection_box_and_original_sample_index(
    detection_forward_output: ForwardOutput, tmp_path: Path
) -> None:
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    explainer = _RecordingExplainer()

    results = list(
        explain_detection(
            inputs=torch.zeros(2, 3, 8, 8),
            forward_output=detection_forward_output,
            backend=backend,
            explainer=explainer,
            explainer_target="t",
            explainer_name="x",
            visualisers=[],
            base_run_dir=tmp_path,
            raitap_kwargs={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            call_kwargs={},
        )
    )

    assert all(isinstance(r.detection_box, DetectionBox) for r in results)
    assert all(r.original_sample_index == 0 for r in results)
    assert results[0].detection_box is not None
    assert results[0].detection_box.xyxy == (0.0, 0.0, 10.0, 10.0)
    assert results[0].detection_box.label_index == 1


def test_explain_detection_rejects_invalid_max_boxes(
    detection_forward_output: ForwardOutput, tmp_path: Path
) -> None:
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    explainer = _RecordingExplainer()

    with pytest.raises(RaitapError, match="max_boxes"):
        list(
            explain_detection(
                inputs=torch.zeros(2, 3, 8, 8),
                forward_output=detection_forward_output,
                backend=backend,
                explainer=explainer,
                explainer_target="t",
                explainer_name="x",
                visualisers=[],
                base_run_dir=tmp_path,
                raitap_kwargs={"detection": {"max_boxes": 0}},
                call_kwargs={},
            )
        )


def test_explain_detection_with_list_inputs(
    detection_forward_output: ForwardOutput, tmp_path: Path
) -> None:
    """When inputs is a list[Tensor] (ragged, one tensor per image),
    each explain() call must receive a (1, C, H, W) tensor, not a list slice.
    Tests both the correct shape and that different native resolutions are preserved."""
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    explainer = _RecordingExplainer()

    # Two differently-sized images (C=3, but H*W differ) to confirm per-sample sizing.
    # detection_forward_output has sample 0 with 2 boxes passing threshold=0.5 (scores 0.9, 0.7),
    # sample 1 has zero detections.
    list_inputs: list[torch.Tensor] = [
        torch.zeros(3, 8, 8),  # sample 0 — native (3,8,8)
        torch.zeros(3, 12, 10),  # sample 1 — native (3,12,10), but no detections → unused
    ]

    results = list(
        explain_detection(
            inputs=list_inputs,
            forward_output=detection_forward_output,
            backend=backend,
            explainer=explainer,
            explainer_target="t",
            explainer_name="x",
            visualisers=[],
            base_run_dir=tmp_path,
            raitap_kwargs={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            call_kwargs={},
        )
    )

    # Same 2 boxes as the dense-tensor test (scores 0.9 and 0.7 for sample 0).
    assert len(results) == 2
    # The explainer must receive a (1, C, H, W) tensor for sample 0's native size.
    for call in explainer.calls:
        assert call["inputs_shape"] == (1, 3, 8, 8), (
            f"Expected (1, 3, 8, 8) but got {call['inputs_shape']} — "
            "list slice was probably passed instead of unsqueezed tensor"
        )


def test_explain_detection_forwards_call_provenance(
    detection_forward_output: ForwardOutput, tmp_path: Path
) -> None:
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    explainer = _RecordingExplainer()
    provenance = {"baselines": {"source": "cfg", "n_samples": 1}}

    list(
        explain_detection(
            inputs=torch.zeros(2, 3, 8, 8),
            forward_output=detection_forward_output,
            backend=backend,
            explainer=explainer,
            explainer_target="t",
            explainer_name="x",
            visualisers=[],
            base_run_dir=tmp_path,
            raitap_kwargs={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            call_kwargs={},
            call_provenance=provenance,
        )
    )

    assert explainer.calls, "expected at least one explain() call"
    for call in explainer.calls:
        assert call["call_provenance"] == provenance


def test_sample_as_batch_helper_list_and_tensor() -> None:
    """Unit-test _sample_as_batch directly for both list and dense tensor inputs."""
    from raitap.transparency.explain_detection import _sample_as_batch

    # List case — differently-sized tensors.
    t0 = torch.zeros(3, 8, 8)
    t1 = torch.zeros(3, 12, 10)
    lst: list[torch.Tensor] = [t0, t1]
    out0 = _sample_as_batch(lst, 0)
    assert out0.shape == (1, 3, 8, 8)
    out1 = _sample_as_batch(lst, 1)
    assert out1.shape == (1, 3, 12, 10)

    # Dense tensor case.
    batch = torch.zeros(4, 3, 8, 8)
    out = _sample_as_batch(batch, 2)
    assert out.shape == (1, 3, 8, 8)
