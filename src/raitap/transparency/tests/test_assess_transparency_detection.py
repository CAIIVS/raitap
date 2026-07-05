"""Routing test for ``assess_transparency`` when ``task_kind == detection``.

The detection branch reuses the per-explainer setup that
``prepare_explainer`` runs for every task family (create explainer +
visualisers, compat checks, resolve backend + run_dir). This test asserts
the routing reaches ``explain_detection`` with
the right kwargs and emits one ExplanationResult per kept detection box —
without booting a real backend or explainer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from raitap.transparency.results import ExplanationResult

from raitap.configs import set_output_root
from raitap.configs.schema import AppConfig, TransparencyConfig
from raitap.models.base_backend import ModelBackend
from raitap.pipeline.outputs import ForwardOutput
from raitap.transparency.phase import assess_transparency
from raitap.types import Capability, TaskKind


class _FakeBackend(ModelBackend):
    """Minimal backend stub the detection branch only needs to read.

    Subclasses :class:`ModelBackend` so ``_require_model_backend`` accepts it.
    """

    provides = frozenset({Capability.AUTOGRAD})

    def __init__(self) -> None:
        self._task_kind = TaskKind.detection

    @property
    def task_kind(self) -> TaskKind:
        return self._task_kind

    @property
    def hardware_label(self) -> str:
        return "fake"

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs

    def __call__(self, inputs: torch.Tensor) -> Any:
        raise NotImplementedError("routing test mocks the explain path")


class _FakeModel:
    def __init__(self) -> None:
        self.backend = _FakeBackend()


def _make_detection_forward_output(num_samples: int) -> ForwardOutput:
    return ForwardOutput(
        task_kind=TaskKind.detection,
        batch_size=num_samples,
        payload=[
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3], dtype=torch.int64),
            }
            for _ in range(num_samples)
        ],
    )


class _FakeData:
    def __init__(self, num_samples: int = 2) -> None:
        self.tensor = torch.zeros(num_samples, 3, 64, 64)
        self.sample_ids = [f"img_{i}.jpg" for i in range(num_samples)]


def test_assess_transparency_routes_detection_kind_to_explain_detection(
    tmp_path: Path,
) -> None:
    """ForwardOutput(detection) → ``_assess_transparency_detection`` →
    ``explain_detection`` is called once per configured explainer with the
    correct backend, inputs, and forward_output."""
    config = AppConfig(experiment_name="routing-test")
    set_output_root(config, tmp_path)
    config.transparency = {
        "ig_det": TransparencyConfig(
            _target_="CaptumExplainer",
            algorithm="IntegratedGradients",
            call={"target": 0},
            raitap={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            visualisers=[{"_target_": "DetectionImageVisualiser"}],
        )
    }

    model = _FakeModel()
    data = _FakeData(num_samples=2)
    forward = _make_detection_forward_output(num_samples=2)

    captured: dict[str, Any] = {}

    def fake_explain_detection(**kwargs: Any) -> Iterator[ExplanationResult]:
        captured.update(kwargs)
        # Yield one fake result per sample so we can assert downstream flatten.
        from raitap.transparency.results import ExplanationResult as _Result

        for _ in range(forward.batch_size):
            r = _Result.__new__(_Result)
            r.attributions = torch.zeros(1, 3, 64, 64)
            r.inputs = torch.zeros(1, 3, 64, 64)
            r.visualisers = []
            yield r

    class _FakeExplainer:
        algorithm = "IntegratedGradients"

        def check_backend_compat(self, backend: object) -> None:
            return None

    with (
        patch(
            "raitap.transparency.factory.create_explainer",
            return_value=(_FakeExplainer(), "raitap.transparency.CaptumExplainer"),
        ),
        patch("raitap.transparency.factory.create_visualisers", return_value=[]),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_payload_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_semantic_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.explain_detection.explain_detection",
            side_effect=fake_explain_detection,
        ),
    ):
        explanations = assess_transparency(
            config,
            model,  # type: ignore[arg-type]
            data,  # type: ignore[arg-type]
            forward,
            input_metadata=None,
            resolved_preprocessing=None,
        ).explanations

    # The routing reached explain_detection with the right kwargs:
    assert captured["backend"] is model.backend
    assert captured["forward_output"] is forward
    assert captured["explainer_name"] == "ig_det"
    assert captured["explainer_target"] == "raitap.transparency.CaptumExplainer"
    assert "raitap_kwargs" in captured
    assert "call_kwargs" in captured
    # And the fake results were flattened into the explanations list:
    assert len(explanations) == 2


def test_detection_transparency_renders_class_name_end_to_end(tmp_path: Path) -> None:
    """Caller-side enrichment must fill ``DetectionBox.label_name`` from the
    resolved category-names table BEFORE ``result.visualise()`` runs, so the
    configured class name lands on the in-memory box that the report builder
    later reads for the heading + overlay. The per-box figure itself stays
    title-less (asserted below).

    ``explain_detection`` is mocked to yield a *real* ``ExplanationResult`` that
    carries a raw box (``label_name=None``) plus a real
    ``DetectionImageVisualiser``; the only thing under test is that the
    transparency caller enriches the box ahead of the render path."""
    from raitap.transparency.contracts import (
        DetectionBox,
        ExplanationOutputSpace,
        ExplanationPayloadKind,
        ExplanationScope,
        InputSpec,
        MethodFamily,
        OutputSpaceSpec,
        ScopeDefinitionStep,
    )
    from raitap.transparency.results import (
        ConfiguredVisualiser,
        ExplanationResult,
        ExplanationSemantics,
    )
    from raitap.transparency.visualisers.detection_image_visualiser import (
        DetectionImageVisualiser,
    )

    predicted_label_id = 7
    config = AppConfig(experiment_name="render-test")
    set_output_root(config, tmp_path)
    config.model.class_names = (
        ["__background__"]
        + [f"c{i}" for i in range(1, predicted_label_id)]
        + ["kite"]
        + [f"c{i}" for i in range(predicted_label_id + 1, 91)]
    )
    config.transparency = {
        "ig_det": TransparencyConfig(
            _target_="CaptumExplainer",
            algorithm="IntegratedGradients",
            call={"target": 0},
            raitap={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            visualisers=[{"_target_": "DetectionImageVisualiser"}],
        )
    }

    model = _FakeModel()
    data = _FakeData(num_samples=1)
    forward = _make_detection_forward_output(num_samples=1)

    semantics = ExplanationSemantics(
        scope=ExplanationScope.LOCAL,
        scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
        payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
        method_families=frozenset({MethodFamily.GRADIENT}),
        target=None,
        sample_selection=None,
        input_spec=InputSpec(kind="image", shape=(1, 3, 64, 64), layout="NCHW"),
        output_space=OutputSpaceSpec(
            space=ExplanationOutputSpace.DETECTION_BOXES,
            shape=(1, 3, 64, 64),
            layout="NCHW",
        ),
    )

    def fake_explain_detection(**kwargs: Any) -> Iterator[ExplanationResult]:
        run_dir = kwargs["base_run_dir"] / "sample_0" / "box_0"
        result = ExplanationResult(
            attributions=torch.zeros(1, 3, 64, 64),
            inputs=torch.rand(1, 3, 64, 64),
            run_dir=run_dir,
            experiment_name="render-test",
            adapter_target=kwargs["explainer_target"],
            algorithm="IntegratedGradients",
            name=kwargs["explainer_name"],
            visualisers=[ConfiguredVisualiser(DetectionImageVisualiser())],
            semantics=semantics,
        )
        # Raw box as explain_detection emits it: label_name is unresolved (None).
        result.detection_box = DetectionBox(
            display_index=0,
            raw_index=0,
            xyxy=(10.0, 10.0, 50.0, 50.0),
            score=0.9,
            label_index=predicted_label_id,
            label_name=None,
        )
        result.original_sample_index = 0
        yield result

    class _FakeExplainer:
        algorithm = "IntegratedGradients"

        def check_backend_compat(self, backend: object) -> None:
            return None

    with (
        patch(
            "raitap.transparency.factory.create_explainer",
            return_value=(_FakeExplainer(), "raitap.transparency.CaptumExplainer"),
        ),
        patch(
            "raitap.transparency.factory.create_visualisers",
            return_value=[ConfiguredVisualiser(DetectionImageVisualiser())],
        ),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_payload_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_semantic_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.explain_detection.explain_detection",
            side_effect=fake_explain_detection,
        ),
    ):
        explanations = assess_transparency(
            config,
            model,  # type: ignore[arg-type]
            data,  # type: ignore[arg-type]
            forward,
            input_metadata=None,
            resolved_preprocessing=None,
        ).explanations
    visualisations = [v for e in explanations for v in e.visualisations]

    # Caller enrichment populated the in-memory box that the report builder
    # later reads for the heading + overlay.
    assert any(
        e.detection_box is not None and e.detection_box.label_name == "kite" for e in explanations
    )
    # The render path ran end-to-end through the real visualiser and produced a
    # figure with NO title (label/score/GT live on the overlay + heading only).
    titles = [v.figure.axes[0].get_title() for v in visualisations if hasattr(v, "figure")]
    assert titles  # at least one per-box figure rendered
    assert all(t == "" for t in titles)
    assert all(not v.figure.texts for v in visualisations if hasattr(v, "figure"))  # no suptitle


def test_detection_transparency_matches_ground_truth_end_to_end(tmp_path: Path) -> None:
    """When ``data.labels`` carries detection GT, the caller matches each box to
    GT by IoU and the true label + match IoU land on the in-memory box (which the
    report builder renders on the heading + overlay) — exercising the caller
    GT-wiring branch (positional lookup, bounds guard, threshold pass-through),
    not just ``enrich_detection_box`` in isolation. The per-box figure stays
    title-less (asserted below). The GT class (20) differs from the predicted
    class (7) so the class-agnostic match surfaces a disagreement, which is the
    point of the feature."""
    from raitap.transparency.contracts import (
        DetectionBox,
        ExplanationOutputSpace,
        ExplanationPayloadKind,
        ExplanationScope,
        InputSpec,
        MethodFamily,
        OutputSpaceSpec,
        ScopeDefinitionStep,
    )
    from raitap.transparency.results import (
        ConfiguredVisualiser,
        ExplanationResult,
        ExplanationSemantics,
    )
    from raitap.transparency.visualisers.detection_image_visualiser import (
        DetectionImageVisualiser,
    )

    predicted_label_id = 7
    gt_label_id = 20
    config = AppConfig(experiment_name="gt-match-test")
    set_output_root(config, tmp_path)
    config.model.class_names = ["__background__"] + [f"c{i}" for i in range(1, 91)]
    config.transparency = {
        "ig_det": TransparencyConfig(
            _target_="CaptumExplainer",
            algorithm="IntegratedGradients",
            call={"target": 0},
            raitap={"detection": {"score_threshold": 0.5, "max_boxes": 5, "iou_threshold": 0.5}},
            visualisers=[{"_target_": "DetectionImageVisualiser"}],
        )
    }

    model = _FakeModel()
    data = _FakeData(num_samples=1)
    # GT for sample 0: one box at the SAME coords as the predicted box (IoU 1.0)
    # but a different class id, so the match is spatial (class-agnostic).
    data.labels = [  # type: ignore[attr-defined]
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([gt_label_id], dtype=torch.int64),
        }
    ]
    forward = _make_detection_forward_output(num_samples=1)

    semantics = ExplanationSemantics(
        scope=ExplanationScope.LOCAL,
        scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
        payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
        method_families=frozenset({MethodFamily.GRADIENT}),
        target=None,
        sample_selection=None,
        input_spec=InputSpec(kind="image", shape=(1, 3, 64, 64), layout="NCHW"),
        output_space=OutputSpaceSpec(
            space=ExplanationOutputSpace.DETECTION_BOXES,
            shape=(1, 3, 64, 64),
            layout="NCHW",
        ),
    )

    def fake_explain_detection(**kwargs: Any) -> Iterator[ExplanationResult]:
        run_dir = kwargs["base_run_dir"] / "sample_0" / "box_0"
        result = ExplanationResult(
            attributions=torch.zeros(1, 3, 64, 64),
            inputs=torch.rand(1, 3, 64, 64),
            run_dir=run_dir,
            experiment_name="gt-match-test",
            adapter_target=kwargs["explainer_target"],
            algorithm="IntegratedGradients",
            name=kwargs["explainer_name"],
            visualisers=[ConfiguredVisualiser(DetectionImageVisualiser())],
            semantics=semantics,
        )
        result.detection_box = DetectionBox(
            display_index=0,
            raw_index=0,
            xyxy=(10.0, 10.0, 50.0, 50.0),
            score=0.9,
            label_index=predicted_label_id,
            label_name=None,
        )
        result.original_sample_index = 0
        yield result

    class _FakeExplainer:
        algorithm = "IntegratedGradients"

        def check_backend_compat(self, backend: object) -> None:
            return None

    with (
        patch(
            "raitap.transparency.factory.create_explainer",
            return_value=(_FakeExplainer(), "raitap.transparency.CaptumExplainer"),
        ),
        patch(
            "raitap.transparency.factory.create_visualisers",
            return_value=[ConfiguredVisualiser(DetectionImageVisualiser())],
        ),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_payload_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_semantic_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.explain_detection.explain_detection",
            side_effect=fake_explain_detection,
        ),
    ):
        explanations = assess_transparency(
            config,
            model,  # type: ignore[arg-type]
            data,  # type: ignore[arg-type]
            forward,
            input_metadata=None,
            resolved_preprocessing=None,
        ).explanations
    visualisations = [v for e in explanations for v in e.visualisations]

    # GT match reached the in-memory box: evaluated, matched, true class = GT (20).
    # The builder renders this on the heading + overlay; the per-box figure has
    # no title (asserted below), so the GT detail is carried by the box itself.
    box = explanations[0].detection_box
    assert box is not None
    assert box.ground_truth_evaluated is True
    assert box.true_label_index == gt_label_id
    assert box.true_label_name == f"c{gt_label_id}"
    assert box.true_match_iou == pytest.approx(1.0)
    # The render path ran and produced a title-less figure.
    titles = [v.figure.axes[0].get_title() for v in visualisations if hasattr(v, "figure")]
    assert titles
    assert all(t == "" for t in titles)
    assert all(not v.figure.texts for v in visualisations if hasattr(v, "figure"))  # no suptitle


def test_detection_per_box_figure_has_no_suptitle_with_sample_names(tmp_path: Path) -> None:
    # Regression guard (#233): with the axis title removed, the sample-name
    # suptitle fallback in ExplanationResult.visualise must stay suppressed for
    # detection per-box figures even when show_sample_names is on — they are
    # title-less by design (info lives on the overlay + heading).
    from raitap.transparency.contracts import (
        DetectionBox,
        ExplanationOutputSpace,
        ExplanationPayloadKind,
        ExplanationScope,
        InputSpec,
        MethodFamily,
        OutputSpaceSpec,
        ScopeDefinitionStep,
    )
    from raitap.transparency.results import (
        ConfiguredVisualiser,
        ExplanationResult,
        ExplanationSemantics,
    )
    from raitap.transparency.visualisers.detection_image_visualiser import (
        DetectionImageVisualiser,
    )

    sem = ExplanationSemantics(
        scope=ExplanationScope.LOCAL,
        scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
        payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
        method_families=frozenset({MethodFamily.GRADIENT}),
        target=None,
        sample_selection=None,
        input_spec=InputSpec(kind="image", shape=(1, 3, 32, 32), layout="NCHW"),
        output_space=OutputSpaceSpec(
            space=ExplanationOutputSpace.DETECTION_BOXES, shape=(1, 3, 32, 32), layout="NCHW"
        ),
    )
    result = ExplanationResult(
        attributions=torch.zeros(1, 3, 32, 32),
        inputs=torch.rand(1, 3, 32, 32),
        run_dir=tmp_path / "box_0",
        experiment_name="x",
        adapter_target="t",
        algorithm="IntegratedGradients",
        name="ig",
        semantics=sem,
        visualisers=[ConfiguredVisualiser(DetectionImageVisualiser())],
        kwargs={"show_sample_names": True, "sample_names": ["street.jpg"]},
    )
    result.detection_box = DetectionBox(
        display_index=0,
        raw_index=0,
        xyxy=(2, 3, 22, 23),
        score=0.9,
        label_index=1,
        label_name="kite",
    )
    result.original_sample_index = 0

    vis_results = result._visualise()
    assert vis_results
    for vr in vis_results:
        assert vr.figure.axes[0].get_title() == ""
        assert vr.figure.texts == []  # no sample-name suptitle leaked in


def test_assess_transparency_detection_skips_when_no_explainers(tmp_path: Path) -> None:
    config = AppConfig(experiment_name="empty")
    set_output_root(config, tmp_path)
    forward = _make_detection_forward_output(num_samples=1)

    explanations = assess_transparency(
        config,
        _FakeModel(),  # type: ignore[arg-type]
        _FakeData(num_samples=1),  # type: ignore[arg-type]
        forward,
        input_metadata=None,
        resolved_preprocessing=None,
    ).explanations
    assert explanations == []


def test_assess_transparency_detection_handles_model_block_without_class_names(
    tmp_path: Path,
) -> None:
    """Regression for #240: a struct-mode ``model:`` block that omits
    ``class_names`` must not crash the transparency phase.

    The real pipeline hands ``config.model`` as a YAML-loaded struct-mode
    ``DictConfig`` (``object_type=dict``). When the block never declares the
    optional ``class_names`` key, the unconditional attribute read raised
    ``ConfigAttributeError`` before ``resolve_category_names`` could fall back
    to ``backend.category_names``. The read must be defensive so the optional
    field + backend fallback work as designed.
    """
    from omegaconf import OmegaConf

    config = AppConfig(experiment_name="no-class-names")
    set_output_root(config, tmp_path)
    # Mirror a YAML-loaded model block: a plain struct-mode DictConfig that
    # never declares ``class_names`` (object_type=dict, struct=True).
    model_block = OmegaConf.create({"source": "fasterrcnn_resnet50_fpn_v2"})
    OmegaConf.set_struct(model_block, True)
    config.model = model_block  # type: ignore[assignment]
    config.transparency = {
        "ig_det": TransparencyConfig(
            _target_="CaptumExplainer",
            algorithm="IntegratedGradients",
            call={"target": 0},
            raitap={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            visualisers=[{"_target_": "DetectionImageVisualiser"}],
        )
    }

    model = _FakeModel()
    data = _FakeData(num_samples=1)
    forward = _make_detection_forward_output(num_samples=1)

    def fake_explain_detection(**kwargs: Any) -> Iterator[ExplanationResult]:
        from raitap.transparency.results import ExplanationResult as _Result

        r = _Result.__new__(_Result)
        r.attributions = torch.zeros(1, 3, 64, 64)
        r.inputs = torch.zeros(1, 3, 64, 64)
        r.visualisers = []
        yield r

    class _FakeExplainer:
        algorithm = "IntegratedGradients"

        def check_backend_compat(self, backend: object) -> None:
            return None

    with (
        patch(
            "raitap.transparency.factory.create_explainer",
            return_value=(_FakeExplainer(), "raitap.transparency.CaptumExplainer"),
        ),
        patch("raitap.transparency.factory.create_visualisers", return_value=[]),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_payload_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_semantic_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.explain_detection.explain_detection",
            side_effect=fake_explain_detection,
        ),
    ):
        explanations = assess_transparency(
            config,
            model,  # type: ignore[arg-type]
            data,  # type: ignore[arg-type]
            forward,
            input_metadata=None,
            resolved_preprocessing=None,
        ).explanations

    assert len(explanations) == 1
