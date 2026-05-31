from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

import raitap.transparency.explainers.base_explainer as base_explainer_module
from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    MethodFamily,
    ScopeDefinitionStep,
)
from raitap.transparency.explainers.base_explainer import AttributionOnlyExplainer


class _StrictExplainer(AttributionOnlyExplainer):
    algorithm = "Saliency"

    def __init__(self) -> None:
        super().__init__()
        self.last_target: int | None = None

    def compute_attributions(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        target: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del model, kwargs
        self.last_target = target
        return inputs


class _GlobalScopeExplainer(_StrictExplainer):
    output_scope = ExplanationScope.GLOBAL


class _UnknownAlgorithmExplainer(AttributionOnlyExplainer):
    algorithm = "UnregisteredAlgorithm"

    def __init__(self) -> None:
        super().__init__()
        self.compute_called = False

    def compute_attributions(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del model, kwargs
        self.compute_called = True
        return inputs


class _BatchRecordingExplainer(AttributionOnlyExplainer):
    algorithm = "IntegratedGradients"

    def __init__(self) -> None:
        super().__init__()
        self.seen_targets: list[list[int]] = []
        self.seen_background_sizes: list[int] = []

    def compute_attributions(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        target: list[int] | torch.Tensor | None = None,
        background_data: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del model, kwargs
        if isinstance(target, list):
            self.seen_targets.append(target)
        elif target is not None:
            self.seen_targets.append(cast("list[int]", cast("Any", target).tolist()))
        else:
            self.seen_targets.append([])

        background_size = -1 if background_data is None else int(background_data.shape[0])
        self.seen_background_sizes.append(background_size)
        return inputs


class _BaselineDeclaringExplainer(AttributionOnlyExplainer):
    algorithm = "IntegratedGradients"
    baseline_kwarg = "baselines"
    baseline_defaults: ClassVar[Mapping[str, str]] = {"IntegratedGradients": "zero"}

    def compute_attributions(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        target: int | None = None,
        baselines: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del model, target, baselines, kwargs
        return inputs


class _GradTrackingExplainer(AttributionOnlyExplainer):
    algorithm = "Saliency"

    def compute_attributions(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del model, kwargs
        source = inputs.clone().requires_grad_(True)
        return source * 2


class _TupleExplainer(AttributionOnlyExplainer):
    algorithm = "Saliency"

    def compute_attributions(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        del model, kwargs
        return inputs, torch.zeros(inputs.shape[0])


class _ListExplainer(AttributionOnlyExplainer):
    algorithm = "Saliency"

    def compute_attributions(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        del model, kwargs
        return [inputs]


def _raitap_kwargs_for(inputs: torch.Tensor, **overrides: Any) -> dict[str, Any]:
    if inputs.ndim == 4:
        input_metadata = {"kind": "image", "shape": tuple(inputs.shape), "layout": "NCHW"}
    else:
        input_metadata = {"kind": "tabular", "shape": tuple(inputs.shape), "layout": "(B,F)"}
    return {"input_metadata": input_metadata, **overrides}


def test_explain_rejects_unknown_method_family_before_compute() -> None:
    explainer = _UnknownAlgorithmExplainer()
    inputs = torch.randn(2, 3, 4, 4)

    with pytest.raises(
        Exception,
        match=(
            "method-family inference is not implemented for framework "
            "<unknown> and algorithm UnregisteredAlgorithm"
        ),
    ):
        explainer.explain(
            torch.nn.Identity(),
            inputs,
            raitap_kwargs=_raitap_kwargs_for(inputs),
        )

    assert explainer.compute_called is False


def test_explain_stores_raitap_visualisation_metadata_in_kwargs() -> None:
    explainer = _StrictExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(2, 3, 4, 4)

    result = explainer.explain(
        model,
        inputs,
        target=7,
        raitap_kwargs=_raitap_kwargs_for(
            inputs,
            sample_names=["ISIC_1", "ISIC_2"],
            show_sample_names=True,
        ),
    )

    assert torch.equal(result.attributions, inputs)
    assert explainer.last_target == 7
    # Rendering/propagation behavior is covered at the ExplanationResult.visualise layer.
    assert result.call_kwargs["target"] == 7
    assert result.kwargs["sample_names"] == ["ISIC_1", "ISIC_2"]
    assert result.kwargs["show_sample_names"] is True


def test_explain_builds_semantics_with_sample_ids_separate_from_display_names() -> None:
    explainer = _StrictExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(2, 3, 4, 4)

    result = explainer.explain(
        model,
        inputs,
        target=7,
        raitap_kwargs={
            "input_metadata": {"kind": "image", "shape": inputs.shape, "layout": "NCHW"},
            "sample_ids": ["stable-1", "stable-2"],
            "sample_names": ["Display 1", "Display 2"],
            "show_sample_names": True,
        },
    )

    assert result.semantics.scope is ExplanationScope.LOCAL
    assert result.semantics.scope_definition_step is ScopeDefinitionStep.EXPLAINER_OUTPUT
    assert result.semantics.payload_kind is ExplanationPayloadKind.ATTRIBUTIONS
    assert result.semantics.method_families == frozenset({MethodFamily.GRADIENT})
    assert result.semantics.sample_selection is not None
    assert result.semantics.sample_selection.sample_ids == ["stable-1", "stable-2"]
    assert result.semantics.sample_selection.sample_display_names == ["Display 1", "Display 2"]
    assert result.semantics.input_spec is not None
    assert result.semantics.input_spec.kind == "image"
    assert result.semantics.input_spec.layout == "NCHW"
    assert result.semantics.output_space.space is ExplanationOutputSpace.INPUT_FEATURES
    assert result.kwargs["sample_names"] == ["Display 1", "Display 2"]


def test_explain_uses_declared_output_scope() -> None:
    explainer = _GlobalScopeExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(2, 3, 4, 4)

    result = explainer.explain(
        model,
        inputs,
        raitap_kwargs=_raitap_kwargs_for(inputs),
    )

    assert result.semantics.scope is ExplanationScope.GLOBAL


def test_explain_uses_optional_batching_and_slices_per_sample_kwargs() -> None:
    explainer = _BatchRecordingExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(5, 3)
    background = torch.randn(2, 3)

    result = explainer.explain(
        model,
        inputs,
        target=[0, 1, 2, 3, 4],
        background_data=background,
        raitap_kwargs=_raitap_kwargs_for(inputs, batch_size=2),
    )

    assert torch.equal(result.attributions, inputs)
    assert explainer.seen_targets == [[0, 1], [2, 3], [4]]
    assert explainer.seen_background_sizes == [2, 2, 2]


def test_explain_prepares_each_batch_with_backend() -> None:
    class _PreparingBackend:
        def __init__(self) -> None:
            self.prepared_batch_sizes: list[int] = []

        def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
            self.prepared_batch_sizes.append(int(inputs.shape[0]))
            return inputs + 10

    explainer = _BatchRecordingExplainer()
    backend = _PreparingBackend()
    model = torch.nn.Identity()
    inputs = torch.randn(5, 3)

    result = explainer.explain(
        model,
        inputs,
        backend=backend,
        raitap_kwargs=_raitap_kwargs_for(inputs, batch_size=2),
    )

    assert backend.prepared_batch_sizes == [2, 2, 1]
    assert torch.equal(result.inputs, inputs)
    assert torch.equal(result.attributions, inputs + 10)


def test_explain_accepts_progress_kwargs_without_forwarding_to_compute() -> None:
    explainer = _StrictExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(3, 3)

    result = explainer.explain(
        model,
        inputs,
        target=7,
        raitap_kwargs=_raitap_kwargs_for(
            inputs,
            batch_size=1,
            show_progress=True,
            progress_desc="SHAP smoke",
        ),
    )

    assert torch.equal(result.attributions, inputs)
    assert explainer.last_target == 7


def test_explain_enables_progress_wrapping_by_default_for_batched_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explainer = _BatchRecordingExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(5, 3)
    wrapped: dict[str, Any] = {}

    def _record_wrap(starts: range, *, total_batches: int, progress_desc: str | None) -> range:
        wrapped["starts"] = starts
        wrapped["total_batches"] = total_batches
        wrapped["progress_desc"] = progress_desc
        return starts

    monkeypatch.setattr(explainer, "_wrap_with_progress", _record_wrap)

    result = explainer.explain(
        model,
        inputs,
        target=[0, 1, 2, 3, 4],
        raitap_kwargs=_raitap_kwargs_for(inputs, batch_size=2),
    )

    assert torch.equal(result.attributions, inputs)
    assert wrapped["starts"] == range(0, 5, 2)
    assert wrapped["total_batches"] == 3
    assert wrapped["progress_desc"] is None


def test_explain_allows_disabling_progress_wrapping_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explainer = _BatchRecordingExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(5, 3)
    wrap_calls = 0

    def _record_wrap(starts: range, *, total_batches: int, progress_desc: str | None) -> range:
        del starts, total_batches, progress_desc
        nonlocal wrap_calls
        wrap_calls += 1
        return range(0)

    monkeypatch.setattr(explainer, "_wrap_with_progress", _record_wrap)

    result = explainer.explain(
        model,
        inputs,
        target=[0, 1, 2, 3, 4],
        raitap_kwargs=_raitap_kwargs_for(inputs, batch_size=2, show_progress=False),
    )

    assert torch.equal(result.attributions, inputs)
    assert wrap_calls == 0


def test_explain_flushes_cuda_cache_between_outer_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explainer = _BatchRecordingExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(5, 3)
    gc_collect_calls = 0
    empty_cache_calls = 0

    def _record_gc_collect() -> int:
        nonlocal gc_collect_calls
        gc_collect_calls += 1
        return 0

    def _record_empty_cache() -> None:
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    monkeypatch.setattr(base_explainer_module.gc, "collect", _record_gc_collect)
    monkeypatch.setattr(base_explainer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(base_explainer_module.torch.cuda, "empty_cache", _record_empty_cache)

    result = explainer.explain(
        model,
        inputs,
        target=[0, 1, 2, 3, 4],
        raitap_kwargs=_raitap_kwargs_for(inputs, batch_size=2, show_progress=False),
    )

    assert torch.equal(result.attributions, inputs)
    assert gc_collect_calls == 3
    assert empty_cache_calls == 3


def test_explain_detaches_batched_chunks_before_concatenation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explainer = _GradTrackingExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(5, 3)
    observed_chunks: list[torch.Tensor] = []
    original_cat = base_explainer_module.torch.cat

    def _record_cat(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
        observed_chunks.extend(tensors)
        return original_cat(tensors, dim=dim)

    monkeypatch.setattr(base_explainer_module.torch, "cat", _record_cat)

    result = explainer.explain(
        model,
        inputs,
        raitap_kwargs=_raitap_kwargs_for(inputs, batch_size=2, show_progress=False),
    )

    assert result.attributions.device.type == "cpu"
    assert not result.attributions.requires_grad
    assert len(observed_chunks) == 3
    assert all(chunk.device.type == "cpu" for chunk in observed_chunks)
    assert all(not chunk.requires_grad for chunk in observed_chunks)


def test_explain_normalises_unbatched_attributions_to_detached_cpu() -> None:
    explainer = _GradTrackingExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(2, 3)

    result = explainer.explain(model, inputs, raitap_kwargs=_raitap_kwargs_for(inputs))

    assert result.attributions.device.type == "cpu"
    assert not result.attributions.requires_grad
    assert torch.equal(result.attributions, inputs * 2)


def test_explain_rejects_tuple_attribution_outputs_before_normalisation() -> None:
    explainer = _TupleExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(2, 3)

    with pytest.raises(
        TypeError,
        match=r"tuple/list attribution outputs.*convergence deltas.*not first-class payloads",
    ):
        explainer.explain(
            model,
            inputs,
            raitap_kwargs=_raitap_kwargs_for(inputs),
        )


def test_explain_rejects_list_attribution_outputs_in_batched_path() -> None:
    explainer = _ListExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(4, 3)

    with pytest.raises(
        TypeError,
        match=r"tuple/list attribution outputs.*convergence deltas.*not first-class payloads",
    ):
        explainer.explain(
            model,
            inputs,
            raitap_kwargs={
                **_raitap_kwargs_for(inputs, batch_size=2, show_progress=False),
            },
        )


def test_explain_rejects_invalid_progress_kwarg_types() -> None:
    explainer = _StrictExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(2, 3)

    with pytest.raises(TypeError, match="show_progress must be a bool"):
        explainer.explain(
            model,
            inputs,
            raitap_kwargs=_raitap_kwargs_for(inputs, show_progress="yes"),
        )

    with pytest.raises(TypeError, match="progress_desc must be a str"):
        explainer.explain(
            model,
            inputs,
            raitap_kwargs=_raitap_kwargs_for(
                inputs,
                batch_size=1,
                show_progress=True,
                progress_desc=123,
            ),
        )


def test_explain_rejects_invalid_batch_size_kwargs() -> None:
    explainer = _StrictExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(2, 3)

    with pytest.raises(TypeError, match="batch_size must be an int"):
        explainer.explain(model, inputs, raitap_kwargs=_raitap_kwargs_for(inputs, batch_size="2"))

    with pytest.raises(ValueError, match="batch_size must be > 0"):
        explainer.explain(model, inputs, raitap_kwargs=_raitap_kwargs_for(inputs, batch_size=0))


def test_explain_threads_backend_task_kind_into_infer_output_space() -> None:
    """When backend.task_kind is detection, the inferred output space is
    DETECTION_BOXES (DETECTION branch in infer_output_space)."""
    from raitap.models.backend import TorchBackend
    from raitap.transparency.contracts import ExplanationOutputSpace
    from raitap.types import TaskKind

    explainer = _StrictExplainer()
    inputs = torch.zeros(1, 3, 4, 4)
    model = torch.nn.Identity()
    backend = TorchBackend(model, task_kind=TaskKind.detection)

    result = explainer.explain(
        model,
        inputs,
        backend=backend,
        raitap_kwargs=_raitap_kwargs_for(inputs),
    )
    assert result.semantics.output_space.space is ExplanationOutputSpace.DETECTION_BOXES


def test_explain_classification_unchanged_when_backend_has_no_task_kind() -> None:
    """No backend → task_kind defaults to None → classification path stays
    untouched (INPUT_FEATURES for an NCHW image input)."""
    from raitap.transparency.contracts import ExplanationOutputSpace

    explainer = _StrictExplainer()
    inputs = torch.zeros(1, 3, 4, 4)
    model = torch.nn.Identity()

    result = explainer.explain(
        model,
        inputs,
        raitap_kwargs=_raitap_kwargs_for(inputs),
    )
    assert result.semantics.output_space.space is ExplanationOutputSpace.INPUT_FEATURES


def test_explainer_baseline_declarations() -> None:
    from raitap.transparency.explainers.base_explainer import BaseExplainer
    from raitap.transparency.explainers.captum_explainer import CaptumExplainer
    from raitap.transparency.explainers.shap_explainer import ShapExplainer

    # Base default: no baseline kwarg, no implicit defaults.
    assert BaseExplainer.baseline_kwarg is None
    assert BaseExplainer.baseline_defaults == {}

    # Captum: only IntegratedGradients has a meaningful zero default.
    assert CaptumExplainer.baseline_kwarg == "baselines"
    assert CaptumExplainer.baseline_defaults == {"IntegratedGradients": "zero"}
    assert "Saliency" not in CaptumExplainer.baseline_defaults

    # SHAP: Gradient/Deep/Kernel fall back to the input batch; Tree does not.
    assert ShapExplainer.baseline_kwarg == "background_data"
    assert ShapExplainer.baseline_defaults == {
        "GradientExplainer": "input_batch",
        "DeepExplainer": "input_batch",
        "KernelExplainer": "input_batch",
    }
    assert "TreeExplainer" not in ShapExplainer.baseline_defaults


def test_explain_attaches_zero_baseline_when_absent(tmp_path: Path) -> None:
    explainer = _BaselineDeclaringExplainer()
    inputs = torch.randn(2, 3, 4, 4)

    result = explainer.explain(
        torch.nn.Identity(),
        inputs,
        run_dir=tmp_path / "exp",
        target=0,
        raitap_kwargs=_raitap_kwargs_for(inputs),
    )

    assert result.baseline is not None
    assert result.baseline.mode == "zero"
    assert result.baseline.kwarg_name == "baselines"
    assert result.baseline.shape == (1, 3, 4, 4)  # torch.zeros_like(inputs[:1])
    assert result.baseline.sha256
    # Image modality -> a baseline PNG was rendered into run_dir.
    assert result.baseline.image_path is not None
    assert (result.run_dir / result.baseline.image_path).exists()


def test_explain_survives_baseline_documentation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(**_kwargs: Any) -> None:
        raise RuntimeError("baseline render exploded")

    monkeypatch.setattr(base_explainer_module, "build_baseline_record", _boom)

    explainer = _BaselineDeclaringExplainer()
    inputs = torch.randn(2, 3, 4, 4)

    result = explainer.explain(
        torch.nn.Identity(),
        inputs,
        run_dir=tmp_path / "exp",
        target=0,
        raitap_kwargs=_raitap_kwargs_for(inputs),
    )

    # Documentation failure must not discard the scientific output.
    assert result.baseline is None
    assert torch.equal(result.attributions, inputs)
    assert (result.run_dir / "metadata.json").exists()
    assert (result.run_dir / "attributions.pt").exists()


def test_explain_records_configured_baseline_with_provenance(tmp_path: Path) -> None:
    explainer = _BaselineDeclaringExplainer()
    inputs = torch.randn(1, 3, 4, 4)
    baseline_tensor = torch.zeros(1, 3, 4, 4)

    result = explainer.explain(
        torch.nn.Identity(),
        inputs,
        run_dir=tmp_path / "exp",
        target=0,
        baselines=baseline_tensor,
        call_provenance={"baselines": {"source": "zeros_cfg", "n_samples": 1}},
        raitap_kwargs=_raitap_kwargs_for(inputs),
    )

    assert result.baseline is not None
    assert result.baseline.mode == "configured"
    assert result.baseline.source == "zeros_cfg"
    assert result.baseline.n_samples == 1
    # call_provenance is a named kwarg, not forwarded into compute_attributions
    # kwargs, so it never lands in call_kwargs.
    assert "call_provenance" not in result.call_kwargs
