from __future__ import annotations

from typing import Any

import pytest
import torch

import raitap.transparency.explainers.base_explainer as base_explainer_module
from raitap.transparency.explainers.base_explainer import AttributionOnlyExplainer


class _StrictExplainer(AttributionOnlyExplainer):
    algorithm = "StrictAlgo"

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


class _BatchRecordingExplainer(AttributionOnlyExplainer):
    algorithm = "BatchRecorder"

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
        if isinstance(target, torch.Tensor):
            self.seen_targets.append(target.tolist())
        elif isinstance(target, list):
            self.seen_targets.append(target)
        else:
            self.seen_targets.append([])

        background_size = -1 if background_data is None else int(background_data.shape[0])
        self.seen_background_sizes.append(background_size)
        return inputs


class _GradTrackingExplainer(AttributionOnlyExplainer):
    algorithm = "GradTracker"

    def compute_attributions(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del model, kwargs
        source = inputs.clone().requires_grad_(True)
        return source * 2


def test_explain_stores_raitap_visualisation_metadata_in_kwargs() -> None:
    explainer = _StrictExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(2, 3, 4, 4)

    result = explainer.explain(
        model,
        inputs,
        target=7,
        raitap_kwargs={
            "sample_names": ["ISIC_1", "ISIC_2"],
            "show_sample_names": True,
        },
    )

    assert torch.equal(result.attributions, inputs)
    assert explainer.last_target == 7
    # Rendering/propagation behavior is covered at the ExplanationResult.visualise layer.
    assert result.kwargs["target"] == 7
    assert result.kwargs["sample_names"] == ["ISIC_1", "ISIC_2"]
    assert result.kwargs["show_sample_names"] is True


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
        raitap_kwargs={"batch_size": 2},
    )

    assert torch.equal(result.attributions, inputs)
    assert explainer.seen_targets == [[0, 1], [2, 3], [4]]
    assert explainer.seen_background_sizes == [2, 2, 2]


def test_explain_accepts_progress_kwargs_without_forwarding_to_compute() -> None:
    explainer = _StrictExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(3, 3)

    result = explainer.explain(
        model,
        inputs,
        target=7,
        raitap_kwargs={
            "batch_size": 1,
            "show_progress": True,
            "progress_desc": "SHAP smoke",
        },
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
        raitap_kwargs={"batch_size": 2},
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
        raitap_kwargs={"batch_size": 2, "show_progress": False},
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
        raitap_kwargs={"batch_size": 2, "show_progress": False},
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
        raitap_kwargs={"batch_size": 2, "show_progress": False},
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

    result = explainer.explain(model, inputs)

    assert result.attributions.device.type == "cpu"
    assert not result.attributions.requires_grad
    assert torch.equal(result.attributions, inputs * 2)


def test_explain_rejects_invalid_progress_kwarg_types() -> None:
    explainer = _StrictExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(2, 3)

    with pytest.raises(TypeError, match="show_progress must be a bool"):
        explainer.explain(model, inputs, raitap_kwargs={"show_progress": "yes"})

    with pytest.raises(TypeError, match="progress_desc must be a str"):
        explainer.explain(
            model,
            inputs,
            raitap_kwargs={"batch_size": 1, "show_progress": True, "progress_desc": 123},
        )


def test_explain_rejects_invalid_batch_size_kwargs() -> None:
    explainer = _StrictExplainer()
    model = torch.nn.Identity()
    inputs = torch.randn(2, 3)

    with pytest.raises(TypeError, match="batch_size must be an int"):
        explainer.explain(model, inputs, raitap_kwargs={"batch_size": "2"})

    with pytest.raises(ValueError, match="batch_size must be > 0"):
        explainer.explain(model, inputs, raitap_kwargs={"batch_size": 0})
