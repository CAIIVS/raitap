"""Classification task family.

Classification is a peer family, not the default (spec D4). The dense-NCHW
input path, the logits-tensor ``payload``, and shared-loop transparency all
live here. Heavy members (load/forward/explain/metrics) are added in the
phase-migration tasks; this file starts with the constants + validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from raitap.task_families.registry import task_family
from raitap.transparency.contracts import ExplanationOutputSpace
from raitap.types import TaskKind

if TYPE_CHECKING:
    from raitap.task_families.base import ExplainContext, ForwardContext


@task_family
class ClassificationFamily:
    kind: TaskKind = TaskKind.classification
    output_space: ExplanationOutputSpace = ExplanationOutputSpace.INPUT_FEATURES

    def validate_payload(self, payload: object) -> None:
        if not isinstance(payload, torch.Tensor):
            raise ValueError("ForwardOutput(task_kind=classification) requires a tensor payload.")

    def supports_robustness(self) -> bool:
        return True

    @property
    def allows_preprocessing(self) -> bool:
        return True

    def adapt_loaded_inputs(self, tensor: Any) -> Any:
        # Classification keeps the dense (N, C, H, W) tensor as-is.
        return tensor

    def validate_inputs(self, tensor: Any) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Classification data must be a dense (N, ...) tensor, got {type(tensor).__name__}."
            )
        if tensor.ndim < 2:
            raise ValueError(
                "Classification data must be a batched (N, ...) tensor with ndim >= 2, "
                f"got shape {tuple(tensor.shape)}."
            )
        if tensor.shape[0] < 1:
            raise ValueError("Classification data is empty; loaded zero samples.")

    def load_labels(self, cfg: Any, *, tensor: Any, sample_ids: Any) -> Any:
        from raitap.data.data import load_classification_labels

        return load_classification_labels(cfg, tensor=tensor, sample_ids=sample_ids)

    def extract_forward(self, ctx: ForwardContext, *, batch_size: int) -> torch.Tensor:
        from raitap.pipeline.phases.forward_pass import extract_primary_tensor

        backend, inputs = ctx.backend, ctx.inputs
        total_batch = len(inputs)
        if total_batch <= batch_size:
            prepared = backend._prepare_inputs(inputs)
            out = extract_primary_tensor(backend(prepared)).detach().cpu()
            del prepared
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return out
        chunks: list[torch.Tensor] = []
        for start in range(0, total_batch, batch_size):
            end = min(start + batch_size, total_batch)
            prepared = backend._prepare_inputs(inputs[start:end])
            chunks.append(extract_primary_tensor(backend(prepared)).detach().cpu())
            del prepared
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(chunks, dim=0)

    def explain(self, ctx: ExplainContext) -> list:
        """Build one ``ExplanationResult`` for the whole batch.

        Resolves the ``auto_pred`` runtime target from the classification
        logits (device-moving just that tensor, exactly as the old single
        full-dict ``_prepare_kwargs`` did), enforces the sample-names length
        invariant, then drives the explainer once over the dense input tensor.
        """
        from raitap.transparency.phase import resolve_explainer_runtime_kwargs
        from raitap.utils.errors import SampleNamesLengthError

        prepared = ctx.prepared
        inputs = ctx.data.tensor

        runtime_kwargs = resolve_explainer_runtime_kwargs(
            prepared.explainer_config,
            forward_output=ctx.forward_output.as_classification(),
        )
        if runtime_kwargs:
            # Move only the runtime target tensor — ``prepared.merged_kwargs``
            # was already prepared by ``prepare_explainer``. Each tensor is
            # device-moved exactly once, matching the pre-refactor behaviour.
            runtime_kwargs = prepared.backend._prepare_kwargs(runtime_kwargs)

        resolved_sample_names = prepared.raitap_kwargs.get("sample_names")
        if resolved_sample_names is not None:
            resolved_list = list(resolved_sample_names)
            if resolved_list and len(resolved_list) != int(inputs.shape[0]):
                raise SampleNamesLengthError(
                    got=len(resolved_list),
                    expected=int(inputs.shape[0]),
                    source="raitap.sample_names",
                )

        result = prepared.explainer.explain(
            prepared.backend.as_model_for_explanation(),
            inputs,
            backend=prepared.backend,
            run_dir=prepared.base_run_dir,
            experiment_name=prepared.experiment_name,
            explainer_target=prepared.explainer_target,
            explainer_name=prepared.name,
            visualisers=prepared.visualisers,
            raitap_kwargs=prepared.raitap_kwargs,
            call_provenance=prepared.call_provenance,
            **{**prepared.merged_kwargs, **runtime_kwargs},
        )
        result._visualise()  # populates result.visualisations (was run_adapters)
        return [result]

    def metrics_inputs(self, forward_output: Any, labels: Any) -> Any:  # Task 9
        raise NotImplementedError

    def prediction_summaries(self, payload: Any) -> list | None:  # Task 11
        raise NotImplementedError
