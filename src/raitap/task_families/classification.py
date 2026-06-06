"""Classification task family.

Classification is a peer family, not the default (spec D4). The dense-NCHW
input path, the logits-tensor ``payload``, and shared-loop transparency all
live here. Heavy members (load/forward/explain/metrics) are added in the
phase-migration tasks; this file starts with the constants + validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from raitap.task_families.registry import task_family
from raitap.types import TaskKind
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch

    from raitap.task_families.base import ExplainContext, ForwardContext
    from raitap.transparency.contracts import ExplanationOutputSpace
else:
    # Lazy so importing this module (eagerly, for @task_family registration)
    # does not require torch — the bare/no-extras CLI bootstrap imports the
    # package before installing torch.
    torch = lazy_import("torch")


@task_family
class ClassificationFamily:
    kind: TaskKind = TaskKind.classification
    # Classification output space is DYNAMIC (CAM -> IMAGE_SPATIAL_MAP, else
    # INPUT_FEATURES); ``None`` signals ``infer_output_space`` to fall through
    # to the method-family logic instead of using a fixed space.
    fixed_output_space: ExplanationOutputSpace | None = None
    supports_robustness: bool = True
    allows_preprocessing: bool = True

    def matches_model(self, model: Any) -> bool:
        # Classification is the auto-inference fallback, not architecture-detected.
        return False

    def validate_payload(self, payload: object) -> None:
        if not isinstance(payload, torch.Tensor):
            raise ValueError("ForwardOutput(task_kind=classification) requires a tensor payload.")

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

    def validate_labels(self, labels: Any) -> None:
        # A ``list[dict]`` is a detection-shaped label set; a tensor (or None)
        # is classification-shaped. Disagreement means model and data declare
        # different task families.
        if isinstance(labels, list):
            raise ValueError(
                "classification model loaded detection-shaped labels (list[dict]); "
                "model and data disagree. Set model.task_kind to match your data, "
                "or point data.labels.source at classification labels."
            )

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

    def payload_batch_size(self, payload: Any) -> int:
        return int(payload.shape[0])

    def explain(self, ctx: ExplainContext) -> list:
        """Build one ``ExplanationResult`` for the whole batch.

        Resolves the ``auto_pred`` runtime target from the classification
        logits (device-moving just that tensor, exactly as the old single
        full-dict ``_prepare_kwargs`` did), enforces the sample-names length
        invariant, then drives the explainer once over the dense input tensor.
        """
        from raitap.models.access import explanation_model
        from raitap.transparency.phase import resolve_explainer_runtime_kwargs
        from raitap.utils.errors import SampleNamesLengthError

        prepared = ctx.prepared
        # Classification data is the dense tensor (enforced by ``validate_inputs``);
        # ``Data.tensor`` is widened to ``Tensor | DetectionInputs`` for detection.
        inputs = cast("torch.Tensor", ctx.data.tensor)

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
            explanation_model(prepared.backend, prepared.explainer),
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

    def metrics_inputs(self, config: Any, forward_output: Any, labels: Any) -> Any:
        from raitap.metrics import metrics_prediction_pair, resolve_metric_targets

        predictions_tensor = forward_output.as_classification()
        if (
            getattr(config.metrics, "num_classes", None) is None
            and predictions_tensor.ndim == 2
            and predictions_tensor.shape[1] >= 2
        ):
            # ``MetricsConfig`` base only carries ``_target_``; ``num_classes``
            # lives on the multiclass typed subclass at runtime.
            config.metrics.num_classes = int(predictions_tensor.shape[1])  # type: ignore[attr-defined]
        preds, _ = metrics_prediction_pair(predictions_tensor)
        classification_labels = labels if not isinstance(labels, list) else None
        targs = resolve_metric_targets(preds, classification_labels)
        return preds, targs

    def prediction_summaries(
        self, payload: Any, *, sample_ids: Any = None, targets: Any = None
    ) -> list | None:
        from raitap.pipeline.outputs import PredictionSummary
        from raitap.pipeline.phases.prediction_summaries import valid_targets_for_reporting

        predictions_tensor = payload
        if predictions_tensor.ndim != 2 or predictions_tensor.shape[1] < 2:
            return None

        classification_targets = targets if not isinstance(targets, list) else None
        probabilities = torch.softmax(predictions_tensor.detach().cpu(), dim=1)
        confidences, predictions = probabilities.max(dim=1)
        resolved_targets = valid_targets_for_reporting(
            targets=classification_targets,
            expected=int(predictions.shape[0]),
        )

        summaries: list[PredictionSummary] = []
        names = [] if sample_ids is None else [str(item) for item in sample_ids]
        pairs = zip(predictions, confidences, strict=False)
        for index, (predicted_class, confidence) in enumerate(pairs):
            target_class: int | None = None
            correct: bool | None = None
            if resolved_targets is not None:
                target_class = int(resolved_targets[index].item())
                correct = int(predicted_class.item()) == target_class
            summaries.append(
                PredictionSummary(
                    sample_index=index,
                    sample_id=names[index] if index < len(names) else None,
                    predicted_class=int(predicted_class.item()),
                    target_class=target_class,
                    confidence=float(confidence.item()),
                    correct=correct,
                )
            )
        return summaries
