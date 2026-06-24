"""Detection task family.

Per-box explanation semantics (K boxes per sample) live here. Robustness is a
Phase 4 deliverable, so ``supports_robustness`` is ``False``; detection models
normalise internally, so ``allows_preprocessing`` is ``False``. Heavy members
are added in the phase-migration tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from raitap.data.data import _normalise_sample_id, _resolve_id_strategy
from raitap.task_families.registry import task_family
from raitap.transparency.contracts import ExplanationOutputSpace
from raitap.types import TaskKind

if TYPE_CHECKING:
    import torch

    from raitap.models.torch_backend import TorchBackend
    from raitap.task_families.base import ExplainContext, ForwardContext


def _align_detection_records(
    records: list[dict[str, Any]],
    *,
    expected: int,
    sample_ids: Any,
    strategy: str = "auto",
) -> list[dict[str, torch.Tensor]]:
    """Align native detection records to ``sample_ids`` and build tensors.

    Extracted from ``DetectionFamily.load_labels`` so label-format adapters can
    feed converted records through the same alignment + validation path.

    When ``sample_ids`` is provided, both the discovered ids and record
    ``sample_id`` fields are normalised via ``_normalise_sample_id`` using the
    resolved ``strategy``, matching how the classification path handles nested
    image directories.
    """
    import pandas as pd
    import torch

    if sample_ids is not None:
        # Collect raw record ids first so _resolve_id_strategy can inspect them.
        raw_record_ids: list[str] = []
        for index, record in enumerate(records):
            record_id = record.get("sample_id") if isinstance(record, dict) else None
            if record_id is None:
                raise ValueError(
                    f"Detection labels record {index} is missing 'sample_id' "
                    "(required when the dataset exposes sample_ids)."
                )
            raw_record_ids.append(str(record_id))

        resolved = _resolve_id_strategy(strategy, pd.Series(raw_record_ids))

        by_id: dict[str, dict[str, Any]] = {}
        for record, record_id in zip(records, raw_record_ids, strict=True):
            norm_id = _normalise_sample_id(record_id, resolved)
            if norm_id in by_id:
                raise ValueError(
                    f"Detection labels file contains duplicate sample_id {record_id!r}."
                )
            by_id[norm_id] = record

        ordered_records = []
        missing: list[str] = []
        for sample_id in sample_ids:
            norm_sid = _normalise_sample_id(sample_id, resolved)
            record = by_id.get(norm_sid)
            if record is None:
                missing.append(sample_id)
            else:
                ordered_records.append(record)
        if missing:
            raise ValueError(
                f"Detection labels file is missing entries for sample_ids: {missing!r}."
            )
        records_iter: list[dict[str, Any]] = ordered_records
    else:
        if len(records) != expected:
            raise ValueError(
                f"Detection labels file has {len(records)} records but the "
                f"dataset has {expected} samples; provide sample_id fields and "
                "set data.labels.source so records can be aligned by id, or "
                "match the record count to the sample count."
            )
        records_iter = records

    out: list[dict[str, torch.Tensor]] = []
    for index, record in enumerate(records_iter):
        boxes_raw = record.get("boxes", [])
        labels_raw = record.get("labels", [])
        if len(boxes_raw) != len(labels_raw):
            raise ValueError(
                f"Sample index {index}: boxes and labels must have matching "
                f"length (got {len(boxes_raw)} boxes vs {len(labels_raw)} labels)."
            )
        boxes_tensor = (
            torch.tensor(boxes_raw, dtype=torch.float32)
            if boxes_raw
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels_tensor = (
            torch.tensor(labels_raw, dtype=torch.int64)
            if labels_raw
            else torch.zeros((0,), dtype=torch.int64)
        )
        if boxes_tensor.ndim != 2 or boxes_tensor.shape[1] != 4:
            raise ValueError(
                f"Sample index {index}: boxes must be shape (M_i, 4); got "
                f"{tuple(boxes_tensor.shape)}."
            )
        out.append({"boxes": boxes_tensor, "labels": labels_tensor})

    if len(out) != expected:
        raise ValueError(
            f"Detection labels alignment produced {len(out)} entries but the "
            f"dataset has {expected} samples."
        )
    return out


@task_family
class DetectionFamily:
    kind: TaskKind = TaskKind.detection
    fixed_output_space: ExplanationOutputSpace | None = ExplanationOutputSpace.DETECTION_BOXES
    # Robustness is a Phase 4 deliverable; detection models normalise internally.
    supports_robustness: bool = False
    allows_preprocessing: bool = False

    def matches_model(self, model: Any) -> bool:
        from raitap.models.torch_backend import _is_torchvision_detection_model

        return _is_torchvision_detection_model(model)

    def validate_payload(self, payload: object) -> None:
        if not isinstance(payload, list) or not all(isinstance(p, dict) for p in payload):
            raise ValueError("ForwardOutput(task_kind=detection) requires a list[dict] payload.")

    def prediction_summaries(
        self, payload: Any, *, sample_ids: Any = None, targets: Any = None, output_kind: Any = None
    ) -> list | None:
        # Detection has no per-sample "predicted class + confidence" concept.
        return None

    def adapt_loaded_inputs(self, tensor: Any) -> Any:
        # Detection wants a ragged list of per-image (C, H, W) tensors. A dense
        # tensor (uniform-size dir / single file / demo sample) is unbound;
        # ragged dirs already arrive as a list from the loader and pass through.
        if isinstance(tensor, list):
            return tensor
        return list(tensor.unbind(0))

    def validate_inputs(self, tensor: Any) -> None:
        import torch

        if not isinstance(tensor, list):
            raise TypeError(
                "Detection data must be a list of per-image (C, H, W) tensors, "
                f"got {type(tensor).__name__}."
            )
        if not tensor:
            raise ValueError("Detection data is empty; loaded zero images.")
        for index, image in enumerate(tensor):
            if not isinstance(image, torch.Tensor) or image.ndim != 3:
                shape = tuple(image.shape) if isinstance(image, torch.Tensor) else None
                raise ValueError(
                    "Detection data entries must be (C, H, W) tensors; entry "
                    f"{index} is {type(image).__name__}"
                    + (f" with shape {shape}." if shape is not None else ".")
                )

    def validate_labels(self, labels: Any) -> None:
        # The detection loader returns ``list[dict]`` or ``None``. A bare tensor
        # is a classification-shaped label set; disagreement means model and
        # data declare different task families.
        if labels is not None and not isinstance(labels, list):
            raise ValueError(
                "detection model loaded classification-shaped labels; model and "
                "data disagree. Set model.task_kind to match your data, or point "
                "data.labels.source at detection labels (JSON list of records)."
            )

    def extract_forward(self, ctx: ForwardContext, *, batch_size: int) -> list[dict]:
        import torch

        # Detection is torchvision-bound (see ``matches_model``); the per-batch
        # collation helper lives only on ``TorchBackend``, not the ABC.
        backend = cast("TorchBackend", ctx.backend)
        inputs = ctx.inputs
        total_batch = len(inputs)
        detection_predictions: list[dict] = []
        for start in range(0, total_batch, batch_size):
            end = min(start + batch_size, total_batch)
            prepared = backend.prepare_detection_inputs(inputs[start:end])
            raw = backend(prepared)
            if not isinstance(raw, list):
                raise TypeError(
                    "forward_pass(detection) expected list[dict] from backend; "
                    f"got {type(raw).__name__}."
                )
            for sample_dict in raw:
                if not isinstance(sample_dict, dict):
                    raise TypeError(
                        "forward_pass(detection) expected each backend output entry to be a "
                        f"dict of tensors; got {type(sample_dict).__name__}."
                    )
                detection_predictions.append({k: v.detach().cpu() for k, v in sample_dict.items()})
            del prepared, raw
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return detection_predictions

    def payload_batch_size(self, payload: Any) -> int:
        return len(payload)

    def explain(self, ctx: ExplainContext) -> list:
        """Yield one ``ExplanationResult`` per detected box (K-loop).

        Resolves the id->name table once (explicit config > backend), drives
        :func:`explain_detection` over the pre-computed predictions, and enriches
        each raw box with the predicted name + (optional) ground-truth match
        before populating its report visualisations.
        """
        from raitap import raitap_log
        from raitap.transparency.detection_labels import (
            enrich_detection_box,
            resolve_category_names,
        )
        from raitap.transparency.explain_detection import (
            _DEFAULT_IOU_THRESHOLD,
            explain_detection,
        )

        prepared = ctx.prepared
        backend = prepared.backend

        category_names = resolve_category_names(
            prepared.class_names,
            backend.category_names,
        )
        data_labels = getattr(ctx.data, "labels", None)
        detection_ground_truth = data_labels if isinstance(data_labels, list) else None

        raitap_cfg = prepared.raitap_kwargs
        ground_truth_iou_threshold = float(
            raitap_cfg.get("detection", {}).get("iou_threshold", _DEFAULT_IOU_THRESHOLD)
        )

        explanations: list = []
        for result in explain_detection(
            inputs=ctx.data.tensor,
            forward_output=ctx.forward_output,
            backend=backend,
            explainer=prepared.explainer,
            explainer_target=prepared.explainer_target,
            explainer_name=prepared.name,
            visualisers=prepared.visualisers,
            base_run_dir=prepared.base_run_dir,
            raitap_kwargs=raitap_cfg,
            call_kwargs=prepared.merged_kwargs,
            call_provenance=prepared.call_provenance,
        ):
            if result.detection_box is not None:
                sample_index = result.original_sample_index
                ground_truth_for_sample = None
                if detection_ground_truth is not None and sample_index is not None:
                    if sample_index < len(detection_ground_truth):
                        ground_truth_for_sample = detection_ground_truth[sample_index]
                    else:
                        # Loader guarantees len(detection_ground_truth) == n_samples, so this
                        # only fires on a genuine prediction/label misalignment;
                        # surface it instead of silently skipping the GT match.
                        raitap_log.warn(
                            "Detection ground truth has %d entries but an explanation "
                            "references sample_index=%d; rendering this box without a "
                            "true label (prediction/label misalignment).",
                            len(detection_ground_truth),
                            sample_index,
                        )
                result.detection_box = enrich_detection_box(
                    result.detection_box,
                    category_names=category_names,
                    ground_truth_for_sample=ground_truth_for_sample,
                    iou_threshold=ground_truth_iou_threshold,
                )
            result._visualise()  # populates result.visualisations
            explanations.append(result)
        return explanations

    def metrics_inputs(self, config: Any, forward_output: Any, labels: Any) -> Any:
        from raitap import raitap_log

        if labels is None:
            raitap_log.warn(
                "Detection metrics require dataset labels "
                "(list[dict] detection targets); none provided. "
                "Skipping metrics."
            )
            return None
        if not isinstance(labels, list):
            raitap_log.warn(
                "Detection metrics require list[dict] targets; got "
                f"{type(labels).__name__}. Skipping metrics."
            )
            return None
        return forward_output.as_detection(), labels
