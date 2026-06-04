"""Detection-task transparency phase — one ExplanationResult per detected box.

Reads pre-computed detection predictions from :class:`ForwardOutput` (D24, no
second forward pass), filters per sample by ``score_threshold`` then top
``max_boxes``, and calls the explainer K times per sample with a faithful
``DetectionTarget(mode="reference_match", ...)`` anchored to that box's xyxy
+ label. Issue #146 Phase 3 (D10 / D13 / D21 / D24, A1 raw-index fix).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from raitap import raitap_log
from raitap.models.task_wrappers import DetectionTarget, ScalarDetectionWrapper
from raitap.transparency.contracts import DetectionBox
from raitap.types import DetectionInputs, TaskKind
from raitap.utils.errors import RaitapError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from raitap.models.backend import ModelBackend
    from raitap.pipeline.outputs import ForwardOutput
    from raitap.transparency.results import ConfiguredVisualiser, ExplanationResult


_DEFAULT_SCORE_THRESHOLD = 0.5
_DEFAULT_MAX_BOXES = 5
_DEFAULT_IOU_THRESHOLD = 0.5


def _sample_as_batch(inputs: torch.Tensor | DetectionInputs, index: int) -> torch.Tensor:
    """Return a (1, C, H, W) tensor for sample *index* from either input form.

    - ``list[Tensor]`` (ragged, one ``(C,H,W)`` per image): unsqueeze at dim 0.
    - Dense ``(N,C,H,W)`` tensor: standard one-element slice preserving the
      batch dimension.
    """
    if isinstance(inputs, list):
        return inputs[index].unsqueeze(0)
    return inputs[index : index + 1]


def explain_detection(
    *,
    inputs: torch.Tensor | DetectionInputs,
    forward_output: ForwardOutput,
    backend: ModelBackend,
    explainer: Any,
    explainer_target: str,
    explainer_name: str,
    visualisers: Sequence[ConfiguredVisualiser],
    base_run_dir: Path,
    raitap_kwargs: dict[str, Any] | None,
    call_kwargs: dict[str, Any],
    call_provenance: Mapping[str, Mapping[str, Any]] | None = None,
) -> Iterator[ExplanationResult]:
    """Yield one ExplanationResult per detected box.

    Per sample i:
    - Read pre-computed predictions from ``forward_output.as_detection()[i]``.
    - Filter by ``score_threshold`` then keep top ``max_boxes`` by score
      (using raw-index masking so resulting indices reference the original
      detector output).
    - For each kept box, wrap the model with a ``ScalarDetectionWrapper`` +
      ``DetectionTarget(mode="reference_match", ...)`` anchored to that box.
    - Call ``explainer.explain(...)`` with a per-box ``run_dir``
      (``<base>/sample_{i}/box_{raw_index}/``) and ``target=0`` forced
      (the wrapper exposes one scalar channel; rejects ``auto_pred``).
    - Attach the ``DetectionBox`` + ``original_sample_index`` to the result
      before yielding it.

    Samples with no boxes passing the threshold are skipped with a warn log.
    """
    if forward_output.task_kind is not TaskKind.detection:
        raise RaitapError(
            "explain_detection invoked on a non-detection ForwardOutput "
            f"(task_kind={forward_output.task_kind!r})."
        )

    detection_predictions = forward_output.as_detection()

    detection_cfg = (raitap_kwargs or {}).get("detection", {})
    score_threshold = float(detection_cfg.get("score_threshold", _DEFAULT_SCORE_THRESHOLD))
    max_boxes = int(detection_cfg.get("max_boxes", _DEFAULT_MAX_BOXES))
    iou_threshold = float(detection_cfg.get("iou_threshold", _DEFAULT_IOU_THRESHOLD))

    if max_boxes < 1:
        raise RaitapError(f"raitap.detection.max_boxes must be >= 1; got {max_boxes!r}.")
    if not 0.0 <= iou_threshold <= 1.0:
        raise RaitapError(
            f"raitap.detection.iou_threshold must lie in [0, 1]; got {iou_threshold!r}."
        )

    # D21 — target normalisation: wrapper exposes one scalar channel.
    requested_target = call_kwargs.get("target")
    if requested_target == "auto_pred":
        raise RaitapError(
            "config.transparency.<explainer>.call.target=auto_pred is not supported "
            "for detection tasks: the ScalarDetectionWrapper exposes a single scalar "
            "channel, so argmax over it always returns 0. Set call.target=0 explicitly."
        )
    if requested_target is not None and requested_target != 0:
        raitap_log.warn(
            f"Overriding call.target={requested_target!r} to 0 for detection task "
            "(wrapper exposes a single scalar channel)."
        )
    normalised_call_kwargs = dict(call_kwargs)
    normalised_call_kwargs["target"] = 0

    base_model = backend.as_model_for_explanation()

    # One baseline preview is shared by every box of a sample (same inputs +
    # call kwargs). Render it once and copy for the rest, keyed by content hash.
    baseline_render_cache: dict[str, Path] = {}

    for sample_index, predictions_i in enumerate(detection_predictions):
        scores = predictions_i.get("scores", torch.zeros(0))
        boxes = predictions_i.get("boxes", torch.zeros((0, 4)))
        labels = predictions_i.get("labels", torch.zeros(0, dtype=torch.int64))

        if scores.numel() == 0:
            raitap_log.warn(
                f"sample_index={sample_index}: 0 detections from forward pass; "
                "emitting no detection explanations."
            )
            continue

        # A1 — raw-index correct filter: nonzero(mask) gives positions in the
        # ORIGINAL scores tensor, not in the masked subset.
        mask = scores >= score_threshold
        raw_candidates = torch.nonzero(mask, as_tuple=False).flatten()
        if raw_candidates.numel() == 0:
            raitap_log.warn(
                f"sample_index={sample_index}: 0 boxes passed "
                f"score_threshold={score_threshold!r}; "
                f"max_score={float(scores.max())!r}; "
                "emitting no detection explanations."
            )
            continue

        order = scores[raw_candidates].argsort(descending=True)
        top_k_raw_indices = raw_candidates[order[:max_boxes]]

        sample_inputs = _sample_as_batch(inputs, sample_index)

        for display_index, raw_index_value in enumerate(top_k_raw_indices.tolist()):
            raw_index = int(raw_index_value)
            reference_xyxy_tensor = boxes[raw_index]
            reference_xyxy = tuple(float(v) for v in reference_xyxy_tensor.tolist())
            assert len(reference_xyxy) == 4
            reference_label = int(labels[raw_index].item())
            score = float(scores[raw_index].item())

            target = DetectionTarget(
                mode="reference_match",
                reference_xyxy=reference_xyxy,  # type: ignore[arg-type]
                reference_label=reference_label,
                iou_threshold=iou_threshold,
            )
            wrapped = ScalarDetectionWrapper(base_model, target=target)

            per_box_run_dir = base_run_dir / f"sample_{sample_index}" / f"box_{raw_index}"
            per_box_run_dir.mkdir(parents=True, exist_ok=True)

            # Narrow raitap_kwargs to this single sample — the explainer sees
            # attributions of shape ``(1, ...)``, so ``sample_names`` /
            # ``sample_ids`` must also be length 1 (else SampleNamesLengthError).
            per_box_raitap: dict[str, Any] = dict(raitap_kwargs or {})
            sample_names = per_box_raitap.get("sample_names")
            if sample_names is not None:
                per_box_raitap["sample_names"] = [sample_names[sample_index]]
            sample_ids = per_box_raitap.get("sample_ids")
            if sample_ids is not None:
                per_box_raitap["sample_ids"] = [sample_ids[sample_index]]

            result = explainer.explain(
                wrapped,
                sample_inputs,
                backend=backend,
                run_dir=per_box_run_dir,
                explainer_target=explainer_target,
                explainer_name=explainer_name,
                visualisers=list(visualisers),
                raitap_kwargs=per_box_raitap,
                call_provenance=call_provenance,
                baseline_render_cache=baseline_render_cache,
                **normalised_call_kwargs,
            )
            result.detection_box = DetectionBox(
                display_index=display_index,
                raw_index=raw_index,
                xyxy=reference_xyxy,  # type: ignore[arg-type]
                score=score,
                label_index=reference_label,
                label_name=None,
            )
            result.original_sample_index = sample_index

            yield result
