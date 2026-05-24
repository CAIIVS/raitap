"""Forward-pass phase — primary-tensor extraction + batched forward."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.pipeline.outputs import ForwardOutput
from raitap.types import DetectionInputs, TaskKind
from raitap.utils.lazy import lazy_import

# Conservative default for prediction/metrics forwards. Transparency methods
# have their own per-explainer ``transparency.*.raitap.batch_size`` controls.
_DEFAULT_FORWARD_BATCH_SIZE = 32


if TYPE_CHECKING:
    import torch

    from raitap.configs.schema import AppConfig
else:
    torch = lazy_import("torch")


def _tensor_candidates_from_dict(
    obj: dict[Any, Any],
) -> list[torch.Tensor]:
    """Return tensor values from ``obj``, preferring well-known keys first."""
    preferred_keys = ("logits", "pred", "prediction", "output", "scores")
    seen: set[int] = set()
    candidates: list[torch.Tensor] = []
    for key in preferred_keys:
        value = obj.get(key)
        if isinstance(value, torch.Tensor) and id(value) not in seen:
            candidates.append(value)
            seen.add(id(value))
    for value in obj.values():
        if isinstance(value, torch.Tensor) and id(value) not in seen:
            candidates.append(value)
            seen.add(id(value))
    return candidates


def _select_primary_from_tensor_candidates(
    candidates: list[torch.Tensor],
) -> torch.Tensor:
    """Pick the most "prediction-like" tensor from candidates."""
    batchy = [t for t in candidates if t.ndim >= 2]
    if batchy:
        return batchy[0]
    return max(candidates, key=lambda t: t.numel())


def extract_primary_tensor(model_output: object) -> torch.Tensor:
    """Reduce an arbitrary model output to its primary prediction tensor.

    Handles raw ``Tensor``, ``tuple``/``list``, and ``dict`` outputs. Prefers
    keys like ``logits`` / ``pred`` in dicts, falls back to "biggest tensor"
    when no keyed candidate fits.
    """
    if isinstance(model_output, torch.Tensor):
        return model_output

    candidates: list[torch.Tensor]
    if isinstance(model_output, (tuple, list)):
        candidates = [t for t in model_output if isinstance(t, torch.Tensor)]
    elif isinstance(model_output, dict):
        candidates = _tensor_candidates_from_dict(model_output)
    else:
        candidates = []

    if not candidates:
        raise TypeError(
            f"Cannot extract a tensor from model output of type {type(model_output).__name__}."
        )
    return _select_primary_from_tensor_candidates(candidates)


def resolve_forward_batch_size(config: AppConfig) -> int:
    """Resolve prediction/metrics forward batch size from config, falling back to 32."""
    configured = getattr(getattr(config, "run", None), "forward_batch_size", None)
    if configured is None:
        configured = getattr(getattr(config, "data", None), "forward_batch_size", None)
    if configured is None:
        return _DEFAULT_FORWARD_BATCH_SIZE
    if not isinstance(configured, int):
        raise TypeError(f"forward_batch_size must be an int, got {type(configured).__name__}.")
    if configured <= 0:
        raise ValueError(f"forward_batch_size must be > 0, got {configured}.")
    return configured


def _validate_inputs_for_task(inputs: torch.Tensor | DetectionInputs, task_kind: TaskKind) -> None:
    """Guard the input contract before ``len()``/slicing assumes it.

    ``Data`` already validates at load time; this catches direct callers that
    bypass it. Detection wants a ``list`` of per-image tensors; classification
    wants a dense ``(N, ...)`` tensor (``len()`` on a stray ``(C, H, W)`` would
    silently count channels, not samples).
    """
    if task_kind is TaskKind.detection:
        if not isinstance(inputs, list):
            raise TypeError(
                "forward_pass(detection) expected a list of per-image (C, H, W) "
                f"tensors; got {type(inputs).__name__}."
            )
        return
    if not isinstance(inputs, torch.Tensor):
        raise TypeError(
            f"forward_pass({task_kind.value}) expected a dense (N, ...) tensor; "
            f"got {type(inputs).__name__}."
        )
    if inputs.ndim < 2:
        raise ValueError(
            f"forward_pass({task_kind.value}) expected a batched (N, ...) tensor "
            f"with ndim >= 2; got shape {tuple(inputs.shape)}."
        )


def forward_pass(
    config: AppConfig, backend: Any, inputs: torch.Tensor | DetectionInputs
) -> ForwardOutput:
    """Run the model backend forward in chunks of ``forward_batch_size``.

    Returns a typed :class:`ForwardOutput` keyed by ``backend.task_kind``.
    Classification backends produce ``predictions_tensor`` (CPU-detached);
    detection backends produce ``detection_predictions`` (a length-N list of
    per-sample dicts with ``boxes`` / ``scores`` / ``labels`` tensors).

    For detection, ``inputs`` is a ragged ``list[torch.Tensor]`` with one
    native-resolution ``(C, H, W)`` tensor per image (sizes may differ, so
    they cannot be stacked). For classification, ``inputs`` is a dense
    ``(N, C, H, W)`` tensor.
    """
    batch_size = resolve_forward_batch_size(config)
    task_kind = backend.task_kind
    _validate_inputs_for_task(inputs, task_kind)
    total_batch = len(inputs)

    if task_kind is TaskKind.detection:
        detection_predictions: list[dict[str, torch.Tensor]] = []
        for start in range(0, total_batch, batch_size):
            end = min(start + batch_size, total_batch)
            prepared_inputs = backend.prepare_detection_inputs(inputs[start:end])
            raw_output: Any = backend(prepared_inputs)
            if not isinstance(raw_output, list):
                raise TypeError(
                    "forward_pass(detection) expected list[dict] from backend; "
                    f"got {type(raw_output).__name__}."
                )
            for sample_dict in raw_output:
                if not isinstance(sample_dict, dict):
                    raise TypeError(
                        "forward_pass(detection) expected each backend output entry to be a "
                        f"dict of tensors; got {type(sample_dict).__name__}."
                    )
                detection_predictions.append({k: v.detach().cpu() for k, v in sample_dict.items()})
            del prepared_inputs, raw_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return ForwardOutput(
            task_kind=TaskKind.detection,
            batch_size=len(detection_predictions),
            detection_predictions=detection_predictions,
        )

    if total_batch <= batch_size:
        prepared_inputs = backend._prepare_inputs(inputs)
        raw_output_any: Any = backend(prepared_inputs)
        predictions_tensor = extract_primary_tensor(raw_output_any).detach().cpu()
        del prepared_inputs, raw_output_any
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        chunks: list[torch.Tensor] = []
        for start in range(0, total_batch, batch_size):
            end = min(start + batch_size, total_batch)
            prepared_inputs = backend._prepare_inputs(inputs[start:end])
            raw_output_any = backend(prepared_inputs)
            chunks.append(extract_primary_tensor(raw_output_any).detach().cpu())
            del prepared_inputs, raw_output_any
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        predictions_tensor = torch.cat(chunks, dim=0)

    return ForwardOutput(
        task_kind=TaskKind.classification,
        batch_size=int(predictions_tensor.shape[0]),
        predictions_tensor=predictions_tensor,
    )
