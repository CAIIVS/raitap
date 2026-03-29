"""Pick a primary tensor from arbitrary model forward returns (tensor, tuple, dict)."""

from __future__ import annotations

from typing import Any

import torch


def _tensor_candidates_from_dict(model_output: dict[str, Any]) -> list[torch.Tensor]:
    """Ordered candidates: preferred keys first, then remaining tensor values (no duplicates)."""
    candidates: list[torch.Tensor] = []
    seen_ids: set[int] = set()
    for key in ("logits", "pred", "prediction", "output", "scores"):
        value = model_output.get(key)
        if isinstance(value, torch.Tensor):
            tid = id(value)
            if tid not in seen_ids:
                seen_ids.add(tid)
                candidates.append(value)
    for value in model_output.values():
        if isinstance(value, torch.Tensor):
            tid = id(value)
            if tid not in seen_ids:
                seen_ids.add(tid)
                candidates.append(value)
    return candidates


def _select_primary_from_tensor_candidates(candidates: list[torch.Tensor]) -> torch.Tensor:
    """
    Prefer a batch-like tensor (``ndim >= 2``), e.g. ``(loss, logits)`` → logits.

    If none qualify, use the tensor with the largest ``numel()`` (e.g. a long 1-D
    vector over a scalar loss). If still tied, ``max`` picks the first such tensor.
    """
    if not candidates:
        raise TypeError("Internal error: no tensor candidates to select from.")
    for tensor in candidates:
        if tensor.ndim >= 2:
            return tensor
    return max(candidates, key=lambda t: t.numel())


def extract_primary_tensor(model_output: object) -> torch.Tensor:
    """
    Unwrap a single tensor from common forward return shapes.

    For a raw ``Tensor``, returns it. For ``tuple`` / ``list``, collects every
    ``Tensor`` and picks the best candidate (see ``_select_primary_from_tensor_candidates``).
    For ``dict``, prefers known keys (``logits``, ``pred``, …) when building the candidate
    list, then applies the same selection.
    """
    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, (tuple, list)):
        candidates = [item for item in model_output if isinstance(item, torch.Tensor)]
        if not candidates:
            raise TypeError("Model forward returned a sequence with no torch.Tensor elements.")
        return _select_primary_from_tensor_candidates(candidates)

    if isinstance(model_output, dict):
        candidates = _tensor_candidates_from_dict(model_output)
        if not candidates:
            raise TypeError("Model forward returned a dict with no torch.Tensor values.")
        return _select_primary_from_tensor_candidates(candidates)

    raise TypeError(
        f"Unsupported model output type {type(model_output).__name__!r}; "
        "expected Tensor, sequence of Tensors, or dict containing a Tensor."
    )
