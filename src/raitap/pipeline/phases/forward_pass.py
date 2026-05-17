"""Forward-pass phase — primary-tensor extraction + batched forward."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def forward_pass(config: AppConfig, backend: Any, inputs: torch.Tensor) -> torch.Tensor:
    """Run the model backend forward in chunks of ``forward_batch_size``.

    Returns a CPU-detached primary tensor.
    """
    batch_size = resolve_forward_batch_size(config)
    total_batch = int(inputs.shape[0])
    if total_batch <= batch_size:
        prepared_inputs = backend._prepare_inputs(inputs)
        raw_output: Any = backend(prepared_inputs)
        forward_output = extract_primary_tensor(raw_output).detach().cpu()
        del prepared_inputs, raw_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return forward_output

    chunks: list[torch.Tensor] = []
    for start in range(0, total_batch, batch_size):
        end = min(start + batch_size, total_batch)
        prepared_inputs = backend._prepare_inputs(inputs[start:end])
        raw_output = backend(prepared_inputs)
        chunks.append(extract_primary_tensor(raw_output).detach().cpu())
        del prepared_inputs, raw_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)
