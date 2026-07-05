"""Backend-agnostic model runtime base + shared shape adaptation."""

from __future__ import annotations

from abc import ABC, abstractmethod

# Runtime (not TYPE_CHECKING) import: Callable is in a public annotation that
# get_type_hints() may evaluate at runtime.
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, overload

from raitap.types import Capability, ResolvedHardware, TaskKind
from raitap.utils.errors import ModelInputShapeError
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import torch
else:
    torch = lazy_import("torch")


@overload
def _adapt_input_shape(
    inputs: torch.Tensor,
    expected: tuple[int | None, ...] | None,
) -> torch.Tensor: ...


@overload
def _adapt_input_shape(
    inputs: np.ndarray[Any, Any],
    expected: tuple[int | None, ...] | None,
) -> np.ndarray[Any, Any]: ...


def _adapt_input_shape(
    inputs: torch.Tensor | np.ndarray[Any, Any],
    expected: tuple[int | None, ...] | None,
) -> torch.Tensor | np.ndarray[Any, Any]:
    """Reshape ``inputs`` to match ``expected`` (with ``None`` = batch dim).

    Returns inputs unchanged when ``expected`` is ``None`` or already matches.
    Raises :class:`ModelInputShapeError` when per-sample numel mismatches.
    """
    if expected is None:
        return inputs

    input_shape = tuple(int(dim) for dim in inputs.shape)
    batch = input_shape[0] if input_shape else 1
    target = tuple(batch if dim is None else int(dim) for dim in expected)

    if input_shape == target:
        return inputs

    input_numel = 1
    for dim in input_shape:
        input_numel *= dim
    target_numel = 1
    for dim in target:
        target_numel *= dim

    if input_numel != target_numel:
        raise ModelInputShapeError(expected_shape=expected, input_shape=input_shape)

    return inputs.reshape(target)


class ModelBackend(ABC):
    """Backend-agnostic model runtime interface."""

    provides: ClassVar[frozenset[Capability]] = frozenset()
    extensions: ClassVar[frozenset[str]] = frozenset()
    # uv extra installing this backend's runtime, and the hardware values it
    # ships per-wheel. Set by ``@register``; read import-free by the deps
    # scanner. ``None``/empty for non-file or single-wheel backends.
    extra: ClassVar[str | None] = None
    supported_hardware: ClassVar[frozenset[ResolvedHardware]] = frozenset()
    # Declared per-sample input shape with ``None`` marking dynamic dims
    # (typically the batch dim). ``None`` overall = no rank adaptation.
    expected_input_shape: tuple[int | None, ...] | None = None
    # Optional id->name table for the model's output classes (e.g. torchvision
    # detection ``weights.meta["categories"]``). ``None`` when unavailable.
    category_names: list[str] | None = None

    @classmethod
    def from_path(
        cls, path: Path, *, model_cfg: Any, hardware: str, allow_unsafe_pickle: bool = False
    ) -> ModelBackend:
        """Construct this backend from a model file.

        Default raises: file loading is not universal (in-memory/test backends
        never load from a path). File-backed backends declared with
        ``extensions=`` in ``@register`` override this; the registry calls it
        from ``model._load_from_path``.
        """
        raise NotImplementedError(f"{cls.__name__} does not support construction from a file path.")

    @property
    @abstractmethod
    def hardware_label(self) -> str:
        """Human-readable label for the resolved runtime backend."""

    @property
    def task_kind(self) -> TaskKind:
        """Task family this backend serves. Defaults to ``classification``."""
        return TaskKind.classification

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Adapt runtime inputs to this backend's preferred device/layout."""
        return _adapt_input_shape(inputs, self.expected_input_shape)

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Adapt explainer/runtime kwargs for this backend."""
        return kwargs

    @abstractmethod
    def __call__(self, inputs: torch.Tensor, **kwargs: Any) -> Any:
        """Run inference for ``inputs``."""

    def predict_callable(self) -> Callable[..., torch.Tensor]:
        """Forward-only predict fn. Universal shape; defaults to ``__call__``."""
        return self.__call__
