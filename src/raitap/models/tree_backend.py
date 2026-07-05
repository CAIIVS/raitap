"""Shared base for fitted-estimator tree/tabular backends (XGBoost, sklearn, LightGBM).

Owns everything common across tree libraries: the fitted-estimator accessor
(``EstimatorProvider`` for TreeSHAP), the torch<->numpy ``predict_proba`` bridge,
and the ``(N, C)`` probability output contract. Concrete subclasses add one
library's file loading (``from_path``) and prediction call (``_predict_proba``),
and declare their ``provides`` / ``extensions`` via ``@register``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from raitap.models.base_backend import ModelBackend
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from raitap.types import Capability
else:
    torch = lazy_import("torch")


class TabularTreeBackend(ModelBackend, ABC):
    """Backend for a fitted tree/tabular estimator that emits class probabilities."""

    provides: ClassVar[frozenset[Capability]] = frozenset()

    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator

    def fitted_estimator(self) -> Any:
        """Raw fitted estimator (consumed by shap.TreeExplainer)."""
        return self._estimator

    def __call__(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        prepared = self._prepare_inputs(inputs)
        array = (
            prepared.detach().cpu().numpy()
            if isinstance(prepared, torch.Tensor)
            else np.asarray(prepared)
        )
        # ascontiguousarray (not asarray): predict_proba may return a non-contiguous
        # view, which torch.from_numpy rejects.
        probabilities = np.ascontiguousarray(self._predict_proba(array), dtype=np.float32)
        return torch.from_numpy(probabilities)

    @property
    def hardware_label(self) -> str:
        # Tree libraries run on CPU here. Override in a subclass that supports
        # GPU-accelerated inference.
        return "CPU"

    @abstractmethod
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Return ``(N, C)`` class probabilities for a numpy feature matrix."""

    @classmethod
    @abstractmethod
    def from_path(
        cls, path: Path, *, model_cfg: Any, hardware: str, allow_unsafe_pickle: bool = False
    ) -> ModelBackend:
        """Load the fitted estimator from a file and wrap it."""
