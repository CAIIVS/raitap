"""XGBoost tree backend — loads a fitted ``XGBClassifier`` for TreeSHAP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.models.registration import register
from raitap.models.tree_backend import TabularTreeBackend
from raitap.types import Capability

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

_XGBOOST_INSTALL_HINT = "XGBoostBackend requires xgboost. Install it with `uv sync --extra tree`."


@register(provides={Capability.TREE_MODEL, Capability.PREDICT_PROBA}, extensions={".ubj"})
class XGBoostBackend(TabularTreeBackend):
    """Backend wrapping a fitted XGBoost classifier loaded from a ``.ubj`` file."""

    @classmethod
    def from_path(
        cls, path: Path, *, model_cfg: Any, hardware: str, allow_unsafe_pickle: bool = False
    ) -> XGBoostBackend:
        # model_cfg / hardware / allow_unsafe_pickle unused: .ubj is a safe native
        # binary (not pickle) and XGBoost prediction is CPU-bound here.
        del model_cfg, hardware, allow_unsafe_pickle
        try:
            import xgboost
        except ImportError as error:
            raise ImportError(_XGBOOST_INSTALL_HINT) from error

        estimator = xgboost.XGBClassifier()
        estimator.load_model(str(path))
        return cls(estimator)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return self._estimator.predict_proba(X)
