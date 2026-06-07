"""input_spec is threaded into compute_attributions (#267)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch

from raitap.transparency.contracts import InputKind, InputSpec, TensorLayout
from raitap.transparency.explainers.base_explainer import AttributionOnlyExplainer

if TYPE_CHECKING:
    from collections.abc import Mapping

    from raitap.models.access import ExplanationModel
    from raitap.transparency.contracts import ExplainerSemanticsHints


class _SpyExplainer(AttributionOnlyExplainer):
    algorithm_registry: ClassVar[Mapping[str, ExplainerSemanticsHints]] = {}
    algorithm = "spy"
    seen_kind: ClassVar[InputKind | None] = None

    def compute_attributions(
        self,
        model: ExplanationModel,
        inputs: torch.Tensor,
        *,
        input_spec: object | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del model, kwargs
        type(self).seen_kind = getattr(input_spec, "kind", None)
        return torch.zeros_like(inputs)


def _tabular_spec(rows: int) -> InputSpec:
    return InputSpec(
        kind=InputKind.TABULAR,
        shape=(rows, 4),
        layout=TensorLayout.BATCH_FEATURE,
        feature_names=None,
        metadata=None,
    )


def test_compute_attributions_receives_input_spec_unbatched() -> None:
    _SpyExplainer.seen_kind = None
    _SpyExplainer()._compute_with_optional_batches(
        model=cast("ExplanationModel", object()),  # spy ignores model
        inputs=torch.zeros(1, 4),
        attribution_kwargs={},
        backend=None,
        input_spec=_tabular_spec(1),
    )
    assert _SpyExplainer.seen_kind is InputKind.TABULAR


def test_compute_attributions_receives_input_spec_batched() -> None:
    # batch_size < batch -> the batched loop must also forward input_spec.
    _SpyExplainer.seen_kind = None
    _SpyExplainer()._compute_with_optional_batches(
        model=cast("ExplanationModel", object()),  # spy ignores model
        inputs=torch.zeros(2, 4),
        attribution_kwargs={},
        backend=None,
        input_spec=_tabular_spec(2),
        batch_size=1,
        show_progress=False,
    )
    assert _SpyExplainer.seen_kind is InputKind.TABULAR
