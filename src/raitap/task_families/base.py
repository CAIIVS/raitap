"""TaskFamily protocol + per-call context objects.

A task family is the strategy object for one ``TaskKind``: it owns every
task-specific behavior the pipeline phases used to branch on (data loading,
forward-output extraction, transparency scope strategy, metrics adaptation,
and the per-kind robustness / preprocessing / prediction-summary rules).

Method signatures take ``*Context`` objects so a family can carry
family-specific extras (e.g. a tokenizer or attention masks for seq2seq)
without changing the shared signatures. Classification and detection ignore
``extras``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from raitap.pipeline.outputs import ForwardOutput
    from raitap.transparency.contracts import ExplanationOutputSpace
    from raitap.transparency.results import ExplanationResult
    from raitap.types import TaskKind


@dataclass(frozen=True)
class ForwardContext:
    """Inputs to :meth:`TaskFamily.extract_forward`."""

    backend: Any
    inputs: Any
    # ``frozen=True`` blocks reassigning ``extras``, but the dict itself is
    # intentionally mutable so a family can populate it before forwarding the
    # context (e.g. seq2seq stashing a tokenizer). Do not switch to
    # ``MappingProxyType`` — mutation here is by design.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExplainContext:
    """Inputs to :meth:`TaskFamily.explain`.

    ``prepared`` is the per-explainer setup produced by the shared
    ``prepare_explainer`` helper in the transparency phase (explainer +
    visualisers + resolved kwargs + run dir + backend).
    """

    prepared: Any
    forward_output: Any
    data: Any
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TaskFamily(Protocol):
    """Strategy object for one task family. See module docstring."""

    kind: TaskKind
    fixed_output_space: ExplanationOutputSpace | None

    def validate_payload(self, payload: object) -> None:
        """Raise ``ValueError`` if ``payload`` is wrong for this family.

        Recovers the construction-time invariant that ``ForwardOutput`` used
        to enforce in ``__post_init__``.
        """
        raise NotImplementedError

    def adapt_loaded_inputs(self, tensor: Any) -> Any:
        """Shape the freshly-loaded dense tensor for this family.

        Classification keeps the dense ``(N, C, H, W)`` tensor; detection
        unbinds it into a ragged ``list[(C, H, W)]``.
        """
        raise NotImplementedError

    def validate_inputs(self, tensor: Any) -> None:
        """Validate the (post-adapt) inputs match this family's contract."""
        raise NotImplementedError

    def load_labels(self, cfg: Any, *, tensor: Any, sample_ids: Any) -> Any:
        """Load labels in this family's on-disk shape (or None)."""
        raise NotImplementedError

    def validate_labels(self, labels: Any) -> None:
        """Raise if loaded labels don't match this family's expected shape."""
        raise NotImplementedError

    def extract_forward(self, ctx: ForwardContext, *, batch_size: int) -> Any:
        """Run the backend forward and return this family's payload."""
        raise NotImplementedError

    def explain(self, ctx: ExplainContext) -> list[ExplanationResult]:
        """Produce explanation results (shared loop or per-element K-loop)."""
        raise NotImplementedError

    def metrics_inputs(self, config: Any, forward_output: ForwardOutput, labels: Any) -> Any:
        """Adapt payload + labels into ``(preds, targets)`` for the metric
        adapters, or return ``None`` to skip metrics for this input."""
        raise NotImplementedError

    def supports_robustness(self) -> bool:
        """Whether the robustness phase runs for this family."""
        raise NotImplementedError

    def prediction_summaries(
        self, payload: Any, *, sample_ids: Any = None, targets: Any = None
    ) -> list | None:
        """Per-sample prediction summary rows, or ``None`` if N/A."""
        raise NotImplementedError

    @property
    def allows_preprocessing(self) -> bool:
        """Whether data/model preprocessing transforms are allowed."""
        raise NotImplementedError
