"""TaskFamily protocol + per-call context objects.

A task family is the strategy object for one ``TaskKind``: it owns every
task-specific behavior the pipeline phases used to branch on (data loading,
forward-output extraction, transparency scope strategy, metrics adaptation,
and the per-kind robustness / preprocessing / prediction-summary rules).

Method signatures take ``*Context`` objects so a family can carry
family-specific extras (e.g. a tokenizer or attention masks for seq2seq)
without changing the shared signatures. Classification and detection ignore
``extras``.

Annotation policy: ``payload`` / ``inputs`` / ``labels`` and the
``extract_forward`` / ``metrics_inputs`` returns are typed ``Any`` on purpose —
they vary per family (classification ``Tensor`` vs detection ``list[dict]``),
and a family is resolved at runtime by ``TaskKind``, so no caller knows the
static payload type and generics would erase to ``Any`` at every call site.
Pure-input params the implementation narrows itself use ``object`` (see the
``validate_*`` methods), not ``Any``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch.nn as nn

    from raitap.configs.schema import AppConfig
    from raitap.data.data import Data
    from raitap.models.backend import ModelBackend
    from raitap.pipeline.outputs import ForwardOutput
    from raitap.transparency.contracts import ExplanationOutputSpace
    from raitap.transparency.phase import PreparedExplainer
    from raitap.transparency.results import ExplanationResult
    from raitap.types import TaskKind


@dataclass(frozen=True)
class ForwardContext:
    """Inputs to :meth:`TaskFamily.extract_forward`."""

    backend: ModelBackend
    inputs: Any  # family payload-shaped: Tensor (classification) | list[Tensor] (detection)
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

    prepared: PreparedExplainer
    forward_output: ForwardOutput
    data: Data
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TaskFamily(Protocol):
    """Strategy object for one task family. See module docstring."""

    kind: TaskKind
    fixed_output_space: ExplanationOutputSpace | None
    #: Whether the robustness phase runs for this family (constant per family).
    supports_robustness: bool
    #: Whether data/model preprocessing transforms are allowed (constant per family).
    allows_preprocessing: bool

    def validate_payload(self, payload: object) -> None:
        """Raise ``ValueError`` if ``payload`` is wrong for this family.

        Recovers the construction-time invariant that ``ForwardOutput`` used
        to enforce in ``__post_init__``.
        """
        raise NotImplementedError

    def adapt_loaded_inputs(self, tensor: object) -> Any:
        """Shape the freshly-loaded dense tensor for this family.

        Classification keeps the dense ``(N, C, H, W)`` tensor; detection
        unbinds it into a ragged ``list[(C, H, W)]``.
        """
        raise NotImplementedError

    def validate_inputs(self, tensor: object) -> None:
        """Validate the (post-adapt) inputs match this family's contract."""
        raise NotImplementedError

    def load_labels(self, cfg: AppConfig, *, tensor: object, sample_ids: object) -> Any:
        """Load labels in this family's on-disk shape (or None)."""
        raise NotImplementedError

    def validate_labels(self, labels: object) -> None:
        """Raise if loaded labels don't match this family's expected shape."""
        raise NotImplementedError

    def extract_forward(self, ctx: ForwardContext, *, batch_size: int) -> Any:
        """Run the backend forward and return this family's payload."""
        raise NotImplementedError

    def payload_batch_size(self, payload: object) -> int:
        """Number of samples represented by ``payload``."""
        raise NotImplementedError

    def explain(self, ctx: ExplainContext) -> list[ExplanationResult]:
        """Produce explanation results (shared loop or per-element K-loop)."""
        raise NotImplementedError

    def metrics_inputs(
        self, config: AppConfig, forward_output: ForwardOutput, labels: object
    ) -> Any:
        """Adapt payload + labels into ``(preds, targets)`` for the metric
        adapters, or return ``None`` to skip metrics for this input."""
        raise NotImplementedError

    def prediction_summaries(
        self,
        payload: object,
        *,
        sample_ids: object = None,
        targets: object = None,
        output_kind: Any = None,
    ) -> list | None:
        """Per-sample prediction summary rows, or ``None`` if N/A."""
        raise NotImplementedError

    def matches_model(self, model: nn.Module) -> bool:
        """Whether this family recognises ``model`` by architecture.

        Used by backend task-kind auto-inference. Families that aren't
        auto-detectable (classification is the fallback) need not implement it.
        """
        raise NotImplementedError
