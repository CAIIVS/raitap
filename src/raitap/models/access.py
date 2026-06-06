"""Capability-typed model access for explainers/assessors.

The explanation pipeline asks a backend for the model SHAPE an adapter needs,
keyed by the adapter's declared ``requires`` capability. Each shape is one
capability; backends opt into a shape by implementing its Protocol. The base
``ModelBackend`` carries only the universal ``predict_callable`` shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from raitap.types import Capability
from raitap.utils.errors import BackendIncompatibilityError

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    import torch.nn as nn

    from raitap.models.backend import ModelBackend

    #: What an explainer receives: a torch ``nn.Module`` (autograd shape) or a
    #: forward-only predict callable. Precise for type-checkers; ``Any`` at
    #: runtime (below) keeps this module torch-free for the deps bootstrap.
    ExplanationModel = nn.Module | Callable[[torch.Tensor], torch.Tensor]
else:
    ExplanationModel = Any


@runtime_checkable
class AutogradModelProvider(Protocol):
    """A backend that can hand out a live differentiable torch ``nn.Module``."""

    def autograd_module(self) -> nn.Module: ...


def _require(backend: ModelBackend, proto: type) -> Any:
    """Narrow ``backend`` to ``proto`` or raise. The capability gate runs first,
    so a failure here means the backend declared the capability but did not
    implement its Protocol (a backend-author bug)."""
    if not isinstance(backend, proto):
        raise BackendIncompatibilityError(
            adapter="explanation_model",
            backend=type(backend).__name__,
            missing=[proto.__name__],
        )
    return backend


#: Single authoritative binding: capability -> how to extract its model shape.
#: ``EstimatorProvider`` / ``TREE_MODEL`` join here with the tree backend (#246).
_SHAPE_BY_CAPABILITY: dict[Capability, Callable[[ModelBackend], ExplanationModel]] = {
    Capability.AUTOGRAD: lambda b: _require(b, AutogradModelProvider).autograd_module(),
}
SHAPE_CAPABILITIES = frozenset(_SHAPE_BY_CAPABILITY)


def explanation_model(backend: ModelBackend, adapter: Any) -> ExplanationModel:
    """Return the model shape ``adapter`` needs from ``backend``.

    ``adapter`` is any ``AdapterMixin`` (explainer or assessor); both expose
    ``required_capabilities()``. Invariant: an adapter declares at most one
    shape-capability. Model-agnostic adapters get the universal predict callable.
    """
    shapes = adapter.required_capabilities() & SHAPE_CAPABILITIES
    if not shapes:
        return backend.predict_callable()
    (capability,) = shapes
    return _SHAPE_BY_CAPABILITY[capability](backend)
