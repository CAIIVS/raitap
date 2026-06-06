"""Cross-cutting enum aliases used by :mod:`raitap.configs.schema`.

Lives at the package root so importing it doesn't trigger the heavy
sub-package ``__init__`` modules (``raitap.models``, ``raitap.metrics``)
that pull in torch and torchmetrics. Holds only the enums whose owning
module would itself be heavy to import — ``Hardware`` (models/deps) and
``Task`` (metrics/tracking). Module-local enums live next to their owning
module: see :mod:`raitap.data.types`.

``StrEnum`` members are string subclasses, so YAML / Python users can pass
the raw value (``"cpu"``) **or** the enum member (``Hardware.cpu``)
interchangeably. OmegaConf validates structured-config fields by matching
the input string to the enum *member name*, so member names must equal
their value (lowercase, matching what users write in YAML). Validators in
the consuming modules import the same alias so each value set has exactly
one source of truth.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

#: A ragged batch of detection inputs: one native-resolution ``(C, H, W)``
#: float32 tensor per image. Used as ``Data.tensor`` when
#: ``task_kind == TaskKind.detection``.  Defined with ``TYPE_CHECKING`` so
#: importing this module never triggers a real ``torch`` import.
DetectionInputs = list["torch.Tensor"]


class Hardware(StrEnum):
    cpu = "cpu"
    gpu = "gpu"


class Task(StrEnum):
    binary = "binary"
    multiclass = "multiclass"
    multilabel = "multilabel"


class TaskKind(StrEnum):
    """Model task family.

    Adapters declare which task families they accept via the
    ``supported_tasks: ClassVar[frozenset[TaskKind]]`` attribute on
    :class:`raitap._adapters.AdapterMixin` (default
    ``{TaskKind.classification}`` so legacy adapters stay correct without
    explicit declaration). Issue #146 groundwork.
    """

    classification = "classification"
    detection = "detection"
    segmentation = "segmentation"
    seq2seq = "seq2seq"
    regression = "regression"


class Capability(StrEnum):
    """A backend capability an algorithm may require.

    Backends declare what they ``provides``; algorithm hints declare what they
    ``requires``. An algorithm runs on a backend iff ``requires <= provides``.
    """

    AUTOGRAD = (
        "autograd"  # differentiable live model + input gradients (PGD, IntegratedGradients, CROWN)
    )
    TREE_MODEL = (
        "tree_model"  # roadmap: tree-ensemble structure (TreeSHAP). No provider/requirer yet.
    )
    PREDICT_PROBA = "predict_proba"  # roadmap: class-probability outputs. No provider/requirer yet.


#: A backend that exposes only the universal forward path (no special model
#: shapes). Model-agnostic explainers (requires == empty) run on it; gradient
#: and tree explainers gate out. Named for legibility over a bare ``frozenset()``.
FORWARD_ONLY: frozenset[Capability] = frozenset()
