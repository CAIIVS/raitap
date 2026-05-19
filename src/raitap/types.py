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
