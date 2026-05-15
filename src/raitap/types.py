"""Shared enum aliases used by :mod:`raitap.configs.schema`.

Lives at the package root so importing it doesn't trigger the heavy
sub-package ``__init__`` modules (``raitap.data``, ``raitap.models``,
``raitap.metrics``) that pull in torch, pandas, torchmetrics, and the
tracking subsystem.

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


class LabelEncoding(StrEnum):
    index = "index"
    one_hot = "one_hot"
    argmax = "argmax"


class IdStrategy(StrEnum):
    auto = "auto"
    relative_path = "relative_path"
    stem = "stem"


class Task(StrEnum):
    binary = "binary"
    multiclass = "multiclass"
    multilabel = "multilabel"
