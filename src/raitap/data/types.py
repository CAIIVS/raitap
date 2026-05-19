"""Data-module enum aliases (label encoding, sample-id strategy).

Kept in their own module so :mod:`raitap.configs.schema` can import them
without triggering :mod:`raitap.data.__init__`, which pulls in torch,
pandas, and the tracking subsystem. User-facing re-exports live in
:mod:`raitap.data`.

``StrEnum`` members are string subclasses, so YAML / Python users can pass
the raw value (``"index"``) **or** the enum member (``LabelEncoding.index``)
interchangeably. OmegaConf validates structured-config fields by matching
the input string to the enum *member name*, so member names equal their
value (lowercase, matching what users write in YAML).
"""

from __future__ import annotations

from enum import StrEnum


class LabelEncoding(StrEnum):
    # ``index`` shadows ``str.index`` — pyright (correctly) flags member assignments
    # whose name matches an inherited str method. Runtime works fine because Enum
    # member assignment goes through ``EnumMeta.__setattr__`` and binds the member
    # object, not the method. Suppress just this line.
    index = "index"  # type: ignore[assignment]
    one_hot = "one_hot"
    argmax = "argmax"


class IdStrategy(StrEnum):
    auto = "auto"
    relative_path = "relative_path"
    stem = "stem"


class LabelKind(StrEnum):
    """Discriminator for ``data.labels`` loader routing.

    Values match the YAML / Python API string forms exactly; ``StrEnum``
    members serialise as their ``.value`` via omegaconf, so YAML callers
    can write ``kind: detection`` and Python callers can pass
    ``LabelKind.detection`` interchangeably.
    """

    classification = "classification"
    detection = "detection"


class Preprocessing(StrEnum):
    """Named values for ``DataConfig.preprocessing``.

    Custom-file preprocessing is selected by passing a path string directly
    — it is not enumerated here. ``None`` (the default) means preprocessing
    is OFF.
    """

    model_bundled = "model-bundled"
