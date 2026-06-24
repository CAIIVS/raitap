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


class LabelFormat(StrEnum):
    """On-disk label file format selected by ``LabelsConfig.format``.

    ``native`` is RAITAP's own shape (classification: CSV/TSV/Parquet or the
    ``directory`` source; detection: the JSON record list). The others are
    converted to the native intermediate by a registered
    ``LabelFormatAdapter`` before the task family aligns them. StrEnum so YAML
    users can write the raw value.
    """

    native = "native"
    coco = "coco"
    yolo = "yolo"
    voc = "voc"


#: Reserved ``LabelsConfig.source`` value selecting folder-as-label ingestion:
#: classification labels are derived from each sample's top-level class
#: subdirectory (torchvision ``ImageFolder`` style; no labels file). Kept as a
#: plain ``str`` so it round-trips through OmegaConf; ``LabelsConfig.source``
#: stays ``str | None`` (a path or this sentinel).
DIRECTORY_LABELS_SOURCE = "directory"


class Preprocessing(StrEnum):
    """Named values for ``DataConfig.preprocessing``.

    Custom-file preprocessing is selected by passing a path string directly
    — it is not enumerated here. ``None`` (the default) means preprocessing
    is OFF.
    """

    model_bundled = "model-bundled"


class InputModality(StrEnum):
    """Data modality recorded by :class:`raitap.data.data.Data` at load time.

    Only the modalities raitap loads today. A future non-image family adds its
    member here, an extension entry in ``MODALITY_EXTENSIONS``, a load branch in
    ``Data._load_data``, and a ``kind``/``layout`` branch in
    ``infer_data_input_metadata`` (which switches on the recorded modality).
    """

    image = "image"
    tabular = "tabular"


#: Single source of truth for which file extensions map to which modality.
#: Imported by ``data.data`` and ``data.metadata`` so the sets are not
#: redeclared at each call site.
MODALITY_EXTENSIONS: dict[InputModality, frozenset[str]] = {
    InputModality.image: frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"}),
    InputModality.tabular: frozenset({".csv", ".tsv", ".parquet"}),
}
