import dataclasses
import importlib

import pytest

from raitap.configs.schema import CocoLabelsConfig, DetectionJsonLabelsConfig, DirectoryLabelsConfig


def test_coco_config_has_no_tabular_fields() -> None:
    names = {f.name for f in dataclasses.fields(CocoLabelsConfig)}
    assert "id_column" not in names
    assert "column" not in names
    assert "encoding" not in names
    assert {"_target_", "source", "id_strategy"} <= names


def test_directory_config_has_only_target() -> None:
    names = {f.name for f in dataclasses.fields(DirectoryLabelsConfig)}
    assert names == {"_target_"}


def test_labelformat_enum_is_gone() -> None:
    data_types = importlib.import_module("raitap.data.types")
    with pytest.raises(AttributeError):
        getattr(data_types, "LabelFormat")  # noqa: B009


# Ground truth (see task-2-report.md): composing ``+data/labels=directory`` onto
# the AppConfig schema lands the variant at ``cfg.data.labels`` with the FQN
# ``_target_`` that hydra-zen ``builds()`` injects.
_COMPOSED_TARGET = "raitap.data.label_parsers.directory.DirectoryLabelParser"


def _register_labels_group() -> None:
    """Register the ``data/labels`` group via the canonical ``register_configs``.

    Uses the same registration path as production (and the rest of the suite):
    it sets up the AppConfig schema and every family's group nodes consistently.
    An earlier direct ``store.add_to_hydra_store(overwrite_ok=True)`` workaround
    flushed hydra-zen builders in isolation, clobbering other groups' short
    ``_target_`` schema nodes and breaking later tests (e.g. reporting compose).
    """
    from raitap.configs import register_configs

    register_configs()


def test_directory_parser_group_lands_at_data_labels() -> None:
    """De-risk (Path A): the nested ``data/labels`` group composes onto
    ``cfg.data.labels`` as a single config (flat semantics at a nested path)."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    _register_labels_group()
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="raitap_schema", overrides=["+data/labels=directory"])
    # Assertion runs unconditionally (no swallowing). The composed value is the
    # FQN hydra-zen stores, NOT the short dataclass default.
    assert cfg.data.labels._target_ == _COMPOSED_TARGET


def test_directory_group_rejects_foreign_field() -> None:
    """De-risk (Path A): a field the directory variant lacks fails at compose.

    Uses a struct-mode override (``data.labels.id_column=x`` — no ``+``) so
    OmegaConf's struct check fires; ``+`` force-adds and would bypass it.
    """
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    from hydra.errors import ConfigCompositionException

    _register_labels_group()
    GlobalHydra.instance().clear()
    with pytest.raises(ConfigCompositionException), initialize(version_base=None, config_path=None):
        compose(
            config_name="raitap_schema",
            overrides=["+data/labels=directory", "data.labels.id_column=x"],
        )


def test_create_label_parser_handles_both_target_forms() -> None:
    """``create_label_parser`` must instantiate for BOTH ``_target_`` shapes:

    * short bare name (``DirectoryLabelsConfig()`` dataclass default), resolved
      against the ``raitap.data.label_parsers.`` prefix;
    * the dotted FQN hydra-zen ``builds()`` stamps on the group-composed cfg.
    """
    _register_labels_group()
    from raitap.data.label_parsers.directory import DirectoryLabelParser
    from raitap.data.label_parsers.factory import create_label_parser

    short = create_label_parser(DirectoryLabelsConfig())
    assert isinstance(short, DirectoryLabelParser)

    fqn = create_label_parser({"_target_": _COMPOSED_TARGET})
    assert isinstance(fqn, DirectoryLabelParser)


def test_detection_json_config_has_exactly_target_source_id_strategy() -> None:
    names = {f.name for f in dataclasses.fields(DetectionJsonLabelsConfig)}
    assert names == {"_target_", "source", "id_strategy"}


# ---------------------------------------------------------------------------
# Cross-variant leakage test (Task 10)
# ---------------------------------------------------------------------------

# Fields that belong exclusively to the tabular variant and must NOT appear
# in any other variant's builder dataclass.
_TABULAR_ONLY_FIELDS = {"id_column", "column", "encoding"}

# Fields that belong exclusively to the voc variant.
_VOC_ONLY_FIELDS = {"class_names"}

# Variants that must have ONLY ``_target_`` (no source, no strategy, nothing).
_TARGET_ONLY_VARIANTS: set[str] = {"directory"}

# Variants that carry source + id_strategy but NO tabular fields and NO
# class_names.
_DETECTION_VARIANTS: set[str] = {"coco", "yolo", "detection_json"}


@pytest.mark.parametrize(
    "registry_name",
    ["directory", "tabular", "coco", "yolo", "voc", "detection_json"],
)
def test_no_cross_variant_field_leakage(registry_name: str) -> None:
    """Each label-parser builder dataclass must expose only its own fields.

    Specifically:
    - ``directory`` has only ``_target_``.
    - ``coco``/``yolo``/``detection_json`` have no tabular-only fields and no
      ``class_names``.
    - ``voc`` has ``class_names`` but no tabular-only fields.
    - ``tabular`` has tabular-only fields but no ``class_names``.
    """
    from raitap._adapters import _BUILDERS

    _register_labels_group()

    builders = _BUILDERS.get("data/labels", {})
    assert registry_name in builders, (
        f"Registry name {registry_name!r} not found in _BUILDERS['data/labels']; "
        f"registered: {sorted(builders)}"
    )
    builder = builders[registry_name]
    field_names = {f.name for f in dataclasses.fields(builder)}

    if registry_name in _TARGET_ONLY_VARIANTS:
        assert field_names == {"_target_"}, (
            f"{registry_name!r} builder should have only '_target_' but got {field_names}"
        )

    if registry_name in _DETECTION_VARIANTS:
        leaked = (_TABULAR_ONLY_FIELDS | _VOC_ONLY_FIELDS) & field_names
        assert not leaked, f"{registry_name!r} builder leaks foreign fields: {leaked}"

    if registry_name == "voc":
        leaked = _TABULAR_ONLY_FIELDS & field_names
        assert not leaked, f"voc builder leaks tabular-only fields: {leaked}"
        assert field_names >= _VOC_ONLY_FIELDS, "voc builder must have 'class_names'"

    if registry_name == "tabular":
        assert field_names >= _TABULAR_ONLY_FIELDS, (
            f"tabular builder is missing expected fields; got {field_names}"
        )
        leaked = _VOC_ONLY_FIELDS & field_names
        assert not leaked, "tabular builder should not have 'class_names'"
