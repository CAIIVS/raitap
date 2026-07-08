import dataclasses
import importlib

import pytest

from raitap.configs.schema import CocoLabelsConfig, DetectionJsonLabelsConfig, DirectoryLabelsConfig


def test_coco_config_has_no_tabular_fields() -> None:
    names = {f.name for f in dataclasses.fields(CocoLabelsConfig)}
    assert "id_column" not in names
    assert "column" not in names
    assert "encoding" not in names
    assert {"use", "source", "id_strategy"} <= names


def test_directory_config_has_only_use() -> None:
    names = {f.name for f in dataclasses.fields(DirectoryLabelsConfig)}
    assert names == {"use"}


def test_labelformat_enum_is_gone() -> None:
    data_types = importlib.import_module("raitap.data.types")
    with pytest.raises(AttributeError):
        getattr(data_types, "LabelFormat")  # noqa: B009


# Ground truth (issue #301): composing ``+data/labels=directory`` onto the
# AppConfig schema lands the variant at ``cfg.data.labels`` as a ``use:
# "directory"`` selector (config layer). The real class FQN never appears in
# config — it lives only in ``raitap._adapters._TARGET_FQN`` and is resolved by
# ``raitap.configs.registry_resolve.resolve_target_fqn`` at instantiate-time.
_DIRECTORY_LABEL_PARSER_FQN = "raitap.data.label_parsers.directory.DirectoryLabelParser"


def _register_labels_group() -> None:
    """Register the ``data/labels`` group via the canonical ``register_configs``.

    Uses the same registration path as production (and the rest of the suite):
    it sets up the AppConfig schema and every family's group nodes consistently.
    An earlier direct ``store.add_to_hydra_store(overwrite_ok=True)`` workaround
    flushed hydra-zen builders in isolation, clobbering other groups' short
    ``use`` schema nodes and breaking later tests (e.g. reporting compose).
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
    # Assertion runs unconditionally (no swallowing). The composed node carries
    # a bare ``use:`` selector, never a class FQN (issue #301).
    assert cfg.data.labels.use == "directory"
    assert not hasattr(cfg.data.labels, "_target_")


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


def test_directory_labels_use_resolves_to_directory_label_parser() -> None:
    """The registration + resolver seam Task B1/A1 own: a ``use:`` selector
    resolves to the real class FQN through the closed registry populated at
    adapter registration — never through an arbitrary ``_target_`` in config.

    Migrating ``create_label_parser`` itself to consume ``use:`` instead of
    ``_target_`` is later (Phase D) work; this test scopes to the schema +
    registry seam introduced here.
    """
    _register_labels_group()

    from raitap.configs.registry_resolve import resolve_target_fqn
    from raitap.data.label_parsers.directory import DirectoryLabelParser

    cfg = DirectoryLabelsConfig()
    assert cfg.use == "directory"
    assert not hasattr(cfg, "_target_")

    fqn = resolve_target_fqn("data/labels", cfg.use)
    assert fqn == _DIRECTORY_LABEL_PARSER_FQN

    module_path, _, class_name = fqn.rpartition(".")
    resolved_cls = getattr(importlib.import_module(module_path), class_name)
    assert resolved_cls is DirectoryLabelParser


def test_detection_json_config_has_exactly_use_source_id_strategy() -> None:
    names = {f.name for f in dataclasses.fields(DetectionJsonLabelsConfig)}
    assert names == {"use", "source", "id_strategy"}


# ---------------------------------------------------------------------------
# Cross-variant leakage test (Task 10)
# ---------------------------------------------------------------------------

# Fields that belong exclusively to the tabular variant and must NOT appear
# in any other variant's builder dataclass.
_TABULAR_ONLY_FIELDS = {"id_column", "column", "encoding"}

# Fields that belong exclusively to the voc variant.
_VOC_ONLY_FIELDS = {"class_names"}

# Variants that must have ONLY ``use`` (no source, no strategy, nothing).
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
    - ``directory`` has only ``use``.
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
        assert field_names == {"use"}, (
            f"{registry_name!r} builder should have only 'use' but got {field_names}"
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
