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
    import importlib

    data_types = importlib.import_module("raitap.data.types")
    with pytest.raises(AttributeError):
        getattr(data_types, "LabelFormat")  # noqa: B009


# Ground truth (see task-2-report.md): composing ``+data/labels=directory`` onto
# the AppConfig schema lands the variant at ``cfg.data.labels`` with the FQN
# ``_target_`` that hydra-zen ``builds()`` injects.
_COMPOSED_TARGET = "raitap.data.label_parsers.directory.DirectoryLabelParser"


def _register_labels_group() -> None:
    """Register the ``data/labels`` group + AppConfig schema directly.

    Bypasses ``register_configs()`` (which imports transparency and other
    families that are broken mid-refactor on this branch) by importing only the
    label_parsers package — enough to fire the ``@label_parser`` decorator — and
    flushing the hydra-zen store. The AppConfig schema is needed as the compose
    base so the ``data.labels`` package has a struct to land in.
    """
    importlib.import_module("raitap.data.label_parsers")
    from hydra.core.config_store import ConfigStore

    from raitap._adapters import store
    from raitap.configs.schema import AppConfig

    store.add_to_hydra_store(overwrite_ok=True)
    ConfigStore.instance().store(name="raitap_schema", node=AppConfig)


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
