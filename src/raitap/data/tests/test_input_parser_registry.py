import pytest

from raitap.configs.schema import InputsConfig
from raitap.data.input_parsers.base import InputParser
from raitap.data.input_parsers.registration import input_parser
from raitap.data.label_parsers.factory import create_label_parser
from raitap.data.label_parsers.tabular import TabularLabelParser
from raitap.data.types import InputModality

# Module-level (importable qualname) so hydra-zen's ``builds()`` can resolve a
# ``_target_`` for it; a class defined inline inside a test function has no
# importable qualname and ``_register_core`` silently skips ConfigStore
# registration for it (see ``raitap._adapters._register_core``).
_COMPOSED_TARGET = "raitap.data.tests.test_input_parser_registry.DummyComposeInputParser"


@input_parser(registry_name="dummy_compose", schema=InputsConfig)
class DummyComposeInputParser:
    supported_modalities = frozenset({InputModality.text})

    def parse(self, *, source: str) -> list[str]:
        return ["a", "b"]


def _register_inputs_group() -> None:
    """Register the ``data/inputs`` group via the canonical ``register_configs``.

    Mirrors ``configs/tests/test_labels_schema.py::_register_labels_group``:
    uses the same registration path as production (and the rest of the suite)
    rather than a raw ``store.add_to_hydra_store(overwrite_ok=True)``, which
    would clobber other groups' schema nodes.
    """
    from raitap.configs import register_configs

    register_configs()


def test_integration_compose_data_inputs_dummy() -> None:
    """Composing +data/inputs=dummy_compose lands cfg.data.inputs.use, resolvable to the FQN.

    Regression for the Critical review finding: ``DataConfig`` was missing an
    ``inputs`` field, so composing any ``data/inputs`` variant raised
    ``ConfigKeyError: Key 'inputs' not in 'DataConfig'``.
    """
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    from raitap.configs.registry_resolve import resolve_target_fqn

    _register_inputs_group()
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="raitap_schema", overrides=["+data/inputs=dummy_compose"])
    assert cfg.data.inputs.use == "dummy_compose"
    assert resolve_target_fqn("data/inputs", cfg.data.inputs.use) == _COMPOSED_TARGET


def test_decorated_parser_satisfies_protocol_and_registers() -> None:
    @input_parser(registry_name="dummy_text", schema=InputsConfig)
    class DummyParser:
        supported_modalities = frozenset({InputModality.text})

        def parse(self, *, source: str) -> list[str]:
            return ["a", "b"]

    p = DummyParser()
    assert isinstance(p, InputParser)
    assert InputModality.text in p.supported_modalities
    assert p.parse(source="x") == ["a", "b"]


def test_create_label_parser_still_works_after_factory_refactor() -> None:
    """Regression: the shared-core refactor of label_parsers/factory.py must
    preserve existing label-parser instantiation behaviour, now via ``use:``."""
    parser = create_label_parser({"use": "tabular", "source": "x"})
    assert isinstance(parser, TabularLabelParser)


def test_create_label_parser_rejects_target_key() -> None:
    """A ``_target_`` key in the config is rejected (issue #301 RCE surface)."""
    from raitap.configs.registry_resolve import UnsafeConfigTargetError

    with pytest.raises(UnsafeConfigTargetError):
        create_label_parser({"_target_": "TabularLabelParser", "source": "x"})
