"""End-to-end: a pip-installed entry-point plugin registers an adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_FAKE = Path(__file__).parent / "_fake_plugin"
_FAKE_BROKEN = Path(__file__).parent / "_fake_broken_plugin"
# The user-facing reference plugin (docs/contributor/writing-a-plugin.md
# points here), vs. ``_FAKE`` which is a hidden test-only fixture.
_EXAMPLE_PLUGIN = Path(__file__).parents[3] / "example-plugin"


@pytest.fixture(scope="module")
def _installed_fake_plugin():  # noqa: ANN202
    subprocess.run(["uv", "pip", "install", str(_FAKE)], check=True)
    yield
    subprocess.run(["uv", "pip", "uninstall", "raitap-fakeplugin"], check=True)


@pytest.fixture(scope="module")
def _installed_example_plugin():  # noqa: ANN202
    subprocess.run(["uv", "pip", "install", str(_EXAMPLE_PLUGIN)], check=True)
    yield
    subprocess.run(["uv", "pip", "uninstall", "raitap-example-plugin"], check=True)


@pytest.fixture
def _installed_broken_plugin():  # noqa: ANN202
    subprocess.run(["uv", "pip", "install", str(_FAKE_BROKEN)], check=True)
    yield
    subprocess.run(["uv", "pip", "uninstall", "raitap-fakebrokenplugin"], check=True)


def test_plugin_adapter_registered(_installed_fake_plugin) -> None:  # noqa: ANN001
    import raitap._adapters as adapters_mod

    adapters_mod.discover_third_party_adapters()
    assert "fakeattack" in adapters_mod._BUILDERS["robustness"]


def test_plugin_use_key_resolves(_installed_fake_plugin) -> None:  # noqa: ANN001
    """The #301 ``use:`` seam resolves a plugin-registered adapter exactly like
    an in-tree one — the trusted registry (:data:`raitap._adapters._TARGET_FQN`)
    is populated by the same decorator regardless of where the class lives."""
    import raitap._adapters as adapters_mod
    from raitap.configs.registry_resolve import resolve_target_fqn

    adapters_mod.discover_third_party_adapters()
    fqn = resolve_target_fqn("robustness", "fakeattack")
    assert fqn.endswith("FakeAttackAssessor")


def test_config_schema_includes_installed_plugin(_installed_fake_plugin) -> None:  # noqa: ANN001
    """A plugin's ``use:`` key shows up in the generated JSON Schema enum, so a
    YAML editor pointed at it autocompletes plugin adapters like first-party
    ones. :func:`build_config_schema` runs full discovery itself."""
    from raitap._config_schema import build_config_schema

    schema = build_config_schema()
    enum = schema["properties"]["robustness"]["additionalProperties"]["properties"]["use"]["enum"]
    assert "fakeattack" in enum


def test_plugin_adapter_runs_through_factory(_installed_fake_plugin) -> None:  # noqa: ANN001
    """A plugin-registered assessor resolves and instantiates through the real
    ``raitap.robustness.factory.create_assessor`` path — the same one the
    pipeline uses for in-tree adapters — proving ``use:`` isn't just
    registry-visible but actually runnable end-to-end."""
    import raitap._adapters as adapters_mod
    from raitap.robustness.factory import create_assessor

    adapters_mod.discover_third_party_adapters()
    assessor, target_fqn = create_assessor({"use": "fakeattack", "algorithm": "PGD"})

    assert type(assessor).__name__ == "FakeAttackAssessor"
    assert target_fqn.endswith("FakeAttackAssessor")


def test_example_plugin_end_to_end(_installed_example_plugin) -> None:  # noqa: ANN001
    """CI proof for the user-facing ``example-plugin/`` (the runnable reference
    linked from docs/contributor/writing-a-plugin.md): install it the way a
    real consumer would, then drive its README's ``use: identity_attack``
    example through the real ``raitap.robustness.factory`` path."""
    import raitap._adapters as adapters_mod
    from raitap.robustness.factory import create_assessor

    adapters_mod.discover_third_party_adapters()
    assessor, target_fqn = create_assessor({"use": "identity_attack", "algorithm": "identity"})

    assert type(assessor).__name__ == "IdentityAttackAssessor"
    assert target_fqn == "raitap_example_plugin.IdentityAttackAssessor"


def test_disabled_env_skips(monkeypatch, _installed_fake_plugin) -> None:  # noqa: ANN001
    import raitap._adapters as adapters_mod

    adapters_mod._BUILDERS.get("robustness", {}).pop("fakeattack", None)
    monkeypatch.setenv("RAITAP_DISABLE_PLUGINS", "1")
    adapters_mod.discover_third_party_adapters()
    assert "fakeattack" not in adapters_mod._BUILDERS.get("robustness", {})


def test_crashing_plugin_is_isolated(_installed_broken_plugin) -> None:  # noqa: ANN001
    """A plugin that raises at import is caught, warned about, and skipped —
    discovery completes and in-tree adapters stay resolvable."""
    from raitap.configs import register_configs

    register_configs()  # ensure in-tree adapters are registered

    import raitap._adapters as adapters_mod

    # The broken plugin's import raises; discovery must not propagate it, and
    # must emit a warning naming the plugin.
    with pytest.warns(UserWarning, match="fakebroken"):
        adapters_mod.discover_third_party_adapters()

    assert "fakebroken" not in adapters_mod._BUILDERS.get("robustness", {})
    # In-tree adapter still resolvable — one bad plugin didn't break the registry.
    assert "torchattacks" in adapters_mod._BUILDERS["robustness"]


def test_version_skew_skips(monkeypatch) -> None:  # noqa: ANN001
    import raitap._adapters as adapters_mod

    monkeypatch.setattr(adapters_mod, "_plugin_raitap_specifier", lambda _d: ">=999,<1000")
    ok, why = adapters_mod._plugin_version_ok("whatever")
    assert ok is False
    assert "requires raitap" in why


def test_no_pin_is_malformed(monkeypatch) -> None:  # noqa: ANN001
    import raitap._adapters as adapters_mod

    monkeypatch.setattr(adapters_mod, "_plugin_raitap_specifier", lambda _d: None)
    ok, why = adapters_mod._plugin_version_ok("whatever")
    assert ok is False
    assert "no 'raitap' dependency pin" in why
