"""End-to-end: a pip-installed entry-point plugin registers an adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_FAKE = Path(__file__).parent / "_fake_plugin"
_FAKE_BROKEN = Path(__file__).parent / "_fake_broken_plugin"


@pytest.fixture(scope="module")
def _installed_fake_plugin():  # noqa: ANN202
    subprocess.run(["uv", "pip", "install", str(_FAKE)], check=True)
    yield
    subprocess.run(["uv", "pip", "uninstall", "raitap-fakeplugin"], check=True)


@pytest.fixture
def _installed_broken_plugin():  # noqa: ANN202
    subprocess.run(["uv", "pip", "install", str(_FAKE_BROKEN)], check=True)
    yield
    subprocess.run(["uv", "pip", "uninstall", "raitap-fakebrokenplugin"], check=True)


def test_plugin_adapter_registered(_installed_fake_plugin) -> None:  # noqa: ANN001
    import raitap._adapters as adapters_mod

    adapters_mod.discover_third_party_adapters()
    assert "fakeattack" in adapters_mod._BUILDERS["robustness"]


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
