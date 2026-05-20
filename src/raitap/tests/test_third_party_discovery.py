"""End-to-end: a pip-installed entry-point plugin registers an adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_FAKE = Path(__file__).parent / "_fake_plugin"


@pytest.fixture(scope="module")
def _installed_fake_plugin():  # noqa: ANN202
    subprocess.run(["uv", "pip", "install", str(_FAKE)], check=True)
    yield
    subprocess.run(["uv", "pip", "uninstall", "raitap-fakeplugin"], check=True)


def test_plugin_adapter_registered(_installed_fake_plugin) -> None:  # noqa: ANN001
    from raitap._adapters import _BUILDERS, discover_third_party_adapters

    discover_third_party_adapters()
    assert "fakeattack" in _BUILDERS["robustness"]


def test_disabled_env_skips(monkeypatch, _installed_fake_plugin) -> None:  # noqa: ANN001
    from raitap._adapters import _BUILDERS, discover_third_party_adapters

    _BUILDERS.get("robustness", {}).pop("fakeattack", None)
    monkeypatch.setenv("RAITAP_DISABLE_PLUGINS", "1")
    discover_third_party_adapters()
    assert "fakeattack" not in _BUILDERS.get("robustness", {})


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
