"""Tests for :mod:`raitap.utils.diagnostics`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from raitap.utils.diagnostics import (
    Diagnostic,
    docs_url,
    is_dev_install,
    resolve_diagnostic_from_frames,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    is_dev_install.cache_clear()
    yield
    is_dev_install.cache_clear()


class TestResolveDiagnosticFromFrames:
    def test_returns_default_when_no_raitap_frame(self) -> None:
        """When no qualifying raitap frame is on the stack, defaults pass through."""
        # ``raitap/utils/`` frames are filtered out, so this call site does not
        # contribute a subsystem.
        diag = resolve_diagnostic_from_frames("/external/lib.py", 42)
        assert diag.file == "/external/lib.py"
        assert diag.line == 42
        assert diag.subsystem is None

    def test_detects_third_party_from_default_path(self) -> None:
        diag = resolve_diagnostic_from_frames("/site-packages/captum/attr/foo.py", 10)
        assert diag.third_party_lib == "captum"

    def test_detects_third_party_with_windows_separators(self) -> None:
        diag = resolve_diagnostic_from_frames(r"C:\foo\captum\bar.py", 1)
        assert diag.third_party_lib == "captum"

    def test_classify_subsystem_handles_known_paths(self) -> None:
        """Spot-check the subsystem regex via the module-private helper."""
        from raitap.utils.diagnostics import _classify_subsystem

        assert _classify_subsystem("/x/raitap/metrics/foo.py") == "metrics"
        assert _classify_subsystem(r"C:\x\raitap\transparency\bar.py") == "transparency"
        assert _classify_subsystem("/x/raitap/utils/console.py") == "utils"
        assert _classify_subsystem("/no/raitap/here.py") is None

    def test_classify_subsystem_rejects_nested_raitap_dir_names(self) -> None:
        """CI checkouts at ``/work/raitap/raitap/.venv/...`` must not match
        ``raitap`` itself as a subsystem when the first ``raitap/`` segment is
        followed by another ``raitap/`` (the package directory inside the repo)."""
        from raitap.utils.diagnostics import _classify_subsystem

        assert (
            _classify_subsystem(
                "/home/runner/work/raitap/raitap/.venv/lib/python3.13/"
                "site-packages/_pytest/python.py"
            )
            is None
        )


class TestIsDevInstall:
    def test_true_when_file_outside_site_packages(self) -> None:
        with patch("raitap.__file__", "/home/dev/raitap/src/raitap/__init__.py"):
            assert is_dev_install() is True

    def test_false_when_file_under_site_packages(self) -> None:
        with patch("raitap.__file__", "/usr/lib/python3.13/site-packages/raitap/__init__.py"):
            assert is_dev_install() is False

    def test_handles_windows_paths(self) -> None:
        with patch(
            "raitap.__file__",
            r"C:\venv\Lib\site-packages\raitap\__init__.py",
        ):
            assert is_dev_install() is False

    def test_dist_packages_treated_as_installed(self) -> None:
        """Debian/Ubuntu system Python wheels live under ``dist-packages/``."""
        with patch("raitap.__file__", "/usr/lib/python3/dist-packages/raitap/__init__.py"):
            assert is_dev_install() is False

    def test_case_insensitive_site_packages(self) -> None:
        """Windows venv paths sometimes appear as ``Lib/Site-Packages``."""
        with patch(
            "raitap.__file__",
            r"C:\Venv\Lib\Site-Packages\raitap\__init__.py",
        ):
            assert is_dev_install() is False


class TestDocsUrl:
    def test_raitap_subsystem_returns_module_url(self) -> None:
        diag = Diagnostic(subsystem="metrics", file="/x.py", line=1, third_party_lib=None)
        assert docs_url(diag) == "https://caiivs.github.io/raitap/modules/metrics/"

    def test_third_party_returns_frameworks_page(self) -> None:
        diag = Diagnostic(subsystem="transparency", file="/x.py", line=1, third_party_lib="captum")
        assert (
            docs_url(diag)
            == "https://caiivs.github.io/raitap/modules/transparency/frameworks-and-libraries.html"
        )

    def test_unknown_subsystem_returns_none(self) -> None:
        diag = Diagnostic(subsystem="random_thing", file="/x.py", line=1, third_party_lib=None)
        assert docs_url(diag) is None

    def test_no_subsystem_returns_none(self) -> None:
        diag = Diagnostic(subsystem=None, file="/x.py", line=1, third_party_lib=None)
        assert docs_url(diag) is None
