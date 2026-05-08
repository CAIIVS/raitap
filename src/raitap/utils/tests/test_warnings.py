"""Tests for :mod:`raitap.utils.warnings`."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

import raitap.utils.warnings as warnings_module
from raitap.utils.warnings import (
    WarningOrigin,
    docs_url,
    is_dev_install,
    resolve_warn_origin,
    suppress_warning,
)


@pytest.fixture(autouse=True)
def _clear_origins() -> Iterator[None]:
    warnings_module._clear_origins()
    is_dev_install.cache_clear()
    yield
    warnings_module._clear_origins()
    is_dev_install.cache_clear()


class TestResolveWarnOrigin:
    def test_returns_default_when_no_raitap_frame(self) -> None:
        """When no qualifying raitap frame is on the stack, defaults pass through."""
        # ``raitap/utils/`` frames are filtered out by resolve_warn_origin, so this
        # call site does not contribute a subsystem.
        origin = resolve_warn_origin("/external/lib.py", 42)
        assert origin.file == "/external/lib.py"
        assert origin.line == 42
        assert origin.subsystem is None

    def test_detects_third_party_from_default_path(self) -> None:
        origin = resolve_warn_origin("/site-packages/captum/attr/foo.py", 10)
        assert origin.third_party_lib == "captum"

    def test_detects_third_party_with_windows_separators(self) -> None:
        origin = resolve_warn_origin(r"C:\foo\captum\bar.py", 1)
        assert origin.third_party_lib == "captum"

    def test_classify_subsystem_handles_known_paths(self) -> None:
        """Spot-check the subsystem regex via the module-private helper."""
        from raitap.utils.warnings import _classify_subsystem

        assert _classify_subsystem("/x/raitap/metrics/foo.py") == "metrics"
        assert _classify_subsystem(r"C:\x\raitap\transparency\bar.py") == "transparency"
        assert _classify_subsystem("/x/raitap/utils/console.py") == "utils"
        assert _classify_subsystem("/no/raitap/here.py") is None

    def test_classify_subsystem_rejects_nested_raitap_dir_names(self) -> None:
        """CI checkouts at ``/work/raitap/raitap/.venv/...`` must not match
        ``raitap`` itself as a subsystem when the first ``raitap/`` segment is
        followed by another ``raitap/`` (the package directory inside the repo)."""
        from raitap.utils.warnings import _classify_subsystem

        assert (
            _classify_subsystem(
                "/home/runner/work/raitap/raitap/.venv/lib/python3.13/"
                "site-packages/_pytest/python.py"
            )
            is None
        )


class TestSuppressWarning:
    def test_suppresses_matching_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            suppress_warning(message=r"silence-me", category=UserWarning)
            warnings.warn("silence-me please", UserWarning, stacklevel=1)
            warnings.warn("but-not-this", UserWarning, stacklevel=1)
        messages = [str(w.message) for w in caught]
        assert "silence-me please" not in messages
        assert "but-not-this" in messages

    def test_default_category_is_user_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            suppress_warning(message=r"hush")
            warnings.warn("hush now", UserWarning, stacklevel=1)
        assert all("hush now" not in str(w.message) for w in caught)


class TestIsDevInstall:
    def test_true_when_file_outside_site_packages(self) -> None:
        is_dev_install.cache_clear()
        with patch("raitap.__file__", "/home/dev/raitap/src/raitap/__init__.py"):
            assert is_dev_install() is True

    def test_false_when_file_under_site_packages(self) -> None:
        is_dev_install.cache_clear()
        with patch("raitap.__file__", "/usr/lib/python3.13/site-packages/raitap/__init__.py"):
            assert is_dev_install() is False

    def test_handles_windows_paths(self) -> None:
        is_dev_install.cache_clear()
        with patch(
            "raitap.__file__",
            r"C:\venv\Lib\site-packages\raitap\__init__.py",
        ):
            assert is_dev_install() is False


class TestDocsUrl:
    def test_raitap_subsystem_returns_module_url(self) -> None:
        origin = WarningOrigin(subsystem="metrics", file="/x.py", line=1, third_party_lib=None)
        assert docs_url(origin) == "https://caiivs.github.io/raitap/modules/metrics/"

    def test_third_party_returns_frameworks_page(self) -> None:
        origin = WarningOrigin(
            subsystem="transparency", file="/x.py", line=1, third_party_lib="captum"
        )
        assert (
            docs_url(origin)
            == "https://caiivs.github.io/raitap/modules/transparency/frameworks-and-libraries.html"
        )

    def test_unknown_subsystem_returns_none(self) -> None:
        origin = WarningOrigin(subsystem="random_thing", file="/x.py", line=1, third_party_lib=None)
        assert docs_url(origin) is None

    def test_no_subsystem_returns_none(self) -> None:
        origin = WarningOrigin(subsystem=None, file="/x.py", line=1, third_party_lib=None)
        assert docs_url(origin) is None
