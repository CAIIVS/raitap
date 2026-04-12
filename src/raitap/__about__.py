"""Package metadata."""

from importlib.metadata import PackageNotFoundError, version


def _package_version() -> str:
    try:
        return version("raitap")
    except PackageNotFoundError:
        # Editable / raw checkout without install: avoid implying a released 0.0.0
        # when pyproject may be 0.0.1+.
        return "0+unknown"


__version__ = _package_version()
