"""Package metadata."""

from importlib.metadata import PackageNotFoundError, version


def _package_version() -> str:
    try:
        return version("raitap")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _package_version()
