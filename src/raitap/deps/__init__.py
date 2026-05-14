"""raitap-deps: infer uv extras from a Hydra config and run uv for the user.

Public API:
    - :func:`infer_extras` — walk a composed Hydra config and return
      ``(set[str], dict[str, str])`` — extras to install plus per-extra origin.
    - :func:`detect_hardware` — probe the host and return ``cpu``/``cuda``/``xpu``.
    - :func:`validate_conflicts` — assert a set of extras does not violate
      ``[tool.uv].conflicts`` declared in ``pyproject.toml``.
    - :func:`render_command` — render the final ``uv sync``/``uv add`` argv.

CLI entry: ``raitap-deps`` (see :mod:`raitap.deps.__main__`).
"""

from raitap.deps.availability import check_platform_availability
from raitap.deps.command import render_command
from raitap.deps.conflicts import validate_conflicts
from raitap.deps.inference import infer_extras
from raitap.deps.probe import detect_hardware
from raitap.deps.python_version import pick_python_version

__all__ = [
    "check_platform_availability",
    "detect_hardware",
    "infer_extras",
    "pick_python_version",
    "render_command",
    "validate_conflicts",
]
