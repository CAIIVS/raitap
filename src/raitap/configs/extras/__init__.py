"""raitap-deps: infer uv extras from a Hydra config and run uv for the user.

Public API:
    - :func:`infer_extras` — walk a composed Hydra config and return
      ``(set[str], dict[str, str])`` — extras to install plus per-extra origin.
    - :func:`detect_hardware` — probe the host and return ``cpu``/``cuda``/``xpu``.
    - :func:`validate_conflicts` — assert a set of extras does not violate
      ``[tool.uv].conflicts`` declared in ``pyproject.toml``.
    - :func:`render_command` — render the final ``uv sync``/``uv add`` argv.

CLI entry: ``raitap-deps`` (see :mod:`raitap.configs.extras.__main__`).
"""

from raitap.configs.extras.command import render_command
from raitap.configs.extras.conflicts import validate_conflicts
from raitap.configs.extras.inference import infer_extras
from raitap.configs.extras.probe import detect_hardware
from raitap.configs.extras.python_version import pick_python_version

__all__ = [
    "detect_hardware",
    "infer_extras",
    "pick_python_version",
    "render_command",
    "validate_conflicts",
]
