"""raitap-deps: infer uv extras from a Hydra config and run uv for the user.

The user-facing entry to the auto-deps flow is ``raitap.run(cfg,
auto_install=True)`` for scripts and ``raitap --allow-project-edit`` (alias
``-y``) for the CLI. The functions below are the building blocks both
front-ends compose on top of.

Public API:
    - :func:`infer_extras` — walk a composed Hydra config and return
      ``(set[str], dict[str, str])`` — extras to install plus per-extra origin.
    - :func:`detect_hardware` — probe the host and return ``cpu``/``cuda``/``xpu``.
    - :func:`validate_conflicts` — assert a set of extras does not violate
      ``[tool.uv].conflicts`` declared in ``pyproject.toml``.
    - :func:`render_command` — render the final ``uv sync``/``uv add`` argv.
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
