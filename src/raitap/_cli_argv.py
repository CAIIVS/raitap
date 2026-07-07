"""Stdlib-only Hydra-flag argv helpers shared by the two CLI entry points.

``cli.py`` runs *before* the deps bootstrap, so this module must not import
torch or any heavy raitap module — ``pathlib`` only.
"""

from __future__ import annotations

from pathlib import Path

DEMO_FLAG = "--demo"
CONFIG_NAME_FLAGS = ("--config-name", "-cn")
CONFIG_LOCATION_FLAGS = ("--config-path", "-cp", "--config-dir", "-cd")
CONFIG_FLAGS = (*CONFIG_NAME_FLAGS, *CONFIG_LOCATION_FLAGS)
HYDRA_INTROSPECTION_FLAGS = (
    "--help",
    "-h",
    "--hydra-help",
    "--cfg",
    "--info",
    "--version",
)
DEFAULT_CONFIG_NAME = "demo"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def find_flag_value(argv: list[str], flags: tuple[str, ...]) -> str | None:
    for index, arg in enumerate(argv):
        for flag in flags:
            if arg == flag:
                return argv[index + 1] if index + 1 < len(argv) else None
            prefix = f"{flag}="
            if arg.startswith(prefix):
                return arg.removeprefix(prefix)
    return None


def has_flag(argv: list[str], flags: tuple[str, ...]) -> bool:
    return find_flag_value(argv, flags) is not None


def rewrite_demo(argv: list[str]) -> list[str]:
    """Rewrite ``--demo`` to ``--config-name demo`` (bundled config is registered)."""
    if DEMO_FLAG not in argv:
        return argv
    stripped = [arg for arg in argv if arg != DEMO_FLAG]
    if not stripped:
        return argv
    return [stripped[0], "--config-name", "demo", *stripped[1:]]


def needs_help_frame(argv: list[str]) -> bool:
    """True when invoked with no actionable config selector and no Hydra introspection."""
    if not argv:
        return True
    if any(arg in HYDRA_INTROSPECTION_FLAGS for arg in argv):
        return False
    if DEMO_FLAG in argv:
        return False
    for arg in argv:
        for flag in CONFIG_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                return False
    return True


def inject_config_dir(argv: list[str]) -> list[str]:
    """Inject ``--config-dir <cwd>`` for a custom ``--config-name`` without a location flag."""
    config_name = find_flag_value(argv, CONFIG_NAME_FLAGS)
    if config_name in {None, "", DEFAULT_CONFIG_NAME}:
        return argv
    if has_flag(argv, CONFIG_LOCATION_FLAGS):
        return argv
    if not argv:
        return argv
    return [argv[0], "--config-dir", str(Path.cwd()), *argv[1:]]
