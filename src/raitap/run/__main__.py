"""
CLI for an assessment run.

``python -m raitap.run`` executes this module. The ``raitap`` console script
(``pyproject.toml``) calls :func:`main` here via ``raitap.run.__main__:main``.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from raitap.run.pipeline import run
from raitap.utils.console import (
    print_complete_panel,
    print_failure_panel,
    setup_logging,
)

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig

_CONFIG_NAME_FLAGS = ("--config-name", "-cn")
_CONFIG_LOCATION_FLAGS = ("--config-path", "-cp", "--config-dir", "-cd")
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
_DEFAULT_CONFIG_NAME = "config"
logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _hydra_flag_value(argv: list[str], flags: tuple[str, ...]) -> str | None:
    for index, arg in enumerate(argv):
        for flag in flags:
            if arg == flag:
                if index + 1 < len(argv):
                    return argv[index + 1]
                return None
            prefix = f"{flag}="
            if arg.startswith(prefix):
                return arg.removeprefix(prefix)
    return None


def _has_hydra_flag(argv: list[str], flags: tuple[str, ...]) -> bool:
    return _hydra_flag_value(argv, flags) is not None


def _prepare_cli_argv(argv: list[str]) -> list[str]:
    config_name = _hydra_flag_value(argv, _CONFIG_NAME_FLAGS)
    if config_name in {None, "", _DEFAULT_CONFIG_NAME}:
        return argv
    if _has_hydra_flag(argv, _CONFIG_LOCATION_FLAGS):
        return argv
    if not argv:
        return argv
    return [argv[0], "--config-dir", str(Path.cwd()), *argv[1:]]


@hydra.main(version_base="1.3", config_path=str(_CONFIG_DIR), config_name=_DEFAULT_CONFIG_NAME)
def _hydra_main(config: AppConfig) -> None:
    setup_logging(level=logging.INFO)
    start_time = time.perf_counter()
    try:
        run(config)
    except Exception as exc:
        duration = _format_duration(time.perf_counter() - start_time)
        print_failure_panel(exc, duration)
        raise
    duration = _format_duration(time.perf_counter() - start_time)
    print_complete_panel(duration)


def main() -> None:
    sys.argv = _prepare_cli_argv(list(sys.argv))
    _hydra_main()


if __name__ == "__main__":
    main()
