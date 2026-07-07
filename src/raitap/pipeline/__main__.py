"""
CLI for an assessment run.

``python -m raitap.pipeline`` executes this module. The ``raitap`` console script
(``pyproject.toml``) calls :func:`main` here via ``raitap.pipeline.__main__:main``.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import TYPE_CHECKING

import hydra

from raitap import _cli_argv
from raitap.pipeline.orchestrator import _run_pipeline
from raitap.utils.console import (
    print_complete_panel,
    print_failure_panel,
    setup_logging,
)
from raitap.utils.errors import RaitapError

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig

logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


@hydra.main(
    version_base="1.3",
    config_path=str(_cli_argv.CONFIG_DIR),
    config_name=_cli_argv.DEFAULT_CONFIG_NAME,
)
def _hydra_main(config: AppConfig) -> None:
    setup_logging(level=logging.INFO)
    start_time = time.perf_counter()
    try:
        _run_pipeline(config)
    except Exception as exc:
        duration = _format_duration(time.perf_counter() - start_time)
        print_failure_panel(exc, duration)
        # RaitapError already carries a user-actionable message + diagnostic
        # chips in the failure panel; the raw traceback is noise on top of
        # that, so exit cleanly. Non-raitap exceptions keep propagating so
        # Hydra/rich_traceback can still surface the full stack for triage.
        if isinstance(exc, RaitapError):
            sys.exit(1)
        raise
    duration = _format_duration(time.perf_counter() - start_time)
    print_complete_panel(duration)


def _dispatch_subcommand(argv: list[str]) -> bool:
    """Handle non-Hydra subcommands. Return True if a subcommand ran."""
    if argv[:2] == ["tracking", "stop"]:
        from raitap.tracking import run_stop_command

        setup_logging(level=logging.INFO)
        run_stop_command()
        return True
    return False


def main() -> None:
    if _dispatch_subcommand(sys.argv[1:]):
        return
    sys.argv = _cli_argv.inject_config_dir(list(sys.argv))
    _hydra_main()


if __name__ == "__main__":
    main()
