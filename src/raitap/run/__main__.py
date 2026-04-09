"""
CLI for an assessment run.

``python -m raitap.run`` executes this module. The ``raitap`` console script
(``pyproject.toml``) calls :func:`main` here via ``raitap.run.__main__:main``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from raitap.run.pipeline import run

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


@hydra.main(version_base="1.3", config_path=str(_CONFIG_DIR), config_name="config")
def main(config: AppConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    start_time = time.perf_counter()
    run(config)
    duration = time.perf_counter() - start_time

    logger.info("\n%s", "=" * 60)
    logger.info("Assessment complete!")
    logger.info("Full run duration: %s", _format_duration(duration))
    logger.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
