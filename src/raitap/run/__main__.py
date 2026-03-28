"""
CLI for an assessment run.

``python -m raitap.run`` executes this module. The ``raitap`` console script
(``pyproject.toml``) calls :func:`main` here via ``raitap.run.__main__:main``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from raitap.run.pipeline import print_summary, run

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path=str(_CONFIG_DIR), config_name="config")
def main(config: AppConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    print_summary(config)
    run(config)

    logger.info("\n%s", "=" * 60)
    logger.info("Assessment complete!")
    logger.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
