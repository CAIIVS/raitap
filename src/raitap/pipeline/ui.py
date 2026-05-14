"""Pipeline UI helpers — the run banner / summary panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.models import Model


def print_summary(config: AppConfig, model: Model) -> None:
    """Print the rich summary panel before the run starts."""
    from raitap.utils.console import print_summary_panel

    print_summary_panel(config, model)
