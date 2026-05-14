"""Hydra SearchPathPlugin: append bundled ``pkg://raitap.configs`` to the search path.

Without this, user configs loaded via ``--config-dir`` would not see bundled
group presets (e.g. ``reporting=html``) because Hydra's primary config_path
points at the user's directory. The plugin makes the bundled groups
discoverable from any external config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra.plugins.search_path_plugin import SearchPathPlugin

if TYPE_CHECKING:
    from hydra.core.config_search_path import ConfigSearchPath


class RaitapSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="raitap", path="pkg://raitap.configs")
