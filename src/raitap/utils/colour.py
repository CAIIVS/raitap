"""Raitap colour palette.

Two layers:

- :class:`Status` — the semantic axis. ``WARNING``, ``ERROR``, ``SUCCESS``,
  ``INFO``. Each carries a hue plus the conventional icon/label used in
  framed status output. Renderers should talk in terms of ``Status``, not
  ANSI colour names.
- :func:`colour` — resolves a :class:`Status` to a :class:`Shades` pair of
  Rich :class:`~rich.style.Style` instances (``base`` for the dominant
  region, ``light`` for secondary chips). :class:`Style` composes via ``+``,
  which is what makes ``colour(Status.ERROR).light + Style(link=url)``
  build cleanly without string concatenation.

:data:`THEME` keeps a tiny set of ``logging.level.*`` aliases so the Rich
log formatter colours the level prefix consistently with the Status axis.
Most rendering should hold :class:`Style` instances directly rather than
referencing theme names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from rich.style import Style
from rich.theme import Theme


class Status(Enum):
    """Semantic axis for status output.

    Each member binds a hue, a leading glyph, and the canonical label that
    appears in framed status output. Callers override ``default_label`` per
    site when the wording needs to differ (e.g. ``"Failure"`` for the
    top-level crash panel even though it's an :attr:`ERROR` underneath).
    """

    WARNING = ("yellow", "△ ", "Warning")
    ERROR = ("red", "✕ ", "Error")
    SUCCESS = ("green", "✓ ", "Complete")
    INFO = ("cyan", "▷ ", "Info")

    def __init__(self, hue: str, icon: str, default_label: str) -> None:
        self.hue: str = hue
        self.icon: str = icon
        self.default_label: str = default_label


@dataclass(frozen=True)
class Shades:
    """Two Rich :class:`~rich.style.Style` shades for a :class:`Status`.

    ``base`` for borders / headlines / dominant text, ``light`` for
    secondary chips and metadata that should recede slightly.
    """

    base: Style
    light: Style


def colour(status: Status) -> Shades:
    """Resolve a :class:`Status` to a :class:`Shades` pair."""
    return Shades(
        base=Style.parse(status.hue),
        light=Style.parse(f"bright_{status.hue}"),
    )


# Rich theme: only ``logging.level.*`` aliases so the default level prefix
# rendering matches the Status axis. Inline styling goes through
# :func:`colour` (Style objects), not theme names — Rich's parser doesn't
# resolve theme names inside compound style strings, so keeping the theme
# small avoids a class of bugs.
THEME: Final[Theme] = Theme(
    {
        "log.time": Status.INFO.hue,
        "log.message": "default",
        "logging.level.info": Status.INFO.hue,
        "logging.level.warning": f"bright_{Status.WARNING.hue}",
        "logging.level.error": Status.ERROR.hue,
    }
)


__all__ = ["THEME", "Shades", "Status", "colour"]
