"""Raitap colour palette + Rich theme.

Two layers:

- :data:`BASE_TOKENS` — the raw palette. Each hue exposes a ``_base`` shade
  (used for borders / primary text) and a ``_light`` shade (used for
  secondary chips and dimmed accents). Adding or rebalancing a colour means
  editing one row here; nothing downstream hard-codes ANSI names.
- :data:`SEMANTIC_TOKENS` — semantic names (``logging.level.warning``,
  ``log.time``, …) that reference base tokens. Renderers should refer to
  these where the *meaning* is what matters; base tokens are for places where
  a hue choice is intentional (a warning panel using ``yellow_base``).

:data:`THEME` is the Rich :class:`~rich.theme.Theme` consumed by the console
factories — it merges both layers (with semantic tokens dereferenced to raw
ANSI colours, since Rich can't chase references inside theme values).

Naming convention: ``<hue>_<shade>``. Underscores rather than dots because
Rich's style parser can't carry dots through ``[name]`` markup lookups.
"""

from __future__ import annotations

from typing import Final

from rich.theme import Theme

# Raw palette. Two shades per hue keeps the visual hierarchy explicit:
# ``_base`` for the dominant element of a region (border, headline),
# ``_light`` for chips / metadata that should recede slightly.
BASE_TOKENS: Final[dict[str, str]] = {
    "red_base": "red",
    "red_light": "bright_red",
    "yellow_base": "yellow",
    "yellow_light": "bright_yellow",
    "green_base": "green",
    "green_light": "bright_green",
    "cyan_base": "cyan",
    "cyan_light": "bright_cyan",
}

# Semantic aliases. Values are base-token *names*; the resolver below maps
# them to raw ANSI colours when constructing the Rich theme. Renderers can
# still reference base tokens directly when the hue choice is intentional.
SEMANTIC_TOKENS: Final[dict[str, str]] = {
    "log.time": "cyan_base",
    "log.message": "default",  # passthrough: keep the terminal's foreground.
    "logging.level.info": "cyan_base",
    "logging.level.warning": "yellow_light",
    "logging.level.error": "red_base",
}


def _resolve(token: str) -> str:
    """Return the raw ANSI colour for a token (base token or ``default``)."""
    if token == "default":
        return token
    return BASE_TOKENS[token]


THEME: Final[Theme] = Theme(
    {
        **BASE_TOKENS,
        **{name: _resolve(ref) for name, ref in SEMANTIC_TOKENS.items()},
    }
)


__all__ = ["BASE_TOKENS", "SEMANTIC_TOKENS", "THEME"]
