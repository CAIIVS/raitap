"""Single framed status panel — ``StatusFrame``.

Wraps Rich's :class:`~rich.panel.Panel` with the raitap header convention:
``icon label · chip · chip · …``. One class covers warning, error, failure
(an :attr:`Status.ERROR` with a custom label), and completion panels —
they differ only by :class:`Status`, label, and chip composition.

Renderers compose chips via the :func:`chip` helper, which returns Rich
:class:`~rich.text.Text` instances with pre-resolved :class:`~rich.style.Style`
attributes (colour + link + underline). No theme-name strings cross this
boundary, so Rich's "compound style strings bypass theme lookup" footgun
can't bite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from raitap.utils.colour import Status, colour

if TYPE_CHECKING:
    from collections.abc import Sequence


def chip(
    label: str,
    *,
    style: Style,
    link: str | None = None,
    underline: bool = False,
) -> Text:
    """Build a ``· label`` chip with the given Rich :class:`Style`.

    ``link`` adds an OSC 8 hyperlink without breaking the colour;
    ``underline`` toggles the underline attribute (used for ``View docs``
    affordances). The leading ``· `` separator is part of the chip so chips
    compose by simple concatenation.
    """
    chip_style = style
    if underline:
        chip_style = chip_style + Style(underline=True)
    if link is not None:
        chip_style = chip_style + Style(link=link)
    return Text(f"· {label}", style=chip_style)


@dataclass(frozen=True)
class StatusFrame:
    """A framed status event: warning, error, completion, or similar.

    Renderers pick a :class:`Status` and supply the body; the frame handles
    icon, default label, border colour, and title composition. ``label``
    overrides the status's default label (e.g. ``"Failure"`` for a
    top-level :attr:`Status.ERROR`).
    """

    status: Status
    body: Text
    label: str | None = None
    chips: Sequence[Text] = field(default_factory=tuple)
    padding: tuple[int, int] = (1, 2)

    def render(self) -> Panel:
        shades = colour(self.status)
        label = self.label or self.status.default_label
        # Don't set ``title.style`` on the Text: Panel.__rich_console__ calls
        # ``text.stylize(text.style)`` over the entire title, which would push
        # the base style over our chip spans and lose the lighter shade.
        # Apply styles per-span instead.
        title = Text()
        title.append(f"{self.status.icon}{label}", style=shades.base)
        for c in self.chips:
            title.append(" ")
            title.append_text(c)
        title.overflow = "ellipsis"
        title.no_wrap = True
        return Panel(
            self.body,
            title=title,
            title_align="left",
            border_style=shades.base,
            padding=self.padding,
        )


__all__ = ["StatusFrame", "chip"]
