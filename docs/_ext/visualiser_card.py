"""``visualiser-card`` directive — one consistent block per visualiser.

Enforces the house structure for every visualiser entry in the robustness and
transparency docs: heading, intro, a **required** "How to read it" guide, an
optional kwargs table, optional notes, a compatibility line, and a preview
image. ``:how-to-read:`` being required means a visualiser can't be documented
without telling the reader what the axes are and how to read the figure.

Usage (colon fence so inner ``` code blocks nest cleanly)::

    :::::{visualiser-card}
    :name: OutputBoundsPinnedVisualiser
    :registry: output_bounds_pinned
    :intro: One sub-plot per pinned sample ...
    :how-to-read: The x-axis is the certified value ...
    :wraps: `captum.attr.visualization.visualize_image_attr`
    :kwarg: max_samples
    :default: `4`
    :meaning: Maximum number of samples to render.
    :kwarg: max_classes
    :default: `20`
    :meaning: Classes drawn per sub-plot ...
    :compat: Supports `AssessmentKind.FORMAL_VERIFICATION`.
    :notes: Optional extra prose; may contain fenced code blocks.
    :::::

``:preview:`` overrides the image path; otherwise it is derived from
``:registry:`` as ``../../_static/visualisers/<registry>_visualiser.png``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

if TYPE_CHECKING:
    from sphinx.application import Sphinx

# Single-value page fields (joined to one line) and block page fields (raw
# lines preserved, so prose can span paragraphs and embed code fences).
_INLINE_FIELDS = ("name", "registry", "wraps", "preview")
_BLOCK_FIELDS = ("intro", "how-to-read", "compat", "notes")
_PAGE_FIELDS = _INLINE_FIELDS + _BLOCK_FIELDS
_KWARG_FIELDS = ("kwarg", "default", "meaning")
_REQUIRED = ("name", "intro", "how-to-read")
_KWARG_HEADERS = ("Kwarg", "Default", "Meaning")


@dataclass
class _Card:
    page: dict[str, list[str]] = field(default_factory=lambda: {f: [] for f in _PAGE_FIELDS})
    kwargs: list[dict[str, str]] = field(default_factory=list)


def _parse_marker(line: str) -> tuple[str, str] | None:
    if not line.startswith(":"):
        return None
    end = line.find(":", 1)
    if end == -1:
        return None
    name = line[1:end]
    if name not in _PAGE_FIELDS and name not in _KWARG_FIELDS:
        raise ValueError(
            f"`visualiser-card`: unknown field `:{name}:`. Allowed: "
            + ", ".join(f"`:{f}:`" for f in (*_PAGE_FIELDS, *_KWARG_FIELDS))
        )
    return name, line[end + 1 :].strip()


def _parse(lines: list[str]) -> _Card:
    card = _Card()
    current_page: str | None = None
    current_kwarg: dict[str, str] | None = None

    def close_kwarg() -> None:
        nonlocal current_kwarg
        if current_kwarg is not None:
            missing = [f for f in _KWARG_FIELDS if not current_kwarg.get(f)]
            if missing:
                raise ValueError(
                    "`visualiser-card`: kwarg entry missing "
                    + ", ".join(f"`:{m}:`" for m in missing)
                )
            card.kwargs.append(current_kwarg)
            current_kwarg = None

    for raw in lines:
        marker = _parse_marker(raw.strip())
        if marker is not None:
            name, value = marker
            if name in _PAGE_FIELDS:
                close_kwarg()
                current_page = name
                card.page[name].append(value)
            elif name == "kwarg":
                close_kwarg()
                current_page = None
                current_kwarg = {"kwarg": value, "default": "", "meaning": ""}
            else:  # default / meaning
                if current_kwarg is None:
                    raise ValueError("`visualiser-card`: `:default:`/`:meaning:` need a `:kwarg:`.")
                current_page = None
                current_kwarg[name] = value
            continue
        if current_page in _BLOCK_FIELDS:
            card.page[current_page].append(raw)  # preserve blanks + fences.
        elif not raw.strip():
            close_kwarg()
    close_kwarg()

    for name in _REQUIRED:
        if not any(s.strip() for s in card.page[name]):
            raise ValueError(f"`visualiser-card`: missing required `:{name}:`.")
    return card


def _inline(values: list[str]) -> str:
    return " ".join(v.strip() for v in values if v.strip()).strip()


def _block(values: list[str]) -> list[str]:
    out = list(values)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def _literal(value: str) -> str:
    stripped = value.strip()
    if "`" in stripped or re.search(r"\[[^\]]+\]\([^)]+\)", stripped):
        return stripped  # already has code spans / links — leave as authored.
    return f"`{stripped}`"


def _kwargs_table(rows: list[dict[str, str]]) -> list[str]:
    header = "| " + " | ".join(_KWARG_HEADERS) + " |"
    sep = "| " + " | ".join("---" for _ in _KWARG_HEADERS) + " |"
    body = [
        "| "
        + " | ".join(
            (
                _literal(r["kwarg"]).replace("|", r"\|"),
                _literal(r["default"]).replace("|", r"\|"),
                r["meaning"].replace("|", r"\|"),
            )
        )
        + " |"
        for r in rows
    ]
    return [header, sep, *body]


def _prefix_how_to_read(block: list[str]) -> list[str]:
    out = list(block)
    for i, line in enumerate(out):
        if line.strip():
            out[i] = f"**How to read it.** {line.strip()}"
            break
    return out


def _render(card: _Card) -> list[str]:
    name = _inline(card.page["name"])
    lines: list[str] = [f"### {name}", ""]
    lines.extend(_block(card.page["intro"]))
    lines.append("")
    lines.extend(_prefix_how_to_read(_block(card.page["how-to-read"])))
    lines.append("")
    if _inline(card.page["wraps"]):
        lines.extend([f"Wraps {_inline(card.page['wraps'])}.", ""])
    if card.kwargs:
        lines.extend(_kwargs_table(card.kwargs))
        lines.append("")
    if any(s.strip() for s in card.page["notes"]):
        lines.extend(_block(card.page["notes"]))
        lines.append("")
    if any(s.strip() for s in card.page["compat"]):
        lines.extend(_block(card.page["compat"]))
        lines.append("")
    preview = _inline(card.page["preview"])
    registry = _inline(card.page["registry"])
    if not preview and registry:
        preview = f"../../_static/visualisers/{registry}_visualiser.png"
    if preview:
        lines.append(f"![{name} preview]({preview})")
    return lines


class VisualiserCardDirective(SphinxDirective):
    has_content = True
    option_spec = {}  # noqa: RUF012

    def run(self) -> list[nodes.Node]:
        try:
            card = _parse(list(self.content))
            rendered = _render(card)
        except ValueError as error:
            raise self.error(str(error)) from error
        container = nodes.container()
        self.state.nested_parse(
            StringList(rendered), self.content_offset, container, match_titles=True
        )
        self.set_source_info(container)
        return list(container.children)


def setup(app: Sphinx) -> dict[str, bool]:
    app.add_directive("visualiser-card", VisualiserCardDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
