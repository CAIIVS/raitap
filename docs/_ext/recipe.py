"""``{recipe}`` directive — render a minimal config-plus-output example page.

Pages under ``docs/using-raitap/examples/`` follow a strict shape so they
stay LLM-friendly and copy-paste-able: one-line summary, paired YAML +
Python config, and a sample of the expected output. This directive folds
all of that into one source-of-truth block per recipe so the structure
cannot drift::

    # ImageNet · Captum IG · Torchattacks PGD

    ```{recipe}
    :summary: One-line description used as the page intro + ``llms.txt`` blurb.

    :yaml:
    defaults:
      - raitap_schema
      - _self_
    ...

    :python:
    from raitap.configs.schema import AppConfig
    ...

    :output:
    outputs/<date>/<time>/
    ├── metrics/...
    ```

Renders directly into docutils nodes (no nested markdown parse) so each
recipe page surfaces the same shape: a summary paragraph, a ``## Config``
section with the YAML and Python snippets stacked under language-tagged
literal blocks, and an optional ``## Expected output`` section with a plain
``text`` literal block.

Stacking the YAML and Python blocks vertically (rather than wrapping them in
``{config-tabs}``) is deliberate: scrapers see both copies in source order,
no JS-gated tab selection swallows the second one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config_tabs import create_tab_item
from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from sphinx_design.shared import create_component

if TYPE_CHECKING:
    from sphinx.application import Sphinx


_RECIPE_FIELDS = (":summary:", ":yaml:", ":python:", ":output:")


def _parse_recipe_content(lines: list[str]) -> dict[str, list[str]]:
    """Split the directive body by ``:field:`` markers.

    Each marker can be on its own line (``:yaml:``) followed by indented
    content, or carry inline content on the same line (``:summary: short
    sentence here``). Marker lines themselves are stripped from the field
    body so the renderer can paste each section into its final fence without
    leading marker noise.
    """
    sections: dict[str, list[str]] = {field: [] for field in _RECIPE_FIELDS}
    current: str | None = None
    for raw in lines:
        marker_hit: str | None = None
        inline: str = ""
        stripped = raw.lstrip()
        for field in _RECIPE_FIELDS:
            if stripped == field or stripped.rstrip() == field:
                marker_hit = field
                inline = ""
                break
            if stripped.startswith(field + " ") or stripped.startswith(field + "\t"):
                marker_hit = field
                inline = stripped[len(field) :].strip()
                break
        if marker_hit is not None:
            current = marker_hit
            if inline:
                sections[current].append(inline)
            continue
        if current is None:
            if raw.strip():
                raise ValueError(
                    "`recipe` content must start with one of "
                    + ", ".join(f"`{f}`" for f in _RECIPE_FIELDS)
                    + "."
                )
            continue
        sections[current].append(raw)

    missing = [f for f in (":summary:", ":yaml:", ":python:") if not "".join(sections[f]).strip()]
    if missing:
        raise ValueError(
            "`recipe` missing required field(s): " + ", ".join(f"`{f}`" for f in missing) + "."
        )
    return sections


def _trim_block(lines: list[str]) -> str:
    out = list(lines)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def _code_block(body: str, language: str) -> nodes.literal_block:
    block = nodes.literal_block(body, body)
    block["language"] = language
    return block


_TAB_SYNC_GROUP = "raitap-config"


class RecipeDirective(SphinxDirective):
    """Stack the recipe sections as docutils nodes in source order.

    Uses :func:`config_tabs.create_tab_item` for the YAML / Python pair so
    the look matches every other tabbed config block in the docs. Both
    snippets land in the static HTML — the ``sd-tab-set`` only JS-toggles
    visibility — so LLM scrapers see both in source order.

    Uses ``rubric`` for the ``Expected output`` sub-label so the recipe
    page's H1 stays the only TOC-indexed heading.
    """

    has_content = True
    option_spec = {}  # noqa: RUF012

    def run(self) -> list[nodes.Node]:
        try:
            sections = _parse_recipe_content(list(self.content))
        except ValueError as error:
            raise self.error(str(error)) from error

        summary = " ".join(line.strip() for line in sections[":summary:"] if line.strip())
        yaml_body = _trim_block(sections[":yaml:"])
        python_body = _trim_block(sections[":python:"])
        output_body = _trim_block(sections[":output:"])

        tab_set = create_component(
            "tab-set",
            classes=["sd-tab-set"],
            children=[
                create_tab_item(
                    label="YAML",
                    snippet=yaml_body,
                    language="yaml",
                    sync_group=_TAB_SYNC_GROUP,
                    selected=True,
                ),
                create_tab_item(
                    label="Python",
                    snippet=python_body,
                    language="python",
                    sync_group=_TAB_SYNC_GROUP,
                    selected=False,
                ),
            ],
        )

        result: list[nodes.Node] = [nodes.paragraph(text=summary), tab_set]
        if output_body:
            result.append(nodes.rubric(text="Expected output"))
            result.append(_code_block(output_body, "text"))

        for node in result:
            self.set_source_info(node)
        return result


def setup(app: Sphinx) -> dict[str, bool]:
    app.add_directive("recipe", RecipeDirective)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
