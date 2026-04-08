from __future__ import annotations

from typing import TYPE_CHECKING

from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from sphinx_design.shared import create_component

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def parse_install_tabs_content(lines: list[str]) -> dict[str, str]:
    sections: dict[str, list[str]] = {"uv": [], "pip": []}
    current_section: str | None = None

    for line in lines:
        stripped_line = line.strip()
        if stripped_line in {":uv:", ":pip:"}:
            current_section = stripped_line[1:-1]
            continue
        if current_section is None:
            if not stripped_line:
                continue
            raise ValueError("`install-tabs` content must start with a `:uv:` or `:pip:` marker.")
        sections[current_section].append(line)

    commands = {name: "\n".join(command_lines).strip() for name, command_lines in sections.items()}
    missing_sections = [name for name, command in commands.items() if not command]
    if missing_sections:
        missing_sections_str = ", ".join(f"`:{name}:`" for name in missing_sections)
        raise ValueError(f"`install-tabs` is missing content for {missing_sections_str}.")

    return commands


def create_tab_item(
    *,
    label: str,
    command: str,
    language: str,
    sync_group: str,
    selected: bool,
) -> nodes.container:
    literal_block = nodes.literal_block(command, command)
    literal_block["language"] = language

    tab_label = nodes.rubric(
        label,
        "",
        nodes.Text(label),
        classes=["sd-tab-label"],
    )
    tab_label["sync_group"] = sync_group
    tab_label["sync_id"] = label

    tab_content = create_component(
        "tab-content",
        classes=["sd-tab-content"],
        children=[literal_block],
    )
    return create_component(
        "tab-item",
        classes=["sd-tab-item"],
        children=[tab_label, tab_content],
        selected=selected,
    )


class InstallTabsDirective(SphinxDirective):
    has_content = True
    option_spec = {}  # noqa: RUF012

    def run(self) -> list[nodes.Node]:
        language = "shell"
        sync_group = "install"
        try:
            commands = parse_install_tabs_content(list(self.content))
        except ValueError as error:
            raise self.error(str(error)) from error

        tab_set = create_component(
            "tab-set",
            classes=["sd-tab-set"],
            children=[
                create_tab_item(
                    label="uv",
                    command=commands["uv"],
                    language=language,
                    sync_group=sync_group,
                    selected=True,
                ),
                create_tab_item(
                    label="pip",
                    command=commands["pip"],
                    language=language,
                    sync_group=sync_group,
                    selected=False,
                ),
            ],
        )
        self.set_source_info(tab_set)
        return [tab_set]


def setup(app: Sphinx) -> dict[str, bool]:
    app.add_directive("install-tabs", InstallTabsDirective)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
