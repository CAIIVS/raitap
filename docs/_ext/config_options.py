from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

if TYPE_CHECKING:
    from sphinx.application import Sphinx

CONFIG_OPTION_FIELDS = ("option", "allowed", "default", "description")
CONFIG_PAGE_FIELDS = ("intro", "yaml", "cli")
TABLE_HEADERS = ("Name", "Allowed", "Default", "Description")


@dataclass(frozen=True)
class ConfigOption:
    option: str
    allowed: str
    default: str
    description: str


@dataclass(frozen=True)
class ConfigPage:
    intro: str
    yaml: str
    cli: str
    options: list[ConfigOption]


def normalise_inline_value(lines: list[str]) -> str:
    return " ".join(line.strip() for line in lines if line.strip()).strip()


def normalise_block_value(lines: list[str]) -> str:
    trimmed_lines = list(lines)
    while trimmed_lines and not trimmed_lines[0].strip():
        trimmed_lines.pop(0)
    while trimmed_lines and not trimmed_lines[-1].strip():
        trimmed_lines.pop()
    return "\n".join(trimmed_lines)


def create_empty_entry() -> dict[str, list[str]]:
    return {field: [] for field in CONFIG_OPTION_FIELDS}


def finalise_entry(entry: dict[str, list[str]] | None) -> ConfigOption | None:
    if entry is None or not any(entry.values()):
        return None

    values = {field: normalise_inline_value(entry[field]) for field in CONFIG_OPTION_FIELDS}
    missing_fields = [field for field, value in values.items() if not value]
    if missing_fields:
        missing_fields_str = ", ".join(f"`:{field}:`" for field in missing_fields)
        raise ValueError(
            f"`config-options` entry is missing required field(s): {missing_fields_str}."
        )

    return ConfigOption(
        option=values["option"],
        allowed=values["allowed"],
        default=values["default"],
        description=values["description"],
    )


def parse_field_marker(line: str, allowed_fields: tuple[str, ...]) -> tuple[str, str] | None:
    if not line.startswith(":"):
        return None

    separator_index = line.find(":", 1)
    if separator_index == -1:
        return None

    field_name = line[1:separator_index]
    if field_name not in allowed_fields:
        raise ValueError(
            "Unsupported field marker. Supported fields are "
            + ", ".join(f"`:{field}:`" for field in allowed_fields)
            + "."
        )

    return field_name, line[separator_index + 1 :].strip()


def parse_config_options_content(lines: list[str]) -> list[ConfigOption]:
    options: list[ConfigOption] = []
    current_entry: dict[str, list[str]] | None = None
    current_field: str | None = None

    for line in lines:
        stripped_line = line.strip()
        field_marker = parse_field_marker(stripped_line, CONFIG_OPTION_FIELDS)

        if field_marker is not None:
            field_name, field_value = field_marker
            if field_name == "option" and current_entry is not None and current_entry["option"]:
                option = finalise_entry(current_entry)
                if option is not None:
                    options.append(option)
                current_entry = create_empty_entry()
            elif current_entry is None:
                current_entry = create_empty_entry()
            elif current_entry[field_name]:
                raise ValueError(
                    "`config-options` does not allow multiple "
                    f"`:{field_name}:` fields in one entry."
                )

            current_entry[field_name].append(field_value)
            current_field = field_name
            continue

        if not stripped_line:
            option = finalise_entry(current_entry)
            if option is not None:
                options.append(option)
            current_entry = None
            current_field = None
            continue

        if current_entry is None or current_field is None:
            raise ValueError("`config-options` entries must start with `:option:`.")

        current_entry[current_field].append(stripped_line)

    option = finalise_entry(current_entry)
    if option is not None:
        options.append(option)

    if not options:
        raise ValueError("`config-options` requires at least one option entry.")

    return options


def parse_config_page_content(lines: list[str]) -> ConfigPage:
    page_fields = {field: [] for field in CONFIG_PAGE_FIELDS}
    current_page_field: str | None = None
    current_entry: dict[str, list[str]] | None = None
    current_option_field: str | None = None
    options: list[ConfigOption] = []

    for line in lines:
        stripped_line = line.strip()
        field_marker = parse_field_marker(stripped_line, CONFIG_PAGE_FIELDS + CONFIG_OPTION_FIELDS)

        if field_marker is not None:
            field_name, field_value = field_marker
            if field_name in CONFIG_PAGE_FIELDS:
                option = finalise_entry(current_entry)
                if option is not None:
                    options.append(option)
                current_entry = None
                current_option_field = None

                if page_fields[field_name]:
                    raise ValueError(
                        f"`config-page` does not allow multiple `:{field_name}:` fields."
                    )

                page_fields[field_name].append(field_value)
                current_page_field = field_name
                continue

            current_page_field = None
            if field_name == "option":
                option = finalise_entry(current_entry)
                if option is not None:
                    options.append(option)
                current_entry = create_empty_entry()
            elif current_entry is None:
                raise ValueError("`config-page` option entries must start with `:option:`.")
            elif current_entry[field_name]:
                raise ValueError(
                    "`config-page` does not allow multiple "
                    f"`:{field_name}:` fields in one option entry."
                )

            current_entry[field_name].append(field_value)
            current_option_field = field_name
            continue

        if not stripped_line:
            option = finalise_entry(current_entry)
            if option is not None:
                options.append(option)
                current_entry = None
                current_option_field = None
                continue

            if current_page_field is not None:
                page_fields[current_page_field].append(line)
            continue

        if current_page_field is not None:
            page_fields[current_page_field].append(line)
            continue

        if current_entry is None or current_option_field is None:
            raise ValueError(
                "`config-page` content must start with a page field like "
                "`:intro:`, `:yaml:`, `:cli:`, or an option entry beginning with `:option:`."
            )

        current_entry[current_option_field].append(stripped_line)

    option = finalise_entry(current_entry)
    if option is not None:
        options.append(option)

    if not options:
        raise ValueError("`config-page` requires at least one option entry.")

    intro = normalise_block_value(page_fields["intro"])
    yaml_content = normalise_block_value(page_fields["yaml"])
    cli_override = normalise_inline_value(page_fields["cli"])

    missing_fields = [
        field
        for field, value in {
            "intro": intro,
            "yaml": yaml_content,
            "cli": cli_override,
        }.items()
        if not value
    ]
    if missing_fields:
        missing_fields_str = ", ".join(f"`:{field}:`" for field in missing_fields)
        raise ValueError(f"`config-page` is missing required field(s): {missing_fields_str}.")

    return ConfigPage(
        intro=intro,
        yaml=yaml_content,
        cli=cli_override,
        options=options,
    )


def format_literal(value: str) -> str:
    escaped_value = value.replace("`", r"\`")
    return f"`{escaped_value}`"


def should_render_as_literal(value: str) -> bool:
    stripped = value.strip()
    has_myst_role = re.search(r"\{[A-Za-z0-9:_-]+\}`", stripped) is not None
    has_markdown_link = re.search(r"\[[^\]]+\]\([^)]+\)", stripped) is not None
    has_inline_code = "`" in stripped
    return not (has_myst_role or has_markdown_link or has_inline_code)


def format_table_value(value: str, *, literal: bool) -> str:
    if literal and should_render_as_literal(value):
        return format_literal(value)
    return value


def escape_table_cell(value: str) -> str:
    return value.replace("|", r"\|")


def build_markdown_table_lines(options: list[ConfigOption]) -> list[str]:
    header_line = "| " + " | ".join(TABLE_HEADERS) + " |"
    separator_line = "| " + " | ".join("---" for _ in TABLE_HEADERS) + " |"
    data_lines = [
        "| "
        + " | ".join(
            [
                escape_table_cell(format_table_value(option.option, literal=True)),
                escape_table_cell(format_table_value(option.allowed, literal=True)),
                escape_table_cell(format_table_value(option.default, literal=True)),
                escape_table_cell(format_table_value(option.description, literal=False)),
            ]
        )
        + " |"
        for option in options
    ]
    return [header_line, separator_line, *data_lines]


def build_install_tabs_lines(cli_override: str) -> list[str]:
    uv_command = f"uv run raitap {cli_override}".strip()
    pip_command = f"raitap {cli_override}".strip()
    return [
        "```{install-tabs}",
        ":uv:",
        uv_command,
        "",
        ":pip:",
        pip_command,
        "```",
    ]


def slugify_label_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()


def build_section_label(docname: str, section_name: str) -> str:
    doc_label = slugify_label_part(docname)
    section_label = slugify_label_part(section_name)
    return f"{doc_label}-{section_label}"


def build_config_page_lines(page: ConfigPage, *, docname: str) -> list[str]:
    options_label = build_section_label(docname, "options")
    yaml_label = build_section_label(docname, "yaml-example")
    cli_label = build_section_label(docname, "cli-override-example")
    lines = ["# Configuration", ""]
    lines.extend(page.intro.splitlines())
    lines.extend(["", f"({options_label})=", "## Options", ""])
    lines.extend(build_markdown_table_lines(page.options))
    lines.extend(["", f"({yaml_label})=", "## YAML example", "", "```yaml"])
    lines.extend(page.yaml.splitlines())
    lines.extend(["```", "", f"({cli_label})=", "## CLI override example", ""])
    lines.extend(build_install_tabs_lines(page.cli))
    return lines


class BaseMarkdownRenderingDirective(SphinxDirective):
    has_content = True
    option_spec = {}  # noqa: RUF012

    def parse_markdown_lines(self, lines: list[str]) -> list[nodes.Node]:
        container = nodes.container()
        self.state.nested_parse(
            StringList(lines),
            self.content_offset,
            container,
            match_titles=True,
        )
        self.set_source_info(container)
        return list(container.children)


class ConfigOptionsDirective(BaseMarkdownRenderingDirective):
    has_content = True
    option_spec = {}  # noqa: RUF012

    def run(self) -> list[nodes.Node]:
        try:
            options = parse_config_options_content(list(self.content))
        except ValueError as error:
            raise self.error(str(error)) from error

        return self.parse_markdown_lines(build_markdown_table_lines(options))


class ConfigPageDirective(BaseMarkdownRenderingDirective):
    has_content = True
    option_spec = {}  # noqa: RUF012

    def run(self) -> list[nodes.Node]:
        try:
            page = parse_config_page_content(list(self.content))
        except ValueError as error:
            raise self.error(str(error)) from error

        return self.parse_markdown_lines(build_config_page_lines(page, docname=self.env.docname))


def setup(app: Sphinx) -> dict[str, bool]:
    app.add_directive("config-options", ConfigOptionsDirective)
    app.add_directive("config-page", ConfigPageDirective)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
