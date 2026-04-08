from __future__ import annotations

import sys
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

project = "RAITAP"
author = "Stanislas Laurent, Jonas Vonderhagen, Philipp Denzel, Oliver Forster"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path: list[str] = []
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "archive/**",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
root_doc = "index"

autosummary_generate = True
autosummary_imported_members = True
autodoc_typehints = "none"
autodoc_mock_imports = [
    "captum",
    "captum.attr",
    "faster_coco_eval",
    "mlflow",
    "onnxruntime",
    "openvino",
    "shap",
    "torch",
    "torch.nn",
    "torch.backends",
    "torch.backends.mps",
    "torch.utils",
    "torch.utils.data",
    "torchmetrics",
    "torchmetrics.detection",
    "torchvision",
    "triton_xpu",
]
autodoc_default_options = {
    "members": True,
    "imported-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_heading_anchors = 0

nitpick_ignore_regex = [
    (r"py:class", r"Data"),
    (r"py:class", r"'dict\[str.*"),
    (r"py:class", r"raitap\.transparency\.explainers\.base_explainer\.BaseExplainer"),
    (r"py:class", r"raitap\.transparency\.visualisers\.base_visualiser\.BaseVisualiser"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "furo"
html_title = "RAITAP Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

myst_enable_extensions = ["attrs_block", "colon_fence"]

pygments_style = "friendly"
pygments_dark_style = "monokai"


def _append_code_block(lines: list[str], *, language: str, command: str, fence_width: int) -> None:
    fence = "`" * fence_width
    lines.append(f"{fence}{{code-block}} {language}")
    for command_line in command.splitlines():
        lines.append(command_line)
    lines.append(fence)


class InstallTabsDirective(SphinxDirective):
    has_content = False
    option_spec = {
        "uv": directives.unchanged_required,
        "pip": directives.unchanged_required,
        "language": directives.unchanged,
        "sync-group": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        language = self.options.get("language", "shell")
        sync_group = self.options.get("sync-group", "install")
        uv_command = self.options["uv"]
        pip_command = self.options["pip"]

        myst_lines = [
            ":::::{tab-set}",
            f":sync-group: {sync_group}",
            "",
            "::::{tab-item} uv",
            ":sync: uv",
            "",
        ]
        _append_code_block(myst_lines, language=language, command=uv_command, fence_width=5)
        myst_lines.extend(
            [
                "",
                "::::",
                "",
                "::::{tab-item} pip",
                ":sync: pip",
                "",
            ]
        )
        _append_code_block(myst_lines, language=language, command=pip_command, fence_width=5)
        myst_lines.extend(
            [
                "",
                "::::",
                "",
                ":::::",
            ]
        )

        container = nodes.container()
        self.state.nested_parse(StringList(myst_lines), self.content_offset, container)
        return container.children


def setup(app: object) -> dict[str, bool]:
    app.add_directive("install-tabs", InstallTabsDirective)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
