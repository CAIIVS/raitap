from __future__ import annotations

import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
EXT_DIR = DOCS_DIR / "_ext"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(EXT_DIR))

project = "RAITAP"
author = "Stanislas Laurent, Jonas Vonderhagen, Philipp Denzel, Oliver Forster"

extensions = [
    "config_options",
    "install_tabs",
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

myst_heading_anchors = 4

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

# Copyright footer
copyright = "2025, Stanislas Laurent, Jonas Vonderhagen, Philipp Denzel, Oliver Forster"
html_show_copyright = True

# Custom footer text
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/CAIIVS/raitap",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

myst_enable_extensions = ["attrs_block", "colon_fence"]

pygments_style = "friendly"
pygments_dark_style = "monokai"
