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
