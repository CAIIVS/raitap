"""
Matrix-driven MPL regression tests for transparency visual output.

Only a curated subset of the shared E2E case list opts into pixel regression.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from raitap.transparency.tests.e2e_case_matrix import MATRIX_CASES, MatrixCase, render_mpl_figure

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_BASELINE_DIR = Path(__file__).with_name("mpl_baseline")
_MPL_TOLERANCE = 10 if sys.platform.startswith("win") else 2


def _baseline_file(case: MatrixCase) -> Path:
    baseline_name = case.mpl_baseline_filename
    if baseline_name is None:
        raise ValueError(f"{case.id} does not define an MPL baseline filename.")
    return _BASELINE_DIR / baseline_name


def _ensure_baseline_or_generation_mode(
    request: pytest.FixtureRequest,
    case: MatrixCase,
) -> None:
    generate_path = getattr(request.config.option, "mpl_generate_path", None)
    baseline_file = _baseline_file(case)
    if baseline_file.exists() or generate_path:
        return

    pytest.fail(
        "The MPL baseline image is missing. Generate or provide the baseline "
        f"PNG at: {baseline_file.as_posix()}\n\n"
        "Suggested command to regenerate candidate artifacts locally:\n"
        "uv run pytest src/raitap/transparency/tests/test_e2e_mpl_baseline.py "
        "-m e2e --mpl-generate-path=src/raitap/transparency/tests/mpl_baseline_candidate -v"
    )


def _mpl_case_param(case: MatrixCase) -> object:
    baseline_name = case.mpl_baseline_filename
    if baseline_name is None:
        raise ValueError(f"{case.id} does not define an MPL baseline filename.")
    return pytest.param(
        case,
        id=case.id,
        marks=pytest.mark.mpl_image_compare(
            baseline_dir="mpl_baseline",
            filename=baseline_name,
            remove_text=True,
            savefig_kwargs={"dpi": 150},
            tolerance=_MPL_TOLERANCE,
        ),
    )


MPL_CASES = tuple(case for case in MATRIX_CASES if case.mpl_baseline_filename is not None)


@pytest.mark.e2e
@pytest.mark.mpl
@pytest.mark.parametrize("case", [_mpl_case_param(case) for case in MPL_CASES])
def test_transparency_visual_regression_matrix_case(
    case: MatrixCase,
    request: pytest.FixtureRequest,
) -> Figure:
    _ensure_baseline_or_generation_mode(request, case)
    return render_mpl_figure(case, request)
