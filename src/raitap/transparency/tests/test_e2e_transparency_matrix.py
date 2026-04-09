"""
Centralised transparency E2E behavior matrix.

The heavy PR-only suite is matrix-driven rather than hand-written one test per
combination:

    # fast suite
    uv run pytest -m "not e2e"

    # heavy suite
    uv run pytest -m e2e -v --tb=long --mpl
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from raitap.transparency.tests.e2e_case_matrix import (
    MATRIX_CASES,
    MatrixCase,
    assert_behavior_case,
    run_behavior_case,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.e2e]


@pytest.mark.parametrize("case", MATRIX_CASES, ids=[case.id for case in MATRIX_CASES])
def test_transparency_behavior_matrix_case(
    case: MatrixCase,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    result = run_behavior_case(case, request, tmp_path)
    assert_behavior_case(result, tmp_path=tmp_path)
