"""Reference-value tests for the binomial CI helpers.

Wilson and Clopper-Pearson intervals for (k=8, n=10) at 95% are well-known
textbook values; we pin them so a refactor that silently changes the math fails.
"""

from __future__ import annotations

import pytest

from raitap.robustness.assessors._ci import binomial_ci


def test_wilson_reference_value() -> None:
    low, high = binomial_ci(8, 10, level=0.95, method="wilson")
    assert low == pytest.approx(0.4901, abs=1e-3)
    assert high == pytest.approx(0.9426, abs=1e-3)


def test_clopper_pearson_reference_value() -> None:
    low, high = binomial_ci(8, 10, level=0.95, method="clopper_pearson")
    assert low == pytest.approx(0.4439, abs=1e-3)
    assert high == pytest.approx(0.9748, abs=1e-3)


def test_all_correct_wilson_upper_is_one() -> None:
    low, high = binomial_ci(10, 10, level=0.95, method="wilson")
    assert high == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < low < 1.0


def test_all_wrong_wilson_lower_is_zero() -> None:
    low, high = binomial_ci(0, 10, level=0.95, method="wilson")
    assert low == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < high < 1.0


def test_zero_samples_returns_full_interval() -> None:
    assert binomial_ci(0, 0, method="wilson") == (0.0, 1.0)


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="ci_method"):
        binomial_ci(1, 2, method="bogus")  # type: ignore[arg-type]
