from __future__ import annotations

import pytest

from raitap.utils.errors import RaitapError, SampleNamesLengthError


def test_subclass_of_raitap_error() -> None:
    assert issubclass(SampleNamesLengthError, RaitapError)


def test_message_includes_counts_and_source() -> None:
    exc = SampleNamesLengthError(got=3, expected=2, source="runtime kwarg")
    msg = str(exc)
    assert "3" in msg
    assert "2" in msg
    assert "runtime kwarg" in msg
    assert "sample_names" in msg


def test_attributes_set() -> None:
    exc = SampleNamesLengthError(got=5, expected=4, source="raitap.sample_names")
    assert exc.got == 5
    assert exc.expected == 4


def test_raises_as_expected() -> None:
    with pytest.raises(SampleNamesLengthError):
        raise SampleNamesLengthError(got=1, expected=2, source="x")
