"""Tests for transparency report staging helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from raitap.transparency.exceptions import VisualiserIncompatibilityError
from raitap.transparency.report import _stage_sample_thumbnail
from raitap.transparency.visualisers import InputThumbnailVisualiser

if TYPE_CHECKING:
    from pathlib import Path


def test_stage_sample_thumbnail_skips_visualiser_incompatibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A visualiser that rejects an explanation must be skipped, not propagated.

    Regression: ``validate_explanation`` raises ``VisualiserIncompatibilityError``
    (not ``ValueError``); the per-explanation skip loop must still catch it.
    """

    def _reject(self: object, *args: object, **kwargs: object) -> None:
        raise VisualiserIncompatibilityError(
            visualiser="InputThumbnailVisualiser",
            axis="input metadata",
            declared="tabular",
            accepted="image",
        )

    monkeypatch.setattr(InputThumbnailVisualiser, "validate_explanation", _reject)

    explanation = SimpleNamespace(
        original_sample_index=None,
        attributions=torch.zeros(2, 10),
        inputs=torch.zeros(2, 10),
        kwargs={},
        name="exp",
        run_dir=tmp_path,
    )
    outputs = SimpleNamespace(explanations=[explanation])
    selected = SimpleNamespace(summary=SimpleNamespace(sample_index=0))

    result = _stage_sample_thumbnail(
        outputs,  # type: ignore[arg-type]
        selected=selected,  # type: ignore[arg-type]
        assets_dir=tmp_path,
        target_name="thumb.png",
    )

    assert result is None


def test_stage_sample_thumbnail_propagates_unexpected_valueerror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bare ``ValueError`` from ``validate_explanation`` is a bug, not a rejection.

    Only ``VisualiserIncompatibilityError`` means "this visualiser can't render this
    explanation, skip it". An unexpected ``ValueError`` must surface, not be masked as
    a silently-skipped thumbnail.
    """

    def _bug(self: object, *args: object, **kwargs: object) -> None:
        raise ValueError("unexpected validate bug")

    monkeypatch.setattr(InputThumbnailVisualiser, "validate_explanation", _bug)

    explanation = SimpleNamespace(
        original_sample_index=None,
        attributions=torch.zeros(2, 10),
        inputs=torch.zeros(2, 10),
        kwargs={},
        name="exp",
        run_dir=tmp_path,
    )
    outputs = SimpleNamespace(explanations=[explanation])
    selected = SimpleNamespace(summary=SimpleNamespace(sample_index=0))

    with pytest.raises(ValueError, match="unexpected validate bug"):
        _stage_sample_thumbnail(
            outputs,  # type: ignore[arg-type]
            selected=selected,  # type: ignore[arg-type]
            assets_dir=tmp_path,
            target_name="thumb.png",
        )
