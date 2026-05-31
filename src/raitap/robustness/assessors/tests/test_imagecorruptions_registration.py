"""Adapter registration + lazy-import safety for the imagecorruptions assessor."""

from __future__ import annotations

import importlib.util

import pytest

from raitap.robustness.assessors.imagecorruptions_assessor import ImageCorruptionsAssessor
from raitap.robustness.contracts import AssessmentKind, ThreatModel

_EXPECTED = {
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
}


def test_all_fifteen_corruptions_registered() -> None:
    assert set(ImageCorruptionsAssessor.algorithm_registry) == _EXPECTED


def test_every_entry_is_statistical_sampling_and_not_applicable() -> None:
    for hints in ImageCorruptionsAssessor.algorithm_registry.values():
        assert hints.assessment_kind is AssessmentKind.STATISTICAL_SAMPLING
        assert hints.threat_model is ThreatModel.NOT_APPLICABLE


def test_constructor_stores_algorithm_and_severity() -> None:
    assessor = ImageCorruptionsAssessor(algorithm="gaussian_noise", severity=1)
    assert assessor.algorithm == "gaussian_noise"
    assert assessor.severity == 1


def test_unknown_corruption_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown corruption"):
        ImageCorruptionsAssessor(algorithm="not_a_corruption", severity=1)


def test_severity_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="severity"):
        ImageCorruptionsAssessor(algorithm="fog", severity=9)


@pytest.mark.skipif(
    importlib.util.find_spec("imagecorruptions") is None,
    reason="imagecorruptions extra not installed",
)
def test_apply_perturbation_changes_image_and_preserves_shape() -> None:
    import numpy as np

    assessor = ImageCorruptionsAssessor(algorithm="gaussian_noise", severity=3)
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    out = assessor.apply_perturbation(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, img)  # noise actually perturbs
