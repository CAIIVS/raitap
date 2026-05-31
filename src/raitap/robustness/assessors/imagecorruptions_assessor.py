"""imagecorruptions adapter — average-case robustness via the ImageNet-C suite."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.robustness.assessors.registration import robustness_adapter
from raitap.utils.lazy import lazy_import

from ..contracts import AssessmentKind, Objective, ThreatModel
from ..semantics import AssessorSemanticsHints
from .base_assessor import StatisticalSamplingAssessor

if TYPE_CHECKING:
    import numpy as np
else:
    np = lazy_import("numpy")

_NOISE = frozenset({"common_corruption", "noise"})
_BLUR = frozenset({"common_corruption", "blur"})
_WEATHER = frozenset({"common_corruption", "weather"})
_DIGITAL = frozenset({"common_corruption", "digital"})

_CORRUPTIONS: dict[str, frozenset[str]] = {
    "gaussian_noise": _NOISE,
    "shot_noise": _NOISE,
    "impulse_noise": _NOISE,
    "defocus_blur": _BLUR,
    "glass_blur": _BLUR,
    "motion_blur": _BLUR,
    "zoom_blur": _BLUR,
    "snow": _WEATHER,
    "frost": _WEATHER,
    "fog": _WEATHER,
    "brightness": _WEATHER,
    "contrast": _DIGITAL,
    "elastic_transform": _DIGITAL,
    "pixelate": _DIGITAL,
    "jpeg_compression": _DIGITAL,
}


@robustness_adapter(
    registry_name="imagecorruptions",
    library="imagecorruptions",
    algorithm_registry={
        name: AssessorSemanticsHints(
            AssessmentKind.STATISTICAL_SAMPLING,
            ThreatModel.NOT_APPLICABLE,
            Objective.UNTARGETED,
            families=families,
        )
        for name, families in _CORRUPTIONS.items()
    },
    # imagecorruptions triggers a `pkg_resources is deprecated` UserWarning on
    # import (last released 2020, upstream won't fix). Silence — not actionable.
    suppress_warnings=[
        (r"pkg_resources is deprecated as an API", UserWarning, r"imagecorruptions.*"),
    ],
)
class ImageCorruptionsAssessor(StatisticalSamplingAssessor):
    """Apply one ImageNet-C corruption at one severity to estimate average-case accuracy.

    ``algorithm`` is the corruption name; ``severity`` (1..5) is a constructor kwarg.
    """

    def __init__(self, algorithm: str, *, severity: int = 1, **init_kwargs: Any) -> None:
        if algorithm not in _CORRUPTIONS:
            valid = ", ".join(sorted(_CORRUPTIONS))
            raise ValueError(f"Unknown corruption {algorithm!r}. Known: {valid}.")
        if not 1 <= int(severity) <= 5:
            raise ValueError(f"severity must be in 1..5, got {severity}.")
        self.algorithm = algorithm
        self.severity = int(severity)
        self.init_kwargs = dict(init_kwargs)

    def apply_perturbation(self, image: np.ndarray) -> np.ndarray:
        imagecorruptions = self._lazy_import()
        with self._rethrow():
            corrupted = imagecorruptions.corrupt(
                image, corruption_name=self.algorithm, severity=self.severity
            )
        return np.asarray(corrupted, dtype=np.uint8)
