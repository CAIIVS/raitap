"""Custom-file option test fixture: a minimal user-provided preprocessing factory.

Used by ``test_preprocessing.py`` and ``test_api.py`` to exercise the
``data.preprocessing: <path>.py`` code path. Mirrors the shape of the
example shown in ``docs/modules/data/preprocessing.md`` so the test doubles as
documentation that the example works.
"""

from __future__ import annotations

from torch import nn
from torchvision.transforms import v2


def make_preprocessing() -> nn.Module:
    return nn.Sequential(
        v2.Resize(232, antialias=True),
        v2.CenterCrop(224),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )
