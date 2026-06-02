"""Tests for category-name capture on TorchBackend and _load_pretrained."""

from __future__ import annotations

import pytest


def test_torchbackend_category_names_defaults_none() -> None:
    import torch.nn as nn

    from raitap.models.backend import TorchBackend

    backend = TorchBackend(nn.Linear(2, 2))
    assert backend.category_names is None


def test_torchbackend_category_names_stored() -> None:
    import torch.nn as nn

    from raitap.models.backend import TorchBackend

    backend = TorchBackend(nn.Linear(2, 2), category_names=["__background__", "kite"])
    assert backend.category_names == ["__background__", "kite"]


@pytest.mark.slow
def test_load_pretrained_detection_captures_categories() -> None:
    pytest.importorskip("torchvision")
    from raitap.models.model import _load_pretrained

    backend = _load_pretrained("fasterrcnn_resnet50_fpn_v2", hardware="cpu")
    cats = backend.category_names
    assert cats is not None
    assert cats[0] == "__background__"
    from torchvision.models import get_model_weights

    expected = get_model_weights("fasterrcnn_resnet50_fpn_v2").DEFAULT.meta["categories"]  # type: ignore[attr-defined]
    assert cats == list(expected)
    assert len(cats) > 38
