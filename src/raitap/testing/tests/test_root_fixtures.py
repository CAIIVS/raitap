from __future__ import annotations

from typing import Any


def test_sample_images_fixture_available(sample_images: Any) -> None:
    assert tuple(sample_images.shape) == (4, 3, 32, 32)


def test_seeded_fixture_is_reproducible(seeded: Any) -> None:
    import torch

    first = torch.randn(3)
    seeded()
    second = torch.randn(3)
    assert torch.equal(first, second)
