"""Repo-wide pytest fixtures.

Shared model/data fixtures (formerly only in transparency/conftest.py) plus a
``seeded`` helper used by parity tests. Module-specific fixtures stay local.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def seeded() -> Callable[[int], None]:
    """Seed torch + numpy + random reproducibly; returns a re-seed callable."""

    def _seed(value: int = 0) -> None:
        import numpy as np
        import torch

        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    _seed(0)
    return _seed


@pytest.fixture
def sample_images() -> Any:
    import torch

    torch.manual_seed(0)
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def sample_tabular() -> Any:
    import torch

    torch.manual_seed(0)
    return torch.randn(8, 10)
