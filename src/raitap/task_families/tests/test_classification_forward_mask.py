"""RED->GREEN regression: attention_mask threads through forward_pass (#340 T7)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from raitap.models.torch_backend import TorchBackend
from raitap.pipeline.phases.forward_pass import forward_pass


class _MaskAware(nn.Module):
    def forward(
        self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert attention_mask is not None, "mask must reach forward"
        return ids.float() @ torch.ones((ids.shape[1], 2))


def _config() -> Any:  # a fake AppConfig-shaped object for forward_pass
    return SimpleNamespace(run=SimpleNamespace(forward_batch_size=2), data=None)


def test_mask_threads_to_forward() -> None:
    backend = TorchBackend(_MaskAware(), device=torch.device("cpu"))
    ids = torch.ones((5, 4), dtype=torch.long)
    mask = torch.ones((5, 4), dtype=torch.long)
    out = forward_pass(_config(), backend, ids, forward_kwargs={"attention_mask": mask})
    # ForwardOutput has no `.predictions`; the classification payload is the
    # logits tensor itself (see `ForwardOutput.payload` / `.as_classification()`).
    payload = out.payload
    assert isinstance(payload, torch.Tensor)
    assert payload.shape[0] == 5
