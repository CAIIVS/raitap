"""Coverage guard: ``ModelBackend.__call__`` admits forward kwargs (e.g. attention_mask)."""

import torch
from torch import nn

from raitap.models.torch_backend import TorchBackend


class _TwoInput(nn.Module):
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True) + (0 if attention_mask is None else attention_mask.sum())


def test_backend_call_forwards_kwargs() -> None:
    backend = TorchBackend(_TwoInput(), device=torch.device("cpu"))
    x = torch.ones((2, 3))
    mask = torch.ones((2, 3))
    out = backend(x, attention_mask=mask)
    # Assert on the VALUE, not just shape: the mask must actually reach forward.
    # x.sum(dim=1)=3 per row; mask.sum()=6 => 9. Without forwarding it would be 3.
    expected = x.sum(dim=1, keepdim=True) + mask.sum()
    assert torch.equal(out, expected)
