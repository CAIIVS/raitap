from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch
else:
    torch = lazy_import("torch")


def tensor_to_python(x: Any) -> Any:
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.numel() == 1:
            return float(x.item())
        return x.tolist()
    return x
