from __future__ import annotations

from typing import Any

import torch


def tensor_to_python(x: Any) -> Any:
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.numel() == 1:
            return float(x.item())
        return x.tolist()
    return x
