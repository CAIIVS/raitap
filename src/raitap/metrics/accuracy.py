from __future__ import annotations

from typing import Optional, Union

import torch


def accuracy(
    y_true: torch.Tensor, y_pred: torch.Tensor, ignore_index: Optional[int] = None
) -> float:
    """
    Compute accuracy of classification model

    :param y_true: Ground truth labels
    :param y_pred: Predicted labels
    :param ignore_index: Index to ignore in the calculation
    :return: Accuracy score
    """
    # Check parameters
    if not torch.is_tensor(y_true) or not torch.is_tensor(y_pred):
        raise TypeError(
            f"y_true and y_pred must be torch.Tensors, got {type(y_true)} and {type(y_pred)}"
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true.shape={tuple(y_true.shape)} vs y_pred.shape={tuple(y_pred.shape)}"
        )

    # Flatten tensors
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)

    if ignore_index is not None:
        mask = y_true_f != ignore_index
        y_true_f = y_true_f[mask]
        y_pred_f = y_pred_f[mask]

    n = int(y_true_f.numel())  # Number of elements
    correct = (y_true_f == y_pred_f).sum()
    acc = correct.to(torch.float32) / float(n)
    return float(acc.item())
