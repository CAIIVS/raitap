"""
Calculate the confusion matrix for a multi-class classification task.

This function computes the confusion matrix for given true and predicted
labels of a dataset, along with the specified number of classes. The confusion
matrix summarizes the performance of a classification model by showing the
counts of true positive, false positive, true negative, and false negative
predictions for each class. An optional index can be ignored during the computation.

:param y_true: Tensor containing true class labels.
:param y_pred: Tensor containing predicted class labels.
:param num_classes: Number of classes in the classification problem.
:param ignore_index: Optional index to exclude from confusion matrix computation.
                     Defaults to None.

:return: The computed confusion matrix as a torch.Tensor of shape
         (num_classes, num_classes).
"""

from __future__ import annotations

from typing import Optional

import torch


def confusion_matrix(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int, ignore_index: Optional[int] = None
) -> torch.Tensor:

    if not torch.is_tensor(y_true) or not torch.is_tensor(y_pred):
        raise TypeError(
            f"y_true and y_pred must be torch.Tensors, got {type(y_true)} and {type(y_pred)}"
        )
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}"
        )
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes}")

    # Flatten tensors
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)

    # Mask ignored labels
    if ignore_index is not None:
        mask = y_true_f != ignore_index
        y_true_f = y_true_f[mask]
        y_pred_f = y_pred_f[mask]

    # Empty after masking -> return zeros
    if y_true_f.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=y_true.device)

    # Range Checks
    if (y_true_f.min() < 0) or (y_true_f.max() >= num_classes):
        raise ValueError(
            f"y_true must be in range [0, {num_classes - 1}], got {y_true_f.min()} and {y_true_f.max()}"
        )
    if (y_pred_f.min() < 0) or (y_pred_f.max() >= num_classes):
        raise ValueError(
            f"y_pred must be in range [0, {num_classes - 1}], got {y_pred_f.min()} and {y_pred_f.max()}"
        )

    idx = y_true_f.to(torch.int64) * num_classes + y_pred_f.to(torch.int64)

    # Number of occurences of each pair of indices
    counts = torch.bincount(idx, minlength=num_classes**2)

    # Reshape into a confusion matrix
    cm = counts.reshape(num_classes, num_classes).to(torch.int64)
    return cm


# Test 1
y_true = torch.tensor([0, 0, 1, 1])
y_pred = torch.tensor([0, 1, 1, 1])
cm = confusion_matrix(y_true, y_pred, num_classes=2)
# Expect:
# [[1, 1],
#  [0, 2]]

# Test 2
y_true = torch.tensor([0, -1, 1, -1])
y_pred = torch.tensor([0, 0, 1, 1])
cm = confusion_matrix(y_true, y_pred, num_classes=2, ignore_index=-1)
# Expect:
# [[1, 0],
#  [0, 1]]
