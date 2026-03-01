from __future__ import annotations

from typing import Literal

import torch


def performance_from_confusion_matrix(
    cm: torch.Tensor, *, average: Literal["macro", "weighted"] = "macro", eps: float = 0.0
) -> dict[str, float]:
    """
    Compute precision, recall, f1-score from confusion matrix
    :param cm: Confusion Matrix [C, C], rows: ground truth, cols: predictions
    :param average: macro - average over all classes, weighted - average over all classes weighted by support
    :param eps: Optional numerical stability parameter, keep 0.0 for explicit zero-handling
    :return: dict {"precision": float, "recall": float, "f1": float, "specificity": float}
    """
    if not torch.is_tensor(cm):
        raise TypeError(f"cm must be torch.Tensor, got {type(cm)}")
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be a square matrix, got {cm.shape}")

    cm = cm.to(torch.float32)

    # Compute base values
    tp = torch.diag(cm)
    tn = torch.sum(cm, dim=0) - tp
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    support = cm.sum(dim=1)

    # denominators
    prec_denom = tp + fp
    recall_denom = tp + fn
    spec_denom = tn + fp

    # Compute precision with safe zero
    precision = torch.where(prec_denom > 0, tp / (prec_denom + eps), torch.zeros_like(tp))

    # Compute recall with safe zero
    recall = torch.where(recall_denom > 0, tp / (recall_denom + eps), torch.zeros_like(tp))

    # Compute specificity
    spec = torch.where(spec_denom > 0, tn / (spec_denom + eps), torch.zeros_like(tp))

    # f1 calculatio
    f1_denom = precision + recall
    f1 = torch.where(f1_denom > 0, 2 * precision * recall / (f1_denom + eps), torch.zeros_like(tp))

    if average == "macro":
        precision = precision.mean()
        recall = recall.mean()
        spec = spec.mean()
        f1 = f1.mean()
    elif average == "weighted":
        total = support.sum()
        if total.item() == 0:
            # no samples at all
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        w = support / total
        precision = (precision * w).sum()
        recall = (recall * w).sum()
        spec = (spec * w).sum()
        f1 = (f1 * w).sum()
    else:
        raise ValueError(f"Unknown average='{average}'. Use 'macro' or 'weighted'.")

    return {
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "specificity": float(spec.item()),
        "f1": float(f1.item()),
    }


# Test
cm = torch.tensor([[1, 1], [0, 2]])
print(performance_from_confusion_matrix(cm, average="macro"))
print(performance_from_confusion_matrix(cm, average="weighted"))
