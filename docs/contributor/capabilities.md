---
title: "Backend capabilities"
description: "Reference for the Capability enum: what each capability means, which backends provide it, and which algorithms require it."
myst:
  html_meta:
    "description": "Reference for the Capability enum: what each capability means, which backends provide it, and which algorithms require it."
---

# Backend capabilities

A **capability** is something a model backend can offer that an algorithm may need. Backends declare what they `provides`; algorithms declare what they `requires`. The rule:

> An algorithm runs on a backend iff `algorithm.requires <= backend.provides`.

The inherited `AdapterMixin.check_backend_compat` enforces this and raises `BackendIncompatibilityError` on a mismatch. No adapter writes gate code.

`Capability` is a `StrEnum` in `src/raitap/types.py`. Import: `from raitap.types import Capability`.

## Reference

| Capability | Value | Status | Meaning | Provided by | Required by |
|---|---|---|---|---|---|
| `AUTOGRAD` | `"autograd"` | live | Differentiable live model with input gradients. | `TorchBackend` | captum gradient methods (IntegratedGradients, Saliency, GradCAM), torchattacks, foolbox, auto-LiRPA (CROWN/IBP) |
| `TREE_MODEL` | `"tree_model"` | live | Access to tree-ensemble structure (splits, leaf values) for TreeSHAP-style methods. | `XGBoostBackend` (and future sklearn / LightGBM backends) | SHAP `TreeExplainer` (rejected on torch/ONNX backends) |
| `PREDICT_PROBA` | `"predict_proba"` | live | Calibrated class-probability outputs. | `XGBoostBackend` | Forward pass: detects `OutputKind.PROBABILITIES` and skips softmax so confidences and metrics stay correct. Not a gate-requirer. |

`OnnxBackend` provides the empty set: it runs only model-agnostic algorithms (those with empty `requires`, e.g. SHAP `KernelExplainer`, captum `Occlusion` / `FeatureAblation`).

## Empty `requires` = model-agnostic

The default `requires=frozenset()` means "needs nothing special", so the algorithm runs on **any** backend (torch, ONNX, future tree). Set a capability only when the algorithm genuinely depends on it.

## Adding a capability

Add a member to `Capability` only when a real algorithm needs it and a backend can provide it (ship both in the same change). A capability that nothing requires is dead and untestable.

## See also

- {doc}`adding/adding-a-backend` declare `provides` on a new backend.
- {doc}`adding/adding-an-algorithm` declare `requires` on a new algorithm.
