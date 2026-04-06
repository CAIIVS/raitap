---
name: transparency api refactor
overview: Capture the main limitations in the current transparency abstraction and outline a later refactor that keeps today’s adapters while making room for broader explanation libraries and non-tensor outputs.
todos:
  - id: audit-contracts
    content: Review the current explainer, result, and visualiser contracts for tensor-only assumptions.
    status: pending
  - id: design-payload-model
    content: Define a lightweight generalized explanation payload model that still preserves today’s attribution workflow.
    status: pending
  - id: split-adapter-modes
    content: Define separate adapter expectations for model-specific and black-box explainers.
    status: pending
  - id: preserve-compatibility
    content: Keep Captum and SHAP integrations working while opening room for future libraries.
    status: pending
isProject: false
---

# Transparency API Refactor

## Current Shortcomings
- The current explainer contract is only partially library-agnostic. It is centered on `torch.nn.Module`, `torch.Tensor` inputs, and `torch.Tensor` attributions.
- This works well for Captum and some SHAP flows, but it does not generalize cleanly to black-box explainers, counterfactual methods, rule-based explainers, prototype methods, or richer explanation objects.
- The result and visualisation layers are also biased toward attribution maps, which makes non-attribution explanation types awkward to represent.
- Because ONNX still crosses the transparency boundary as Torch-shaped inputs, outputs, and payloads, ONNX-focused developers still need a Torch runtime today even when they do not want Torch explainers.

## Refactor Direction
- Keep the adapter-based factory pattern in [src/raitap/transparency/factory.py](src/raitap/transparency/factory.py).
- Keep existing Captum and SHAP integrations working.
- Generalize the explainer contract so adapters can consume either:
  - a model-specific explanation target such as a differentiable model object
  - a black-box prediction function or backend wrapper
- Generalize the explanation result model so it can represent more than attribution tensors, while still supporting attribution-first visualisers where applicable.
- Remove Torch-only tensor assumptions at the runtime and explanation boundary so ONNX paths can use backend-neutral inputs and outputs instead of depending on `torch.Tensor` as the universal carrier type.

## Likely Scope
- Revisit [src/raitap/transparency/explainers/base_explainer.py](src/raitap/transparency/explainers/base_explainer.py) so the interface is less tightly coupled to `torch.nn.Module` and tensor-only outputs.
- Revisit [src/raitap/transparency/results.py](src/raitap/transparency/results.py) so explanation payloads can represent attribution, counterfactual, rule, prototype, or other explanation families.
- Revisit visualiser contracts under [src/raitap/transparency/visualisers](src/raitap/transparency/visualisers) so they explicitly declare which payload kinds they can render.
- Add a clean adapter path for black-box explainers that operate on prediction functions instead of model internals.
- Revisit the ONNX-side runtime/data bridge so ONNX-only workflows no longer require `torch-cpu` just to move data through the transparency module.
- Update installation docs and `uv` commands after the refactor so ONNX-focused developers can install an ONNX runtime profile without also installing a Torch runtime profile.

## Outcome
- PyTorch-specific explainers remain first-class.
- ONNX and other non-autograd backends gain a cleaner path for black-box explainers.
- Future integrations such as Alibi or OmniXAI can be added with less distortion and fewer one-off wrappers.
- ONNX-only developer setups can drop `torch-cpu` once the tensor-centric contracts have been removed and the docs/runtime profiles are updated accordingly.