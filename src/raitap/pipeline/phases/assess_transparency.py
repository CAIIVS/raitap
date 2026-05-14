"""Transparency phase — instantiates explainers + collects results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap import raitap_log
from raitap.configs import cfg_to_dict
from raitap.metrics import metrics_prediction_pair
from raitap.transparency.factory import Explanation

if TYPE_CHECKING:
    import torch

    from raitap.configs.schema import AppConfig
    from raitap.data import Data
    from raitap.models import Model
    from raitap.transparency.contracts import InputSpec
    from raitap.transparency.results import ExplanationResult, VisualisationResult


def resolve_explainer_runtime_kwargs(
    explainer_config: Any,
    *,
    forward_output: torch.Tensor,
) -> dict[str, Any]:
    """Resolve runtime kwargs for an explainer config.

    Currently handles the ``target: auto_pred`` sentinel — replaces it with the
    model's argmax predictions so the explainer attributes wrt. the predicted
    class. Returns the empty dict when no rewriting is needed.
    """
    raw_config = cfg_to_dict(explainer_config)
    call_config = raw_config.get("call")
    if not isinstance(call_config, dict):
        return {}
    if call_config.get("target") != "auto_pred":
        return {}
    predictions, _ = metrics_prediction_pair(forward_output)
    return {"target": predictions.detach()}


def assess_transparency(
    config: AppConfig,
    model: Model,
    data: Data,
    forward_output: torch.Tensor,
    *,
    input_metadata: InputSpec | None,
) -> tuple[list[ExplanationResult], list[VisualisationResult]]:
    """Run every explainer declared under ``config.transparency``.

    Returns ``(explanations, visualisations)``. Each explanation's
    ``visualise()`` output is flattened into the visualisations list.
    """
    explainers = list((getattr(config, "transparency", None) or {}).items())
    if not explainers:
        return [], []

    suffix = "s" if len(explainers) > 1 else ""
    raitap_log.info("Performing transparency assessment%s (%d)...", suffix, len(explainers))

    explanations: list[ExplanationResult] = []
    visualisations: list[VisualisationResult] = []
    for name, _explainer_cfg in explainers:
        runtime_kwargs = resolve_explainer_runtime_kwargs(
            config.transparency[name],
            forward_output=forward_output,
        )
        explanation = Explanation(
            config,
            name,
            model,
            data.tensor,
            input_metadata=input_metadata,
            sample_ids=data.sample_ids,
            sample_names=data.sample_ids,
            **runtime_kwargs,
        )
        explanations.append(explanation)
        visualisations.extend(explanation.visualise())
    return explanations, visualisations
