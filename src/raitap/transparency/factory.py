"""Config helpers for building transparency objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from hydra.utils import instantiate

from ..configs.factory_utils import cfg_to_dict, resolve_run_dir, resolve_target
from .explainers import BaseExplainer
from .methods_registry import VisualiserIncompatibilityError
from .results import ExplanationResult
from .visualisers import BaseVisualiser

_TRANSPARENCY_PREFIX = "raitap.transparency."

if TYPE_CHECKING:
    from ..configs.schema import AppConfig


def _raw_transparency_config(explainer_config: Any) -> dict[str, Any]:
    return cfg_to_dict(explainer_config)


class Explanation:
    def __new__(
        cls,
        app_config: AppConfig,
        explainer_name: str,
        model: Any,  # Can be Model or nn.Module
        inputs: torch.Tensor,
        **kwargs: Any,
    ) -> ExplanationResult:
        explainer_config = app_config.explainers[explainer_name]
        raw_transparency_config = _raw_transparency_config(explainer_config)
        algorithm = str(raw_transparency_config.get("algorithm", ""))
        explainer, explainer_target = create_explainer(explainer_config)
        visualisers = create_visualisers(explainer_config)
        validate_visualisers(explainer_target, algorithm, visualisers)

        # Extract nn.Module if a Model instance is passed
        network = getattr(model, "network", model)

        return explainer.explain(
            network,
            inputs,
            run_dir=resolve_run_dir(app_config, subdir=f"transparency/{explainer_name}"),
            experiment_name=str(getattr(app_config, "experiment_name", "")),
            explainer_target=explainer_target,
            **kwargs,
        )


def create_explainer(explainer_config: Any) -> tuple[BaseExplainer, str]:
    raw_transparency_config = _raw_transparency_config(explainer_config)
    explainer_config = {
        key: value for key, value in raw_transparency_config.items() if key != "visualisers"
    }
    target_path = str(explainer_config.get("_target_", ""))
    resolved_target = resolve_target(target_path, _TRANSPARENCY_PREFIX)
    explainer_config["_target_"] = resolved_target

    try:
        explainer = instantiate(explainer_config)
    except Exception as error:
        raise ValueError(
            f"Could not instantiate explainer {target_path!r}.\n"
            "Check that _target_ points to a valid BaseExplainer subclass."
        ) from error

    return explainer, resolved_target


def create_visualisers(explainer_config: Any) -> list[BaseVisualiser]:
    raw_transparency_config = _raw_transparency_config(explainer_config)
    visualisers: list[BaseVisualiser] = []

    for visualiser_config in raw_transparency_config.get("visualisers", []):
        visualiser_target = (
            str(visualiser_config.get("_target_", ""))
            if isinstance(visualiser_config, dict)
            else ""
        )
        resolved_config = (
            {
                **visualiser_config,
                "_target_": resolve_target(visualiser_target, _TRANSPARENCY_PREFIX),
            }
            if isinstance(visualiser_config, dict) and "_target_" in visualiser_config
            else visualiser_config
        )

        try:
            visualiser = instantiate(resolved_config)
        except Exception as error:
            raise ValueError(f"Could not instantiate visualiser {visualiser_target!r}.") from error

        visualisers.append(visualiser)

    return visualisers


def validate_visualisers(
    explainer_target: str,
    algorithm: str,
    visualisers: list[BaseVisualiser],
) -> None:
    for visualiser in visualisers:
        if visualiser.compatible_algorithms and algorithm not in visualiser.compatible_algorithms:
            raise VisualiserIncompatibilityError(
                framework=explainer_target,
                visualiser=type(visualiser).__name__,
                algorithm=algorithm,
                compatible_algorithms=sorted(visualiser.compatible_algorithms),
            )


__all__ = [
    "Explanation",
    "create_explainer",
    "create_visualisers",
    "validate_visualisers",
]
