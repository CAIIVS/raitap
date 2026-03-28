from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hydra.utils import instantiate

from raitap.configs import cfg_to_dict, resolve_run_dir, resolve_target

from .visualisers import VisualiserIncompatibilityError

if TYPE_CHECKING:
    import torch

    from raitap.models import Model

    from ..configs.schema import AppConfig
    from .explainers.base_explainer import BaseExplainer
    from .results import ExplanationResult
    from .visualisers import BaseVisualiser

_TRANSPARENCY_PREFIX = "raitap.transparency."
logger = logging.getLogger(__name__)


def _raw_transparency_config(explainer_config: Any) -> dict[str, Any]:
    return cfg_to_dict(explainer_config)


class Explanation:
    def __new__(
        cls,
        config: AppConfig,
        explainer_name: str,
        model: Model,
        inputs: torch.Tensor,
        **kwargs: Any,
    ) -> ExplanationResult:
        explainer_config = config.transparency[explainer_name]
        raw_transparency_config = _raw_transparency_config(explainer_config)
        algorithm = str(raw_transparency_config.get("algorithm", ""))
        explainer, explainer_target = create_explainer(explainer_config)
        visualisers = create_visualisers(explainer_config)
        check_explainer_visualiser_compat(explainer_target, algorithm, visualisers)

        return explainer.explain(
            model.network,
            inputs,
            run_dir=resolve_run_dir(config, subdir=f"transparency/{explainer_name}"),
            experiment_name=str(getattr(config, "experiment_name", "")),
            explainer_target=explainer_target,
            explainer_name=explainer_name,
            visualisers=visualisers,
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
        logger.exception("Explainer instantiation failed for target %r", target_path)
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
            logger.exception("Visualiser instantiation failed for target %r", visualiser_target)
            raise ValueError(f"Could not instantiate visualiser {visualiser_target!r}.") from error

        visualisers.append(visualiser)

    return visualisers


def check_explainer_visualiser_compat(
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
