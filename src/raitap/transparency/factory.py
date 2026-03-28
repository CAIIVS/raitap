from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from raitap.configs import cfg_to_dict, resolve_run_dir, resolve_target

from .results import ConfiguredVisualiser
from .visualisers import VisualiserIncompatibilityError

if TYPE_CHECKING:
    import torch

    from raitap.models import Model

    from ..configs.schema import AppConfig
    from .explainers.base_explainer import BaseExplainer
    from .results import ExplanationResult

_TRANSPARENCY_PREFIX = "raitap.transparency."
logger = logging.getLogger(__name__)

_EXPLAINER_TOP_LEVEL_KEYS = frozenset(
    {"_target_", "algorithm", "visualisers", "constructor", "call"},
)

_VISUALISER_ENTRY_KEYS = frozenset({"_target_", "constructor", "call"})


def _raw_transparency_config(explainer_config: Any) -> dict[str, Any]:
    return cfg_to_dict(explainer_config)


def _transparency_subdict(value: Any, *, label: str) -> dict[str, Any]:
    """Normalise ``constructor`` / ``call`` blocks to a plain ``dict``."""
    if value is None:
        return {}
    if isinstance(value, dict) and not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, DictConfig):
        container = OmegaConf.to_container(value, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(
                f"Transparency {label!r} must be a mapping, got {type(container).__name__}."
            )
        return cast("dict[str, Any]", dict(container))
    raise TypeError(
        f"Transparency {label!r} must be a dict or DictConfig, got {type(value).__name__}."
    )


def _visualiser_entry_to_dict(visualiser_config: Any) -> dict[str, Any]:
    if isinstance(visualiser_config, dict):
        return dict(visualiser_config)
    if isinstance(visualiser_config, DictConfig):
        container = OmegaConf.to_container(visualiser_config, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(
                f"Each visualisers list entry must be a mapping, got {type(container).__name__}."
            )
        return cast("dict[str, Any]", dict(container))
    raise TypeError(
        "Each visualisers list entry must be a dict or DictConfig, "
        f"got {type(visualiser_config).__name__}."
    )


def _validate_explainer_top_level_keys(raw: dict[str, Any]) -> None:
    unknown = set(raw) - _EXPLAINER_TOP_LEVEL_KEYS
    if unknown:
        sorted_unknown = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown transparency explainer config keys: {sorted_unknown}. "
            "Put constructor args under 'constructor' and per-call args "
            "(e.g. target, baselines) under 'call'."
        )


def _validate_visualiser_entry_keys(entry: dict[str, Any], *, target_hint: str) -> None:
    unknown = set(entry) - _VISUALISER_ENTRY_KEYS
    if unknown:
        sorted_unknown = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown keys in visualiser config {target_hint!r}: {sorted_unknown}. "
            "Use 'constructor' for __init__ kwargs and 'call' for visualise() kwargs."
        )


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

        call_from_config = _transparency_subdict(raw_transparency_config.get("call"), label="call")
        merged_kwargs = {**call_from_config, **kwargs}

        return explainer.explain(
            model.network,
            inputs,
            run_dir=resolve_run_dir(config, subdir=f"transparency/{explainer_name}"),
            experiment_name=str(config.experiment_name),
            explainer_target=explainer_target,
            explainer_name=explainer_name,
            visualisers=visualisers,
            **merged_kwargs,
        )


def create_explainer(explainer_config: Any) -> tuple[BaseExplainer, str]:
    raw_transparency_config = _raw_transparency_config(explainer_config)
    _validate_explainer_top_level_keys(raw_transparency_config)

    constructor_plain = _transparency_subdict(
        raw_transparency_config.get("constructor"), label="constructor"
    )
    target_path = str(raw_transparency_config.get("_target_", ""))
    resolved_target = resolve_target(target_path, _TRANSPARENCY_PREFIX)

    instantiate_cfg: dict[str, Any] = {
        **constructor_plain,
        "algorithm": raw_transparency_config.get("algorithm"),
        "_target_": resolved_target,
    }

    try:
        explainer = instantiate(instantiate_cfg)
    except Exception as error:
        logger.exception("Explainer instantiation failed for target %r", target_path)
        raise ValueError(
            f"Could not instantiate explainer {target_path!r}.\n"
            "Check that _target_ points to a valid BaseExplainer subclass."
        ) from error

    return explainer, resolved_target


def create_visualisers(explainer_config: Any) -> list[ConfiguredVisualiser]:
    raw_transparency_config = _raw_transparency_config(explainer_config)
    out: list[ConfiguredVisualiser] = []

    for visualiser_config in raw_transparency_config.get("visualisers", []):
        entry = _visualiser_entry_to_dict(visualiser_config)
        visualiser_target = str(entry.get("_target_", ""))
        _validate_visualiser_entry_keys(entry, target_hint=visualiser_target or "?")

        constructor_plain = _transparency_subdict(
            entry.get("constructor"), label=f"visualiser constructor ({visualiser_target})"
        )
        call_plain = _transparency_subdict(
            entry.get("call"), label=f"visualiser call ({visualiser_target})"
        )
        resolved_target = resolve_target(visualiser_target, _TRANSPARENCY_PREFIX)

        instantiate_cfg: dict[str, Any] = {
            **constructor_plain,
            "_target_": resolved_target,
        }

        try:
            visualiser = instantiate(instantiate_cfg)
        except Exception as error:
            logger.exception("Visualiser instantiation failed for target %r", visualiser_target)
            raise ValueError(f"Could not instantiate visualiser {visualiser_target!r}.") from error

        out.append(ConfiguredVisualiser(visualiser=visualiser, call_kwargs=call_plain))

    return out


def check_explainer_visualiser_compat(
    explainer_target: str,
    algorithm: str,
    visualisers: list[ConfiguredVisualiser],
) -> None:
    for configured in visualisers:
        visualiser = configured.visualiser
        if visualiser.compatible_algorithms and algorithm not in visualiser.compatible_algorithms:
            raise VisualiserIncompatibilityError(
                framework=explainer_target,
                visualiser=type(visualiser).__name__,
                algorithm=algorithm,
                compatible_algorithms=sorted(visualiser.compatible_algorithms),
            )
