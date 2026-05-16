"""Hydra-driven factory for robustness assessors and visualisers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hydra.utils import instantiate

from raitap.configs import resolve_run_dir
from raitap.configs.adapter_factory import (
    AdapterSchema,
    ParsedAdapterConfig,
    instantiate_adapter,
    instantiate_visualisers,
    parse_adapter_config,
    per_image_transform_from_config,
    raw_config_dict,
    resolve_call_data_sources,
)
from raitap.models.backend import ModelBackend

from .contracts import AssessorAdapter
from .exceptions import MethodKindVisualiserIncompatibilityError, MissingTargetsError
from .results import ConfiguredRobustnessVisualiser

if TYPE_CHECKING:
    import torch

    from raitap.configs.schema import AppConfig
    from raitap.models import Model

    from .results import RobustnessResult

_ROBUSTNESS_PREFIX = "raitap.robustness.assessors."
_VISUALISER_PREFIX = "raitap.robustness.visualisers."

_SCHEMA = AdapterSchema(
    domain="robustness",
    entity="assessor",
    subdict_namespace="Robustness",
    target_prefix=_ROBUSTNESS_PREFIX,
    visualiser_prefix=_VISUALISER_PREFIX,
    top_level_keys=frozenset(
        {"_target_", "algorithm", "constructor", "call", "raitap", "visualisers"}
    ),
    raitap_keys=frozenset(
        {
            "batch_size",
            "input_metadata",
            "show_progress",
            "progress_desc",
            "sample_ids",
            "sample_names",
            "show_sample_names",
        }
    ),
    top_level_error_hint=(
        "Put __init__ kwargs under 'constructor', per-call kwargs under 'call', "
        "and RAITAP-owned options under 'raitap'. Where the attack hyperparameters "
        "(eps / alpha / steps) live depends on the underlying library: torchattacks "
        "treats them as constructor args, foolbox as call args."
    ),
)

_PARSED_ASSESSOR_CONFIG_CACHE: dict[int, ParsedAdapterConfig] = {}


def _parse_assessor_config(assessor_config: Any) -> ParsedAdapterConfig:
    return parse_adapter_config(assessor_config, _SCHEMA)


def _require_model_backend(model: object) -> ModelBackend:
    backend = getattr(model, "backend", None)
    if not isinstance(backend, ModelBackend):
        raise TypeError(
            "RobustnessAssessment expects a raitap.models.Model or another object exposing a "
            "'.backend' that is a ModelBackend instance. "
            f"Got {type(backend).__name__!r} instead."
        )
    return backend


def check_assessor_visualiser_compat(
    assessor: AssessorAdapter,
    assessor_target: str,
    visualisers: list[ConfiguredRobustnessVisualiser],
) -> None:
    """Enforce ``MethodKind`` ↔ ``supported_method_kinds`` at parse time."""
    method_kind = assessor.method_kind
    for configured in visualisers:
        visualiser = configured.visualiser
        supported = getattr(type(visualiser), "supported_method_kinds", frozenset())
        if not supported:
            continue
        if method_kind not in supported:
            raise MethodKindVisualiserIncompatibilityError(
                assessor_target=assessor_target,
                visualiser=type(visualiser).__name__,
                assessor_method_kind=method_kind.value,
                supported_method_kinds=[k.value for k in sorted(supported)],
            )


class RobustnessAssessment:
    """Factory entry-point mirroring ``raitap.transparency.factory.Explanation``."""

    def __new__(
        cls,
        config: AppConfig,
        assessor_name: str,
        model: Model,
        inputs: torch.Tensor,
        targets: torch.Tensor | None,
        input_metadata: Any | None = None,
        sample_ids: list[str] | None = None,
        sample_names: list[str] | None = None,
        **kwargs: Any,
    ) -> RobustnessResult:
        if targets is None:
            raise MissingTargetsError(assessor_name)

        assessor_config = config.robustness[assessor_name]
        parsed = _parse_assessor_config(assessor_config)
        cache_key = id(assessor_config)
        _PARSED_ASSESSOR_CONFIG_CACHE[cache_key] = parsed
        try:
            assessor, assessor_target = create_assessor(assessor_config)
            visualisers = create_robustness_visualisers(assessor_config)
            check_assessor_visualiser_compat(assessor, assessor_target, visualisers)
            backend = _require_model_backend(model)
            assessor.check_backend_compat(backend)

            call_from_config = dict(parsed.call)
            raitap_cfg = dict(parsed.raitap)
            if sample_names is not None:
                raitap_cfg["sample_names"] = sample_names
            if sample_ids is not None:
                raitap_cfg["sample_ids"] = sample_ids
            if input_metadata is not None:
                raitap_cfg["input_metadata"] = input_metadata

            merged_kwargs = resolve_call_data_sources(
                {**call_from_config, **kwargs},
                log_label="robustness call",
                per_image_transform=per_image_transform_from_config(config),
            )
            merged_kwargs = backend._prepare_kwargs(merged_kwargs)

            return assessor.assess(
                backend.as_model_for_explanation(),
                inputs,
                targets,
                backend=backend,
                run_dir=resolve_run_dir(config, subdir=f"robustness/{assessor_name}"),
                experiment_name=str(getattr(config, "experiment_name", "")),
                assessor_target=assessor_target,
                assessor_name=assessor_name,
                visualisers=visualisers,
                raitap_kwargs=raitap_cfg,
                **merged_kwargs,
            )
        finally:
            _PARSED_ASSESSOR_CONFIG_CACHE.pop(cache_key, None)


def create_assessor(assessor_config: Any) -> tuple[AssessorAdapter, str]:
    parsed = _PARSED_ASSESSOR_CONFIG_CACHE.get(id(assessor_config))
    if parsed is None:
        parsed = _parse_assessor_config(assessor_config)
    return instantiate_adapter(
        parsed,
        protocol=AssessorAdapter,
        schema=_SCHEMA,
        instantiate_error_hint=(
            "Check that _target_ points to a valid AssessorAdapter implementation "
            "(e.g. EmpiricalAttackAssessor or FormalVerificationAssessor subclass)."
        ),
        type_error_hint=(
            "Configured assessors must have callable assess() and check_backend_compat() methods, "
            "and a ``method_kind`` attribute."
        ),
        instantiate_fn=instantiate,
    )


def create_robustness_visualisers(
    assessor_config: Any,
) -> list[ConfiguredRobustnessVisualiser]:
    return instantiate_visualisers(
        assessor_config,
        schema=_SCHEMA,
        wrap=lambda viz, call: ConfiguredRobustnessVisualiser(visualiser=viz, call_kwargs=call),
        instantiate_fn=instantiate,
    )


# Internal-test back-compat thin wrappers (keep the test surface stable).
def _raw_assessor_config(assessor_config: Any) -> dict[str, Any]:
    return raw_config_dict(assessor_config)


def _resolve_call_data_sources(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    return resolve_call_data_sources(call_kwargs, log_label="robustness call")
