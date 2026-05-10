from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from hydra.utils import instantiate

from raitap import raitap_log
from raitap._adapter_factory import (
    AdapterSchema,
    ParsedAdapterConfig,
    instantiate_adapter,
    instantiate_visualisers,
    parse_adapter_config,
    raw_config_dict,
    resolve_call_data_sources,
)
from raitap.configs import resolve_run_dir
from raitap.models.backend import ModelBackend

from .algorithm_allowlist import ensure_algorithm_in_allowlist
from .contracts import (
    ExplainerAdapter,
    ExplanationOutputSpace,
    InputSpec,
    MethodFamily,
    explainer_output_kind,
)
from .exceptions import PayloadVisualiserIncompatibilityError, VisualiserIncompatibilityError
from .explainers.captum_explainer import CaptumExplainer
from .explainers.shap_explainer import ShapExplainer
from .results import ConfiguredVisualiser
from .semantics import explainer_capability

if TYPE_CHECKING:
    import torch

    from raitap.models import Model

    from ..configs.schema import AppConfig
    from .results import ExplanationResult

_TRANSPARENCY_PREFIX = "raitap.transparency."

_SCHEMA = AdapterSchema(
    domain="transparency",
    entity="explainer",
    subdict_namespace="Transparency",
    target_prefix=_TRANSPARENCY_PREFIX,
    visualiser_prefix=_TRANSPARENCY_PREFIX,
    top_level_keys=frozenset(
        {"_target_", "algorithm", "visualisers", "constructor", "call", "raitap"}
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
        "Put constructor args under 'constructor', library kwargs "
        "(e.g. target, baselines) under 'call', and RAITAP-owned options "
        "under 'raitap'."
    ),
    removed_raitap_keys={
        "max_batch_size": "raitap.max_batch_size has been removed; use raitap.batch_size instead."
    },
)

_PARSED_EXPLAINER_CONFIG_CACHE: dict[int, ParsedAdapterConfig] = {}


def _parse_explainer_config(explainer_config: Any) -> ParsedAdapterConfig:
    return parse_adapter_config(explainer_config, _SCHEMA)


def _require_model_backend(model: object) -> ModelBackend:
    backend = getattr(model, "backend", None)
    if not isinstance(backend, ModelBackend):
        raise TypeError(
            "Explanation expects a raitap.models.Model or another object exposing a "
            "'.backend' that is a ModelBackend instance. "
            f"Got {type(backend).__name__!r} instead."
        )
    return backend


def check_explainer_visualiser_payload_compat(
    explainer: object,
    explainer_target: str,
    visualisers: list[ConfiguredVisualiser],
) -> None:
    kind = explainer_output_kind(explainer)
    for configured in visualisers:
        visualiser = configured.visualiser
        supported = getattr(type(visualiser), "supported_payload_kinds", frozenset())
        if len(supported) == 0:
            continue
        if kind not in supported:
            raise PayloadVisualiserIncompatibilityError(
                explainer_target=explainer_target,
                visualiser=type(visualiser).__name__,
                output_payload_kind=kind.value,
                supported_payload_kinds=[k.value for k in sorted(supported, key=lambda x: x.value)],
            )


def check_explainer_visualiser_semantic_compat(
    explainer: object,
    explainer_target: str,
    visualisers: list[ConfiguredVisualiser],
) -> None:
    if not _requires_registry_semantics(explainer, explainer_target):
        return

    capability = explainer_capability(explainer)

    for configured in visualisers:
        visualiser = configured.visualiser
        supported_method_families = _enum_frozenset(
            getattr(type(visualiser), "supported_method_families", frozenset()),
            MethodFamily,
        )
        if supported_method_families and not capability.method_families.intersection(
            supported_method_families
        ):
            raise ValueError(
                f"Visualiser {type(visualiser).__name__!r} does not support explainer "
                f"method families {sorted(f.value for f in capability.method_families)}. "
                "Its supported method families are "
                f"{sorted(f.value for f in supported_method_families)}."
            )

        supported_output_spaces = _enum_frozenset(
            getattr(type(visualiser), "supported_output_spaces", frozenset()),
            ExplanationOutputSpace,
        )
        if not supported_output_spaces:
            continue
        if capability.candidate_output_spaces.intersection(supported_output_spaces):
            continue
        raise ValueError(
            f"Visualiser {type(visualiser).__name__!r} does not support explainer "
            "candidate output spaces "
            f"{sorted(s.value for s in capability.candidate_output_spaces)}. "
            "Its supported output spaces are "
            f"{sorted(s.value for s in supported_output_spaces)}."
        )


class Explanation:
    def __new__(
        cls,
        config: AppConfig,
        explainer_name: str,
        model: Model,
        inputs: torch.Tensor,
        input_metadata: InputSpec | dict[str, Any] | None = None,
        sample_ids: list[str] | None = None,
        sample_names: list[str] | None = None,
        **kwargs: Any,
    ) -> ExplanationResult:
        explainer_config = config.transparency[explainer_name]
        parsed = _parse_explainer_config(explainer_config)
        algorithm = str(parsed.algorithm or "")
        cache_key = id(explainer_config)
        _PARSED_EXPLAINER_CONFIG_CACHE[cache_key] = parsed
        try:
            explainer, explainer_target = create_explainer(explainer_config)
            visualisers = create_visualisers(explainer_config)
            check_explainer_visualiser_compat(explainer_target, algorithm, visualisers)
            check_explainer_visualiser_payload_compat(explainer, explainer_target, visualisers)
            check_explainer_visualiser_semantic_compat(
                explainer,
                explainer_target,
                visualisers,
            )
            backend = _require_model_backend(model)
            explainer.check_backend_compat(backend)

            call_from_config = dict(parsed.call)
            raitap_cfg = dict(parsed.raitap)
            if sample_names is not None:
                if "sample_names" in raitap_cfg and raitap_cfg["sample_names"] != sample_names:
                    raitap_log.debug(
                        "Runtime sample_names for explainer %r override "
                        "raitap.sample_names from config.",
                        explainer_name,
                    )
                raitap_cfg["sample_names"] = sample_names
            if sample_ids is not None:
                raitap_cfg["sample_ids"] = sample_ids
            if input_metadata is not None:
                raitap_cfg["input_metadata"] = input_metadata

            merged_kwargs = resolve_call_data_sources(
                {**call_from_config, **kwargs}, log_label="call"
            )
            merged_kwargs = backend._prepare_kwargs(merged_kwargs)

            return explainer.explain(
                backend.as_model_for_explanation(),
                inputs,
                backend=backend,
                run_dir=resolve_run_dir(config, subdir=f"transparency/{explainer_name}"),
                experiment_name=str(getattr(config, "experiment_name", "")),
                explainer_target=explainer_target,
                explainer_name=explainer_name,
                visualisers=visualisers,
                raitap_kwargs=raitap_cfg,
                **merged_kwargs,
            )
        finally:
            _PARSED_EXPLAINER_CONFIG_CACHE.pop(cache_key, None)


def create_explainer(explainer_config: Any) -> tuple[ExplainerAdapter, str]:
    parsed = _PARSED_EXPLAINER_CONFIG_CACHE.get(id(explainer_config))
    if parsed is None:
        parsed = _parse_explainer_config(explainer_config)
    return instantiate_adapter(
        parsed,
        protocol=ExplainerAdapter,
        schema=_SCHEMA,
        instantiate_error_hint=(
            "Check that _target_ points to a valid ExplainerAdapter implementation "
            "(e.g. AttributionOnlyExplainer or FullExplainer subclass)."
        ),
        type_error_hint=(
            "Configured explainers must have callable explain() and check_backend_compat() methods."
        ),
        instantiate_fn=lambda cfg: instantiate(cfg),
    )


def create_visualisers(explainer_config: Any) -> list[ConfiguredVisualiser]:
    return instantiate_visualisers(
        explainer_config,
        schema=_SCHEMA,
        wrap=lambda viz, call: ConfiguredVisualiser(visualiser=viz, call_kwargs=call),
        instantiate_fn=lambda cfg: instantiate(cfg),
    )


def check_explainer_visualiser_compat(
    explainer_target: str,
    algorithm: str,
    visualisers: list[ConfiguredVisualiser],
) -> None:
    for configured in visualisers:
        visualiser = configured.visualiser
        ensure_algorithm_in_allowlist(
            algorithm,
            visualiser.compatible_algorithms,
            error_cls=VisualiserIncompatibilityError,
            framework=explainer_target,
            visualiser=type(visualiser).__name__,
            algorithm=algorithm,
            compatible_algorithms=sorted(visualiser.compatible_algorithms),
        )


def _requires_registry_semantics(explainer: object, explainer_target: str) -> bool:
    target = explainer_target.lower()
    class_name = type(explainer).__name__.lower()
    if "shap" in target or "captum" in target or "shap" in class_name or "captum" in class_name:
        return True
    algorithm = str(getattr(explainer, "algorithm", ""))
    return (
        algorithm in ShapExplainer.algorithm_registry
        or algorithm in CaptumExplainer.algorithm_registry
    )


def _enum_frozenset(value: object, enum_type: type[Any]) -> frozenset[Any]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        iterable = [value]
    else:
        try:
            iterable = list(value)  # type: ignore[arg-type]
        except TypeError:
            iterable = [value]
    out = []
    for item in iterable:
        if isinstance(item, enum_type):
            out.append(item)
        else:
            out.append(enum_type(str(item)))
    return frozenset(out)


# Internal-test back-compat: a couple of older tests touch these names. Keep
# them as thin wrappers so the test surface is unchanged across the refactor.
def _raw_transparency_config(explainer_config: Any) -> dict[str, Any]:
    return raw_config_dict(explainer_config)


def _resolve_call_data_sources(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    return resolve_call_data_sources(call_kwargs, log_label="call")
