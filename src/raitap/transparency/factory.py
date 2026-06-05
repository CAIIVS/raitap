from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hydra.utils import instantiate

from raitap.configs.adapter_factory import (
    AdapterSchema,
    ParsedAdapterConfig,
    instantiate_adapter,
    instantiate_visualisers,
    parse_adapter_config,
    raw_config_dict,
    resolve_call_data_sources,
)
from raitap.models.backend import ModelBackend

from .algorithm_allowlist import ensure_algorithm_in_allowlist
from .contracts import (
    ExplainerAdapter,
    ExplanationOutputSpace,
    MethodFamily,
    explainer_output_kind,
)
from .exceptions import VisualiserIncompatibilityError
from .explainers.captum_explainer import CaptumExplainer
from .explainers.shap_explainer import ShapExplainer
from .results import ConfiguredVisualiser
from .semantics import explainer_capability

if TYPE_CHECKING:
    from raitap.types import TaskKind

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
            # Library-agnostic baseline / reference input, routed to the
            # adapter's ``baseline_kwarg_name`` by ``apply_config_baseline``.
            "baseline",
            # Detection-task knobs consumed by ``explain_detection`` —
            # ``score_threshold`` / ``max_boxes`` / ``iou_threshold``.
            "detection",
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
            "Transparency setup expects a raitap.models.Model or another object exposing "
            "a '.backend' that is a ModelBackend instance. "
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
            raise VisualiserIncompatibilityError(
                visualiser=type(visualiser).__name__,
                axis="payload kind",
                declared=kind.value,
                accepted=", ".join(k.value for k in sorted(supported, key=lambda x: x.value)),
            )


def check_explainer_visualiser_semantic_compat(
    explainer: object,
    explainer_target: str,
    visualisers: list[ConfiguredVisualiser],
    *,
    task_kind: TaskKind | None = None,
) -> None:
    if not _requires_registry_semantics(explainer, explainer_target):
        return

    capability = explainer_capability(explainer, task_kind=task_kind)

    for configured in visualisers:
        visualiser = configured.visualiser
        supported_method_families = _enum_frozenset(
            getattr(type(visualiser), "supported_method_families", frozenset()),
            MethodFamily,
        )
        if supported_method_families and not (
            capability.method_families & supported_method_families
        ):
            raise VisualiserIncompatibilityError(
                visualiser=type(visualiser).__name__,
                axis="method families",
                declared=", ".join(sorted(f.value for f in capability.method_families)),
                accepted=", ".join(sorted(f.value for f in supported_method_families)),
            )

        supported_output_spaces = _enum_frozenset(
            getattr(type(visualiser), "supported_output_spaces", frozenset()),
            ExplanationOutputSpace,
        )
        if not supported_output_spaces:
            continue
        if capability.candidate_output_spaces & supported_output_spaces:
            continue
        raise VisualiserIncompatibilityError(
            visualiser=type(visualiser).__name__,
            axis="candidate output spaces",
            declared=", ".join(sorted(s.value for s in capability.candidate_output_spaces)),
            accepted=", ".join(sorted(s.value for s in supported_output_spaces)),
        )


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
        type_error_hint=("Configured explainers must have a callable explain() method."),
        instantiate_fn=instantiate,
    )


def create_visualisers(explainer_config: Any) -> list[ConfiguredVisualiser]:
    return instantiate_visualisers(
        explainer_config,
        schema=_SCHEMA,
        wrap=lambda viz, call: ConfiguredVisualiser(visualiser=viz, call_kwargs=call),
        instantiate_fn=instantiate,
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
            visualiser=type(visualiser).__name__,
            axis="algorithm",
            declared=algorithm,
            accepted=", ".join(sorted(visualiser.compatible_algorithms)),
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
