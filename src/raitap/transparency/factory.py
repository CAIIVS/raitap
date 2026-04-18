from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from raitap.configs import cfg_to_dict, resolve_run_dir, resolve_target
from raitap.data import load_tensor_from_source
from raitap.models.backend import ModelBackend

from .algorithm_allowlist import ensure_algorithm_in_allowlist
from .contracts import ExplainerAdapter, explainer_output_kind
from .exceptions import PayloadVisualiserIncompatibilityError, VisualiserIncompatibilityError
from .results import ConfiguredVisualiser

if TYPE_CHECKING:
    import torch

    from raitap.models import Model

    from ..configs.schema import AppConfig
    from .results import ExplanationResult

_TRANSPARENCY_PREFIX = "raitap.transparency."
logger = logging.getLogger(__name__)
_ALIBI_BSL_WARNING_EMITTED = False
_PARSED_EXPLAINER_CONFIG_CACHE: dict[int, _ParsedExplainerConfig] = {}

_EXPLAINER_TOP_LEVEL_KEYS = frozenset(
    {"_target_", "algorithm", "visualisers", "constructor", "call", "raitap"},
)

_VISUALISER_ENTRY_KEYS = frozenset({"_target_", "constructor", "call"})
_RAITAP_KEYS = frozenset(
    {
        "batch_size",
        "show_progress",
        "progress_desc",
        "sample_names",
        "show_sample_names",
    }
)
_MISPLACED_RAITAP_CALL_WARNING_KEYS = frozenset(
    {"batch_size", "show_progress", "progress_desc", "sample_names", "show_sample_names"}
)
_REMOVED_RAITAP_KEYS = {
    "max_batch_size": "raitap.max_batch_size has been removed; use raitap.batch_size instead."
}

# Keys that identify a dict value in ``call:`` as a data-source reference.
# A value matches when it is a plain dict containing at least ``source``.
_DATA_SOURCE_KEYS = frozenset({"source", "n_samples"})


class _ParsedExplainerConfig:
    def __init__(
        self,
        *,
        raw: dict[str, Any],
        target_path: str,
        resolved_target: str,
        algorithm: Any,
        constructor: dict[str, Any],
        call: dict[str, Any],
        raitap: dict[str, Any],
    ) -> None:
        self.raw = raw
        self.target_path = target_path
        self.resolved_target = resolved_target
        self.algorithm = algorithm
        self.constructor = constructor
        self.call = call
        self.raitap = raitap


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
            "Put constructor args under 'constructor', library kwargs "
            "(e.g. target, baselines) under 'call', and RAITAP-owned options "
            "under 'raitap'."
        )


def _validate_visualiser_entry_keys(entry: dict[str, Any], *, target_hint: str) -> None:
    unknown = set(entry) - _VISUALISER_ENTRY_KEYS
    if unknown:
        sorted_unknown = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown keys in visualiser config {target_hint!r}: {sorted_unknown}. "
            "Use 'constructor' for __init__ kwargs and 'call' for visualise() kwargs."
        )


def _validate_raitap_keys(raitap_cfg: dict[str, Any], *, explainer_name: str) -> None:
    for key, message in _REMOVED_RAITAP_KEYS.items():
        if key in raitap_cfg:
            raise ValueError(message)

    unknown = set(raitap_cfg) - _RAITAP_KEYS
    if not unknown:
        return

    sorted_unknown = ", ".join(sorted(unknown))
    sorted_valid = ", ".join(sorted(_RAITAP_KEYS))
    logger.warning(
        "Unknown transparency.raitap keys for explainer %r: %s. Supported RAITAP keys: %s.",
        explainer_name,
        sorted_unknown,
        sorted_valid,
    )


def _warn_on_misplaced_raitap_call_keys(call_cfg: dict[str, Any], *, explainer_name: str) -> None:
    misplaced = sorted(set(call_cfg).intersection(_MISPLACED_RAITAP_CALL_WARNING_KEYS))
    if not misplaced:
        return

    keys = ", ".join(misplaced)
    logger.warning(
        "Explainer %r has RAITAP-owned keys under 'call:': %s. These keys usually belong "
        "under 'raitap:' while 'call:' is intended for library kwargs only.",
        explainer_name,
        keys,
    )


def _parse_explainer_config(explainer_config: Any) -> _ParsedExplainerConfig:
    raw_transparency_config = _raw_transparency_config(explainer_config)
    _validate_explainer_top_level_keys(raw_transparency_config)

    target_path = str(raw_transparency_config.get("_target_", ""))
    resolved_target = resolve_target(target_path, _TRANSPARENCY_PREFIX)
    constructor_plain = _transparency_subdict(
        raw_transparency_config.get("constructor"), label="constructor"
    )
    call_plain = _transparency_subdict(raw_transparency_config.get("call"), label="call")
    raitap_plain = _transparency_subdict(raw_transparency_config.get("raitap"), label="raitap")
    explainer_name = resolved_target or target_path or "?"
    _validate_raitap_keys(raitap_plain, explainer_name=explainer_name)
    _warn_on_misplaced_raitap_call_keys(call_plain, explainer_name=explainer_name)

    return _ParsedExplainerConfig(
        raw=raw_transparency_config,
        target_path=target_path,
        resolved_target=resolved_target,
        algorithm=raw_transparency_config.get("algorithm"),
        constructor=constructor_plain,
        call=call_plain,
        raitap=raitap_plain,
    )


def _resolve_call_data_sources(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve any data-source references inside ``call:`` kwargs.

    A value is treated as a data-source reference when it is a plain ``dict``
    whose keys are a subset of ``{"source", "n_samples"}`` and ``"source"`` is
    present.  Such values are replaced with the loaded tensor so the downstream
    explainer receives a ``torch.Tensor`` instead of a raw config dict.

    Example YAML under ``call:``::

        background_data:
          source: "imagenet_samples"   # required: named sample, URL, or local path
          n_samples: 50                # optional: randomly subsample N rows

    This is a generic mechanism — any ``call:`` parameter whose value matches
    the pattern above will be resolved, regardless of explainer type.
    """
    resolved: dict[str, Any] = {}
    for key, value in call_kwargs.items():
        if isinstance(value, dict) and set(value).issubset(_DATA_SOURCE_KEYS) and "source" in value:
            source = value["source"]
            n_samples = value.get("n_samples")
            if n_samples is not None and not isinstance(n_samples, int):
                raise TypeError(
                    f"call.{key}.n_samples must be an int, got {type(n_samples).__name__}."
                )
            logger.info(
                "Resolving call kwarg %r as data source %r (n_samples=%s)",
                key,
                source,
                n_samples,
            )
            resolved[key] = load_tensor_from_source(str(source), n_samples=n_samples)
        else:
            resolved[key] = value
    return resolved


def _instantiate_explainer_from_parsed(
    parsed: _ParsedExplainerConfig,
) -> tuple[ExplainerAdapter, str]:
    instantiate_cfg: dict[str, Any] = {
        **parsed.constructor,
        "algorithm": parsed.algorithm,
        "_target_": parsed.resolved_target,
    }

    try:
        explainer = instantiate(instantiate_cfg)
    except Exception as error:
        logger.exception("Explainer instantiation failed for target %r", parsed.target_path)
        raise ValueError(
            f"Could not instantiate explainer {parsed.target_path!r}.\n"
            "Check that _target_ points to a valid ExplainerAdapter implementation "
            "(e.g. AttributionOnlyExplainer or FullExplainer subclass)."
        ) from error

    if not isinstance(explainer, ExplainerAdapter):
        raise ValueError(
            f"Instantiated explainer {parsed.target_path!r} does not implement ExplainerAdapter. "
            "Configured explainers must have callable explain() and check_backend_compat() methods."
        )

    return cast("ExplainerAdapter", explainer), parsed.resolved_target


def _require_model_backend(model: object) -> ModelBackend:
    backend = getattr(model, "backend", None)
    if not isinstance(backend, ModelBackend):
        raise TypeError(
            "Explanation expects a raitap.models.Model or another object exposing a "
            "'.backend' that is a ModelBackend instance. "
            f"Got {type(backend).__name__!r} instead."
        )
    return backend


def _maybe_emit_third_party_license_warnings(explainer: object) -> None:
    """
    Emit one-time warnings for explainers that bundle non-GPL third-party license terms.
    """
    global _ALIBI_BSL_WARNING_EMITTED
    if not getattr(type(explainer), "ALIBI_BSL_LICENSE_WARNING", False):
        return
    if _ALIBI_BSL_WARNING_EMITTED:
        return
    _ALIBI_BSL_WARNING_EMITTED = True
    logger.warning(
        "This run uses Alibi Explain, which is licensed under Seldon's Business Source "
        "License 1.1 (BSL 1.1), not GPLv3. Non-production use is allowed on Seldon's terms; "
        "production or commercial use may require a separate license from Seldon. "
        "See https://github.com/SeldonIO/alibi/blob/master/LICENSE and Seldon's licensing FAQ. "
        "RAITAP (GPLv3) does not relicense Alibi."
    )


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


class Explanation:
    def __new__(
        cls,
        config: AppConfig,
        explainer_name: str,
        model: Model,
        inputs: torch.Tensor,
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
            _maybe_emit_third_party_license_warnings(explainer)
            visualisers = create_visualisers(explainer_config)
            check_explainer_visualiser_compat(explainer_target, algorithm, visualisers)
            check_explainer_visualiser_payload_compat(explainer, explainer_target, visualisers)
            backend = _require_model_backend(model)
            explainer.check_backend_compat(backend)

            call_from_config = dict(parsed.call)
            raitap_cfg = dict(parsed.raitap)
            if sample_names is not None:
                if "sample_names" in raitap_cfg and raitap_cfg["sample_names"] != sample_names:
                    logger.debug(
                        "Runtime sample_names for explainer %r override "
                        "raitap.sample_names from config.",
                        explainer_name,
                    )
                raitap_cfg["sample_names"] = sample_names

            merged_kwargs = _resolve_call_data_sources({**call_from_config, **kwargs})
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
    return _instantiate_explainer_from_parsed(parsed)


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
        ensure_algorithm_in_allowlist(
            algorithm,
            visualiser.compatible_algorithms,
            error_cls=VisualiserIncompatibilityError,
            framework=explainer_target,
            visualiser=type(visualiser).__name__,
            algorithm=algorithm,
            compatible_algorithms=sorted(visualiser.compatible_algorithms),
        )
