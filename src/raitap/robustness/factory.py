"""Hydra-driven factory for robustness assessors and visualisers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from raitap import raitap_log
from raitap.configs import cfg_to_dict, resolve_run_dir, resolve_target
from raitap.data import load_tensor_from_source
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

_ASSESSOR_TOP_LEVEL_KEYS = frozenset(
    {"_target_", "algorithm", "constructor", "call", "raitap", "visualisers"}
)
_VISUALISER_ENTRY_KEYS = frozenset({"_target_", "constructor", "call"})
_RAITAP_KEYS = frozenset(
    {
        "batch_size",
        "input_metadata",
        "show_progress",
        "progress_desc",
        "sample_ids",
        "sample_names",
        "show_sample_names",
    }
)
_MISPLACED_RAITAP_CALL_WARNING_KEYS = _RAITAP_KEYS

_DATA_SOURCE_KEYS = frozenset({"source", "n_samples"})

_PARSED_ASSESSOR_CONFIG_CACHE: dict[int, _ParsedAssessorConfig] = {}


class _ParsedAssessorConfig:
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


def _raw_assessor_config(assessor_config: Any) -> dict[str, Any]:
    return cfg_to_dict(assessor_config)


def _subdict(value: Any, *, label: str) -> dict[str, Any]:
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
                f"Robustness {label!r} must be a mapping, got {type(container).__name__}."
            )
        return cast("dict[str, Any]", dict(container))
    raise TypeError(
        f"Robustness {label!r} must be a dict or DictConfig, got {type(value).__name__}."
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


def _validate_assessor_top_level_keys(raw: dict[str, Any]) -> None:
    unknown = set(raw) - _ASSESSOR_TOP_LEVEL_KEYS
    if unknown:
        sorted_unknown = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown robustness assessor config keys: {sorted_unknown}. "
            "Put __init__ kwargs under 'constructor', per-call kwargs under 'call', "
            "and RAITAP-owned options under 'raitap'. Where the attack hyperparameters "
            "(eps / alpha / steps) live depends on the underlying library: torchattacks "
            "treats them as constructor args, foolbox as call args."
        )


def _validate_visualiser_entry_keys(entry: dict[str, Any], *, target_hint: str) -> None:
    unknown = set(entry) - _VISUALISER_ENTRY_KEYS
    if unknown:
        sorted_unknown = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown keys in visualiser config {target_hint!r}: {sorted_unknown}. "
            "Use 'constructor' for __init__ kwargs and 'call' for visualise() kwargs."
        )


def _validate_raitap_keys(raitap_cfg: dict[str, Any], *, assessor_name: str) -> None:
    unknown = set(raitap_cfg) - _RAITAP_KEYS
    if not unknown:
        return
    sorted_unknown = ", ".join(sorted(unknown))
    sorted_valid = ", ".join(sorted(_RAITAP_KEYS))

    raitap_log.warn(
        f"Unknown robustness.raitap keys for assessor {assessor_name!r}: "
        f"{sorted_unknown}. Supported RAITAP keys: {sorted_valid}.",
    )


def _warn_on_misplaced_raitap_call_keys(call_cfg: dict[str, Any], *, assessor_name: str) -> None:
    misplaced = sorted(set(call_cfg).intersection(_MISPLACED_RAITAP_CALL_WARNING_KEYS))
    if not misplaced:
        return
    keys = ", ".join(misplaced)

    raitap_log.warn(
        f"Assessor {assessor_name!r} has RAITAP-owned keys under 'call:': {keys}. "
        "These keys belong under 'raitap:' while 'call:' is intended for library "
        "kwargs only.",
    )


def _migrate_misplaced_raitap_call_keys(
    call_cfg: dict[str, Any],
    raitap_cfg: dict[str, Any],
) -> None:
    misplaced = sorted(set(call_cfg).intersection(_MISPLACED_RAITAP_CALL_WARNING_KEYS))
    for key in misplaced:
        value = call_cfg.pop(key)
        raitap_cfg.setdefault(key, value)


def _parse_assessor_config(assessor_config: Any) -> _ParsedAssessorConfig:
    raw = _raw_assessor_config(assessor_config)
    _validate_assessor_top_level_keys(raw)

    target_path = str(raw.get("_target_", ""))
    resolved_target = resolve_target(target_path, _ROBUSTNESS_PREFIX)
    constructor_plain = _subdict(raw.get("constructor"), label="constructor")
    call_plain = _subdict(raw.get("call"), label="call")
    raitap_plain = _subdict(raw.get("raitap"), label="raitap")
    assessor_name = resolved_target or target_path or "?"
    _validate_raitap_keys(raitap_plain, assessor_name=assessor_name)
    _warn_on_misplaced_raitap_call_keys(call_plain, assessor_name=assessor_name)
    _migrate_misplaced_raitap_call_keys(call_plain, raitap_plain)

    return _ParsedAssessorConfig(
        raw=raw,
        target_path=target_path,
        resolved_target=resolved_target,
        algorithm=raw.get("algorithm"),
        constructor=constructor_plain,
        call=call_plain,
        raitap=raitap_plain,
    )


def _resolve_call_data_sources(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Replace ``call:`` values matching ``{source, n_samples}`` with loaded tensors."""
    resolved: dict[str, Any] = {}
    for key, value in call_kwargs.items():
        if isinstance(value, dict) and set(value).issubset(_DATA_SOURCE_KEYS) and "source" in value:
            source = value["source"]
            n_samples = value.get("n_samples")
            if n_samples is not None and not isinstance(n_samples, int):
                raise TypeError(
                    f"call.{key}.n_samples must be an int, got {type(n_samples).__name__}."
                )
            raitap_log.info(
                "Resolving robustness call kwarg %r as data source %r (n_samples=%s)",
                key,
                source,
                n_samples,
            )
            resolved[key] = load_tensor_from_source(str(source), n_samples=n_samples)
        else:
            resolved[key] = value
    return resolved


def _instantiate_assessor_from_parsed(
    parsed: _ParsedAssessorConfig,
) -> tuple[AssessorAdapter, str]:
    instantiate_cfg: dict[str, Any] = {
        **parsed.constructor,
        "algorithm": parsed.algorithm,
        "_target_": parsed.resolved_target,
    }
    try:
        assessor = instantiate(instantiate_cfg)
    except Exception as error:
        raitap_log.exception("Assessor instantiation failed for target %r", parsed.target_path)
        raise ValueError(
            f"Could not instantiate assessor {parsed.target_path!r}.\n"
            "Check that _target_ points to a valid AssessorAdapter implementation "
            "(e.g. EmpiricalAttackAssessor or FormalVerificationAssessor subclass)."
        ) from error

    if not isinstance(assessor, AssessorAdapter):
        raise ValueError(
            f"Instantiated assessor {parsed.target_path!r} does not implement AssessorAdapter. "
            "Configured assessors must have callable assess() and check_backend_compat() methods, "
            "and a ``method_kind`` attribute."
        )

    return cast("AssessorAdapter", assessor), parsed.resolved_target


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

            merged_kwargs = _resolve_call_data_sources({**call_from_config, **kwargs})
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
    return _instantiate_assessor_from_parsed(parsed)


def create_robustness_visualisers(
    assessor_config: Any,
) -> list[ConfiguredRobustnessVisualiser]:
    raw = _raw_assessor_config(assessor_config)
    out: list[ConfiguredRobustnessVisualiser] = []

    for visualiser_config in raw.get("visualisers", []):
        entry = _visualiser_entry_to_dict(visualiser_config)
        visualiser_target = str(entry.get("_target_", ""))
        _validate_visualiser_entry_keys(entry, target_hint=visualiser_target or "?")

        constructor_plain = _subdict(
            entry.get("constructor"),
            label=f"visualiser constructor ({visualiser_target})",
        )
        call_plain = _subdict(
            entry.get("call"),
            label=f"visualiser call ({visualiser_target})",
        )
        resolved_target = resolve_target(visualiser_target, _VISUALISER_PREFIX)

        instantiate_cfg: dict[str, Any] = {
            **constructor_plain,
            "_target_": resolved_target,
        }
        try:
            visualiser = instantiate(instantiate_cfg)
        except Exception as error:
            raitap_log.exception("Visualiser instantiation failed for target %r", visualiser_target)
            raise ValueError(f"Could not instantiate visualiser {visualiser_target!r}.") from error

        out.append(ConfiguredRobustnessVisualiser(visualiser=visualiser, call_kwargs=call_plain))

    return out
