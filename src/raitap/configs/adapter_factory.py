"""Shared YAML-config parser for raitap adapter modules.

Both :mod:`raitap.robustness` and :mod:`raitap.transparency` configure their
adapters with the same YAML shape::

    _target_: raitap.<module>.<Adapter>
    algorithm: <name>
    constructor: { ... }      # __init__ kwargs
    call: { ... }              # per-call library kwargs
    raitap: { ... }            # framework-owned options
    visualisers: [ ... ]       # optional list of visualiser entries

This module owns the mechanics of parsing, validating, and instantiating that
shape so the two domain factories do not duplicate ~250 lines of identical
plumbing. Per-domain logic (compat checks, run-method dispatch, result
wrapping) stays in the domain factories.

Design rules:

* :class:`AdapterSchema` is data-only — no callbacks, no methods. The schema
  describes the YAML shape; orchestration lives in the per-domain
  entry-point class (``Explanation`` / ``RobustnessAssessment``).
* :class:`ParsedAdapterConfig` is a frozen container; it never mutates after
  ``parse_adapter_config`` returns it.
* Functions accept the schema explicitly so callers can see the parametrisation.
"""

from __future__ import annotations

from collections.abc import (  # noqa: TC003 — runtime import: typing.get_type_hints() resolves these
    Callable,
    Mapping,
)
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

if TYPE_CHECKING:
    import torch

from raitap import raitap_log
from raitap.configs.utils import cfg_to_dict, resolve_target
from raitap.data.data import load_tensor_from_source
from raitap.data.preprocessing import (
    ResolvedPreprocessing,
    module_as_per_image_callable,
    resolve_preprocessing,
)

__all__ = [
    "AdapterSchema",
    "ParsedAdapterConfig",
    "instantiate_adapter",
    "instantiate_visualisers",
    "parse_adapter_config",
    "raw_config_dict",
    "resolve_call_data_sources",
    "resolve_per_image_transform",
]


_DATA_SOURCE_KEYS = frozenset({"source", "n_samples"})
_VISUALISER_ENTRY_KEYS = frozenset({"_target_", "constructor", "call"})


@dataclass(frozen=True)
class AdapterSchema:
    """Describes a domain's adapter YAML shape.

    Attributes:
        domain: lowercase domain key used in user-facing messages, e.g.
            ``"robustness"`` or ``"transparency"``.
        entity: lowercase noun naming the configured object,
            e.g. ``"assessor"`` or ``"explainer"``.
        subdict_namespace: title-cased label used in subdict-type errors,
            e.g. ``"Robustness"``.
        target_prefix: prefix prepended to bare ``_target_`` values
            (matching :func:`raitap.configs.resolve_target`).
        visualiser_prefix: prefix for visualiser ``_target_`` values. Often
            equals ``target_prefix`` (transparency) or differs (robustness
            uses a separate ``visualisers.`` namespace).
        top_level_keys: allowed keys directly under the adapter config block.
        raitap_keys: allowed keys under the ``raitap:`` sub-block.
        removed_raitap_keys: keys that were valid in older versions but now
            raise ``ValueError`` with the supplied migration message.
        top_level_error_hint: trailing sentence appended to the
            "Unknown … config keys" error so users see domain-specific guidance.
    """

    domain: str
    entity: str
    subdict_namespace: str
    target_prefix: str
    visualiser_prefix: str
    top_level_keys: frozenset[str]
    raitap_keys: frozenset[str]
    top_level_error_hint: str
    removed_raitap_keys: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedAdapterConfig:
    """Result of :func:`parse_adapter_config` — fully normalised + validated."""

    raw: dict[str, Any]
    target_path: str
    resolved_target: str
    algorithm: Any
    constructor: dict[str, Any]
    call: dict[str, Any]
    raitap: dict[str, Any]


# ---------------------------------------------------------------------------
# Raw config helpers
# ---------------------------------------------------------------------------


def raw_config_dict(adapter_config: Any) -> dict[str, Any]:
    """Convert an adapter ``DictConfig`` (or plain dict) to a plain ``dict``."""
    return cfg_to_dict(adapter_config)


def _subdict(value: Any, *, label: str, schema: AdapterSchema) -> dict[str, Any]:
    """Normalise a ``constructor`` / ``call`` / ``raitap`` block to a ``dict``."""
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
                f"{schema.subdict_namespace} {label!r} must be a mapping, "
                f"got {type(container).__name__}."
            )
        return cast("dict[str, Any]", dict(container))
    raise TypeError(
        f"{schema.subdict_namespace} {label!r} must be a dict or DictConfig, "
        f"got {type(value).__name__}."
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_top_level_keys(raw: dict[str, Any], schema: AdapterSchema) -> None:
    unknown = set(raw) - schema.top_level_keys
    if not unknown:
        return
    sorted_unknown = ", ".join(sorted(unknown))
    raise ValueError(
        f"Unknown {schema.domain} {schema.entity} config keys: {sorted_unknown}. "
        f"{schema.top_level_error_hint}"
    )


def _validate_visualiser_entry_keys(entry: dict[str, Any], *, target_hint: str) -> None:
    unknown = set(entry) - _VISUALISER_ENTRY_KEYS
    if unknown:
        sorted_unknown = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown keys in visualiser config {target_hint!r}: {sorted_unknown}. "
            "Use 'constructor' for __init__ kwargs and 'call' for visualise() kwargs."
        )


def _validate_raitap_keys(
    raitap_cfg: dict[str, Any], *, entity_name: str, schema: AdapterSchema
) -> None:
    for key, message in schema.removed_raitap_keys.items():
        if key in raitap_cfg:
            raise ValueError(message)

    unknown = set(raitap_cfg) - schema.raitap_keys
    if not unknown:
        return

    sorted_unknown = ", ".join(sorted(unknown))
    sorted_valid = ", ".join(sorted(schema.raitap_keys))
    raitap_log.warn(
        f"Unknown {schema.domain}.raitap keys for {schema.entity} {entity_name!r}: "
        f"{sorted_unknown}. Supported RAITAP keys: {sorted_valid}.",
    )


def _warn_on_misplaced_raitap_call_keys(
    call_cfg: dict[str, Any], *, entity_name: str, schema: AdapterSchema
) -> None:
    misplaced = sorted(set(call_cfg).intersection(schema.raitap_keys))
    if not misplaced:
        return
    keys = ", ".join(misplaced)
    raitap_log.warn(
        f"{schema.entity.capitalize()} {entity_name!r} has RAITAP-owned keys "
        f"under 'call:': {keys}. These keys belong under 'raitap:' while "
        "'call:' is intended for library kwargs only.",
    )


def _migrate_misplaced_raitap_call_keys(
    call_cfg: dict[str, Any], raitap_cfg: dict[str, Any], schema: AdapterSchema
) -> None:
    misplaced = sorted(set(call_cfg).intersection(schema.raitap_keys))
    for key in misplaced:
        value = call_cfg.pop(key)
        raitap_cfg.setdefault(key, value)


# ---------------------------------------------------------------------------
# Top-level parsing
# ---------------------------------------------------------------------------


def parse_adapter_config(adapter_config: Any, schema: AdapterSchema) -> ParsedAdapterConfig:
    """Parse + validate an adapter config block. Returns a frozen container."""
    raw = raw_config_dict(adapter_config)
    _validate_top_level_keys(raw, schema)

    target_path = str(raw.get("_target_", ""))
    resolved_target = resolve_target(target_path, schema.target_prefix)
    constructor_plain = _subdict(raw.get("constructor"), label="constructor", schema=schema)
    call_plain = _subdict(raw.get("call"), label="call", schema=schema)
    raitap_plain = _subdict(raw.get("raitap"), label="raitap", schema=schema)
    entity_name = resolved_target or target_path or "?"
    _validate_raitap_keys(raitap_plain, entity_name=entity_name, schema=schema)
    _warn_on_misplaced_raitap_call_keys(call_plain, entity_name=entity_name, schema=schema)
    _migrate_misplaced_raitap_call_keys(call_plain, raitap_plain, schema)

    return ParsedAdapterConfig(
        raw=raw,
        target_path=target_path,
        resolved_target=resolved_target,
        algorithm=raw.get("algorithm"),
        constructor=constructor_plain,
        call=call_plain,
        raitap=raitap_plain,
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


A = TypeVar("A")
W = TypeVar("W")


def instantiate_adapter(
    parsed: ParsedAdapterConfig,
    *,
    protocol: type[A],
    schema: AdapterSchema,
    instantiate_error_hint: str,
    type_error_hint: str,
    instantiate_fn: Callable[[dict[str, Any]], Any] | None = None,
) -> tuple[A, str]:
    """Instantiate the adapter described by ``parsed`` and verify its protocol.

    ``protocol`` is the runtime-checkable Protocol or ABC the adapter must
    satisfy (e.g. ``ExplainerAdapter``, ``AssessorAdapter``).

    ``instantiate_fn`` defaults to :func:`hydra.utils.instantiate` but can be
    overridden so per-domain factories can wire in test doubles or monkey-
    patchable module-level bindings.
    """
    instantiate_cfg: dict[str, Any] = {
        **parsed.constructor,
        "algorithm": parsed.algorithm,
        "_target_": parsed.resolved_target,
    }

    fn = instantiate_fn if instantiate_fn is not None else instantiate
    try:
        adapter = fn(instantiate_cfg)
    except MissingMandatoryValue as error:
        # Surface "you forgot to set <key>" as the headline message instead of
        # burying it under the generic "Could not instantiate <FQN>" wording.
        missing_key = getattr(error, "full_key", None) or "a required field"
        raise ValueError(
            f"Your {schema.entity} config is missing the required `{missing_key}` "
            f"field for `{parsed.target_path}`. Set it in the YAML entry — e.g. "
            f"`{missing_key}: <value>` — or pass it as a keyword argument when "
            f"building the config in Python."
        ) from error
    except Exception as error:
        raitap_log.exception(
            "%s instantiation failed for target %r",
            schema.entity.capitalize(),
            parsed.target_path,
        )
        raise ValueError(
            f"Could not instantiate {schema.entity} {parsed.target_path!r}.\n"
            f"{instantiate_error_hint}"
        ) from error

    if not isinstance(adapter, protocol):
        raise ValueError(
            f"Instantiated {schema.entity} {parsed.target_path!r} does not implement "
            f"{protocol.__name__}. {type_error_hint}"
        )

    return cast("A", adapter), parsed.resolved_target


def instantiate_visualisers(
    adapter_config: Any,
    *,
    schema: AdapterSchema,
    wrap: Callable[[Any, dict[str, Any]], W],
    instantiate_fn: Callable[[dict[str, Any]], Any] | None = None,
) -> list[W]:
    """Instantiate the ``visualisers:`` list and wrap each with ``wrap(viz, call)``.

    ``wrap`` produces the per-domain ``ConfiguredVisualiser`` flavour
    (``ConfiguredVisualiser`` for transparency, ``ConfiguredRobustnessVisualiser``
    for robustness). The shared module is agnostic to the wrapper type.
    """
    raw = raw_config_dict(adapter_config)
    out: list[W] = []

    for visualiser_config in raw.get("visualisers", []):
        entry = _visualiser_entry_to_dict(visualiser_config)
        raw_target = str(entry.get("_target_", ""))

        # Three shapes accepted:
        #   1. Hydra-zen builder with ``zen_meta`` (call/raitap as metadata):
        #      ``_target_`` is the zen-processing wrapper; the real class is
        #      under ``_zen_target``. Let hydra-zen's ``instantiate`` handle
        #      everything (target resolution, kwarg filtering, etc.).
        #   2. YAML dict: ``{_target_, constructor: {...}, call: {...}}``.
        #   3. Flat hydra-zen builder (no zen_meta): ``_target_`` is the real
        #      class FQN, every other key is a constructor kwarg.
        if raw_target == "hydra_zen.funcs.zen_processing":
            visualiser_target = str(entry.get("_zen_target", ""))
            call_plain = _subdict(
                entry.get("call"),
                label=f"visualiser call ({visualiser_target})",
                schema=schema,
            )
            instantiate_cfg: dict[str, Any] = entry
        else:
            visualiser_target = raw_target
            if "constructor" in entry or set(entry).issubset(_VISUALISER_ENTRY_KEYS):
                _validate_visualiser_entry_keys(entry, target_hint=visualiser_target or "?")
                constructor_source: Any = entry.get("constructor")
            else:
                constructor_source = {
                    k: v for k, v in entry.items() if k not in {"_target_", "call", "raitap"}
                }
            constructor_plain = _subdict(
                constructor_source,
                label=f"visualiser constructor ({visualiser_target})",
                schema=schema,
            )
            call_plain = _subdict(
                entry.get("call"),
                label=f"visualiser call ({visualiser_target})",
                schema=schema,
            )
            resolved_target = resolve_target(visualiser_target, schema.visualiser_prefix)
            instantiate_cfg = {**constructor_plain, "_target_": resolved_target}

        fn = instantiate_fn if instantiate_fn is not None else instantiate
        try:
            visualiser = fn(instantiate_cfg)
        except Exception as error:
            raitap_log.exception("Visualiser instantiation failed for target %r", visualiser_target)
            raise ValueError(f"Could not instantiate visualiser {visualiser_target!r}.") from error

        out.append(wrap(visualiser, call_plain))

    return out


# ---------------------------------------------------------------------------
# Call-kwarg data-source resolution
# ---------------------------------------------------------------------------


def resolve_per_image_transform(
    config: Any,
    *,
    resolved_preprocessing: ResolvedPreprocessing | None = None,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Return the shape-half preprocessing callable for *config*, or ``None``.

    Shared helper for factory entry points that need to apply the same
    per-image transform to auxiliary call-data tensors that
    :class:`raitap.data.data.Data` applies to the primary input.

    The orchestrator path always supplies ``resolved_preprocessing`` so the
    run-level :class:`ResolvedPreprocessing` is reused without re-importing
    custom-file preprocessing. The fallback resolver branch is reachable only
    from direct/legacy factory or helper use (e.g. ``Explanation(config, ...)``
    constructed outside the pipeline) and from test mocks lacking ``model`` /
    ``data``; it returns ``None`` in those cases or when preprocessing is off.
    """
    if resolved_preprocessing is not None:
        return module_as_per_image_callable(resolved_preprocessing.data_module)

    model_cfg = getattr(config, "model", None)
    data_cfg = getattr(config, "data", None)
    if model_cfg is None or data_cfg is None:
        return None
    resolved = resolve_preprocessing(model_cfg, data_cfg)
    return module_as_per_image_callable(resolved.data_module)


def resolve_call_data_sources(
    call_kwargs: dict[str, Any],
    *,
    log_label: str = "call",
    per_image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    provenance_out: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replace ``call:`` values matching ``{source, n_samples}`` with loaded tensors.

    A value is treated as a data-source reference when it is a plain ``dict``
    whose keys are a subset of ``{"source", "n_samples"}`` and ``"source"`` is
    present.  Such values are replaced with the loaded tensor so the
    downstream adapter receives a ``torch.Tensor`` instead of a raw config dict.

    Example YAML under ``call:``::

        background_data:
          source: "imagenet_samples"
          n_samples: 50

    When ``per_image_transform`` is supplied (typically the pipeline's data
    preprocessing transform), it is applied per-image so that auxiliary
    tensors (SHAP background, baselines, …) share the same shape as the
    primary ``Data.tensor``.
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
            raitap_log.info(
                "Resolving %s kwarg %r as data source %r (n_samples=%s)",
                log_label,
                key,
                source,
                n_samples,
            )
            if provenance_out is not None:
                provenance_out[key] = {"source": str(source), "n_samples": n_samples}
            resolved[key] = load_tensor_from_source(
                str(source),
                n_samples=n_samples,
                per_image_transform=per_image_transform,
            )
        else:
            resolved[key] = value
    return resolved
