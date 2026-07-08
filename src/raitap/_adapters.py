"""Single point of truth for adapter registration.

Every concrete adapter (explainer, assessor, metric computer, reporter,
tracker, visualiser) is registered via its family decorator (e.g.
``@adapters.transparency``, ``@adapters.robustness``,
``@visualisers.transparency``). The decorator delegates to
:func:`_register_core` which:

* computes the adapter class's FQN and records it in :data:`_TARGET_FQN`
  (``group -> registry_name -> FQN``) — the only trusted place a class FQN is
  looked up from a ``use:`` config key (see :mod:`raitap.configs.registry_resolve`)
* builds a ``use:``-node dataclass (schema fields plus a ``use: <registry_name>``
  field, no ``_target_``) and registers it with Hydra's ``ConfigStore`` (when
  the family owns a top-level config group such as ``transparency`` /
  ``robustness`` / ``metrics``)
* exposes the builder under ``raitap.<module>.<registry_name>`` (lazy
  ``__getattr__`` on each module looks it up in :data:`_BUILDERS`)
* records the ``extra`` dependency for :mod:`raitap.deps.inference`

:class:`AdapterMixin` is now a pure instance-helper mixin
(:meth:`_lazy_import`, :meth:`_rethrow`) — registration mechanics live in the
decorators.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import pkgutil
import re
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    Required,
    TypedDict,
    TypeVar,
    Unpack,
)

from hydra_zen import ZenStore

from raitap.types import TaskKind

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import ModuleType

    from raitap.models.base_backend import ModelBackend
    from raitap.types import Capability

CtxT = TypeVar("CtxT", contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)


class Invoker(Protocol[CtxT, ResultT]):
    """Builds and drives one library algorithm, returning its result.

    A registry entry may carry an ``invoker`` to override its adapter's default
    construct-and-call path (#266). The single argument is a per-family context
    dataclass; the return is that family's result tensor.
    """

    def __call__(self, ctx: CtxT, /) -> ResultT: ...


# Our own ``overwrite_ok=True`` store so re-importing a module — or pytest
# collecting the same class twice via slightly different paths — doesn't
# error on duplicate (group, name) entries.
store = ZenStore(overwrite_ok=True)

# group -> name -> hydra-zen-generated dataclass builder
_BUILDERS: dict[str, dict[str, type]] = {}
# group -> registry_name -> adapter class FQN. The sole trusted seam a
# ``use: <registry_name>`` config key is resolved against
# (:func:`raitap.configs.registry_resolve.resolve_target_fqn`). Group
# ``"_unscoped"`` holds visualisers, which have no Hydra config group.
_TARGET_FQN: dict[str, dict[str, str]] = {}
# adapter class name -> uv extra (consumed by raitap.deps.inference)
ADAPTER_EXTRAS: dict[str, str] = {}
# group -> set of wrapped third-party library names; used by
# :mod:`raitap.utils.diagnostics` to mark a "via <lib>" chip on log messages.
THIRD_PARTY_LIBS: dict[str, set[str]] = {}


@dataclass(frozen=True, slots=True)
class FamilyConfig:
    """Per-family registration constants. One instance per top-level RAITAP family
    (transparency, robustness, metrics, reporting, tracking). Owned by the family
    decorator, not the adapter site. Family-specific required adapter metadata
    (e.g. ``algorithm_registry``) lives as typed kwargs on the family decorator
    itself — pyright errors at the decoration site if missing."""

    group: str
    schema: type
    # ``"nested"`` → hydra package=``"<group>.<name>"``; the schema field is a
    # ``dict[str, Config]`` so multiple named entries can coexist. ``"flat"`` →
    # package=``"<group>"``; the schema field is a single config, names compete.
    package_style: Literal["nested", "flat"]


class AdapterDecoratorOptions(TypedDict, total=False):
    """Public: cross-family registration options every adapter decorator
    accepts. Forwarded into each family decorator via
    ``**common: Unpack[AdapterDecoratorOptions]``. Family-specific required
    kwargs (e.g. ``algorithm_registry``) are declared on the individual
    decorators. Plugin authors may import this type to re-type forwarded kwargs;
    the decorator signatures are the public contract regardless."""

    registry_name: Required[str]
    extra: str
    library: str
    # Raw regex strings → friendly messages. Compiled at registration by
    # ``_register_core`` (mirrors ``suppress_warnings``, which also takes raw
    # strings). Pass ``r"..."``; add inline flags like ``(?i)`` if needed.
    error_patterns: Mapping[str, str]
    suppress_warnings: Sequence[tuple[str, type[Warning], str | None]]
    schema: type


class AdapterMixin:
    """Instance helpers shared across every adapter family.

    Concrete adapters never inherit ``AdapterMixin`` directly — they extend
    their family base class (e.g. ``AttributionOnlyExplainer``,
    ``EmpiricalAttackAssessor``) which mixes in ``AdapterMixin``. Registration
    is done by the family decorator (e.g. ``@adapters.transparency``),
    not by inheritance.
    """

    registry_name: str | None = None
    extra: str | None = None
    # Wrapped third-party library (pip name). Set by ``_register_core``;
    # drives :meth:`_lazy_import` and :meth:`_rethrow`.
    library: str | None = None
    # Hydra config group ("transparency" / "robustness" / ...). Set by
    # ``_register_core`` and read by :meth:`_rethrow` to scope error chips.
    _adapter_group: str | None = None
    # Compiled regex → friendly-message map applied automatically by
    # :meth:`_rethrow`. Stored pre-compiled (``_register_core`` compiles the
    # raw-string ``error_patterns`` decorator kwarg).
    error_patterns: Mapping[re.Pattern[str], str] = {}
    # Task families this adapter accepts. Default classification so legacy
    # adapters stay correct without explicit declaration. Issue #146.
    supported_tasks: ClassVar[frozenset[TaskKind]] = frozenset({TaskKind.classification})

    def _lazy_import(self, submodule: str | None = None) -> ModuleType:
        """Import the wrapped third-party library lazily.

        Raises a clear, install-hint-bearing :class:`ImportError` when the
        library isn't available, instead of the bare ``ModuleNotFoundError``
        users would otherwise see deep in the pipeline. Pass ``submodule`` to
        load a specific subpackage (e.g. ``"attr"`` for ``captum.attr``).
        """
        cls = type(self)
        if not cls.library:
            raise RuntimeError(f"{cls.__name__} has no ``library`` declared")
        target = f"{cls.library}.{submodule}" if submodule else cls.library
        try:
            return importlib.import_module(target)
        except ModuleNotFoundError as exc:
            install_hint = f" (install with `uv sync --extra {cls.extra}`)" if cls.extra else ""
            raise ImportError(
                f"{cls.__name__} requires the {cls.library!r} package{install_hint}."
            ) from exc

    @contextmanager
    def _rethrow(self, *, base_exc: type[BaseException] = Exception) -> Iterator[None]:
        """Wrap a third-party call so curated error patterns get rewritten.

        Equivalent to ``rethrow(module=Module(<group>), third_party_lib=<library>,
        message_map=<error_patterns>)`` but pulls all three from the adapter's
        own class declaration (set by the family decorator at registration time).
        """
        from raitap.utils.diagnostics import Module
        from raitap.utils.errors import rethrow

        cls = type(self)
        with rethrow(
            module=Module(cls._adapter_group) if cls._adapter_group else Module.utils,
            third_party_lib=cls.library,
            message_map=cls.error_patterns or {},
            base_exc=base_exc,
        ):
            yield

    def required_capabilities(self) -> frozenset[Capability]:
        """Capabilities the configured algorithm needs.

        Reads the per-algorithm ``requires`` from this adapter's
        ``algorithm_registry``. Defaults to ``frozenset()`` (model-agnostic, runs
        on any backend) for adapters without a registry or matching algorithm.
        """
        registry = getattr(type(self), "algorithm_registry", None)
        if not isinstance(registry, Mapping):
            return frozenset()
        hints = registry.get(getattr(self, "algorithm", ""))
        return getattr(hints, "requires", frozenset())

    def check_backend_compat(self, backend: ModelBackend | None) -> None:
        """Default backend gate: every required capability must be provided.

        An absent backend provides nothing (empty capability set).

        Override only for a non-capability contract (e.g. ``MarabouAssessor`` uses
        the hook for per-call setup; ``AutoLiRPAAssessor`` adds an XPU warning).
        """
        from raitap.utils.errors import BackendIncompatibilityError

        provided = backend.provides if backend is not None else frozenset()
        missing = self.required_capabilities() - provided
        if missing:
            raise BackendIncompatibilityError(
                adapter=type(self).__name__,
                backend=type(backend).__name__,
                missing=sorted(str(c) for c in missing),
            )


def _class_fqn(cls: type) -> str:
    """Fully-qualified ``module.ClassName`` for ``cls``.

    CI's ``pythonpath = ["src"]`` plus ``src/__init__.py`` makes ``src`` an
    importable package too, so the same class can carry ``__module__ ==
    "src.raitap.…"`` when discovered by pytest before any ``raitap.*`` import
    canonicalises it. Strip the prefix so downstream ``instantiate()`` resolves
    the same module identity ``isinstance`` checks see.
    """
    module = cls.__module__
    if module.startswith("src."):
        module = module[len("src.") :]
    return f"{module}.{cls.__name__}"


def _use_node(schema: type, registry_name: str, cls_name: str) -> type:
    """Build a config-layer dataclass: ``schema`` fields plus a ``use:`` field
    defaulting to ``registry_name``. Never carries ``_target_`` — the real
    class FQN lives only in :data:`_TARGET_FQN`, resolved by
    :mod:`raitap.configs.registry_resolve` at instantiate-time.
    """
    return dataclasses.make_dataclass(
        f"_{cls_name}UseConf",
        [("use", str, dataclasses.field(default=registry_name))],
        bases=(schema,),
    )


@dataclass
class _VisualiserUseBase:
    """Minimal ``use:``-node schema for visualisers (family=None).

    Visualisers previously got a hydra-zen ``builds(cls, populate_full_signature=True,
    zen_meta={"call": {}, "raitap": {}})`` builder; this is the equivalent
    ``use:``-based shape: a bare ``use`` selector plus the three generic
    pass-through blocks (``constructor`` / ``call`` / ``raitap``).
    """

    use: str = ""
    constructor: dict[str, Any] = dataclasses.field(default_factory=dict)
    call: dict[str, Any] = dataclasses.field(default_factory=dict)
    raitap: dict[str, Any] = dataclasses.field(default_factory=dict)


_VISUALISER_USE_SCHEMA = _VisualiserUseBase


def _register_core(
    cls: type,
    *,
    family: FamilyConfig | None,
    **common: Unpack[AdapterDecoratorOptions],
) -> type:
    """Cross-family registration mechanics. Returns ``cls`` unchanged.

    Sets identity attrs on ``cls``, installs warning filters, validates
    family-required class-body attributes (e.g. ``algorithm_registry`` when
    ``family.has_algorithm_registry``), records the class FQN in
    :data:`_TARGET_FQN`, builds a ``use:``-node dataclass (:func:`_use_node`)
    for schema-backed families or the generic visualiser shape
    (:data:`_VISUALISER_USE_SCHEMA`), and registers under
    ``_BUILDERS[family.group][registry_name]`` (or ``_BUILDERS["_unscoped"]``
    when ``family is None``).
    """
    from raitap.utils.log import raitap_log

    registry_name = common["registry_name"]
    # ``extra`` defaults to ``registry_name`` for schema-backed adapters (family
    # is set) — covers 8/10 of the in-tree adapters; metrics overrides to share
    # ``extra="metrics"`` across two metric adapters. Visualisers (family=None)
    # don't get an auto-extra — they ship with their parent adapter's extra and
    # have no standalone uv extra of their own.
    extra = common.get("extra")
    if extra is None and family is not None:
        extra = registry_name
    library = common.get("library")
    error_patterns = common.get("error_patterns")
    suppress_warnings = common.get("suppress_warnings")
    schema_override = common.get("schema")

    cls.registry_name = registry_name
    if extra is not None:
        cls.extra = extra
    if library is not None:
        cls.library = library
    if error_patterns is not None:
        compiled: dict[re.Pattern[str], str] = {}
        for pattern, message in error_patterns.items():
            try:
                compiled[re.compile(pattern)] = message
            except re.error as exc:
                raise ValueError(
                    f"{cls.__name__} (registry_name={registry_name!r}) declares an invalid "
                    f"error_patterns regex {pattern!r}: {exc}"
                ) from exc
        cls.error_patterns = compiled
    if suppress_warnings:
        for pattern, category, module in suppress_warnings:
            raitap_log.suppress(message=pattern, category=category, module=module or "")

    try:
        if family is not None:
            cls._adapter_group = family.group
            fqn = _class_fqn(cls)
            _TARGET_FQN.setdefault(family.group, {})[registry_name] = fqn
            schema = schema_override or family.schema
            builder = _use_node(schema, registry_name, cls.__name__)
            # Hydra groups use ``/`` for nesting; OmegaConf packages use ``.``.
            # A nested group like ``data/labels`` must target package
            # ``data.labels`` so the composed node lands at ``cfg.data.labels``.
            package_base = family.group.replace("/", ".")
            package = (
                f"{package_base}.{registry_name}"
                if family.package_style == "nested"
                else package_base
            )
            # ``to_config=lambda x: x`` stores ``builder`` verbatim. ZenStore's
            # default ``to_config`` would otherwise run it through
            # ``hydra_zen.builds()`` again (dataclass *types* get
            # ``populate_full_signature=True, builds_bases=(target,)``),
            # which stamps a fresh ``_target_`` pointing at ``builder`` itself
            # — reopening exactly the arbitrary-``_target_`` surface this
            # rename closes.
            store(
                builder,
                group=family.group,
                name=registry_name,
                package=package,
                to_config=lambda x: x,
            )
            _BUILDERS.setdefault(family.group, {})[registry_name] = builder
        else:
            fqn = _class_fqn(cls)
            _TARGET_FQN.setdefault("_unscoped", {})[registry_name] = fqn
            builder = _use_node(_VISUALISER_USE_SCHEMA, registry_name, cls.__name__)
            _BUILDERS.setdefault("_unscoped", {})[registry_name] = builder
    except (ModuleNotFoundError, TypeError):
        # Test fixtures defining inline classes without an importable qualname
        # — hydra-zen rejects them and we silently skip.
        return cls

    if extra:
        ADAPTER_EXTRAS[cls.__name__] = extra
    if library and family is not None:
        THIRD_PARTY_LIBS.setdefault(family.group, set()).add(library)
    return cls


def discover(package_path: list[str], package_name: str) -> None:
    """Import every submodule under ``package_name`` so each module's
    ``@register_*_adapter`` decorator fires for every adapter class declared
    in the tree.

    Call from each module's ``__init__.py``::

        from raitap._adapters import discover
        discover(__path__, __name__)
    """
    for _finder, name, _ispkg in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
        if "tests" in name.split("."):
            continue
        importlib.import_module(name)


def _plugin_raitap_specifier(distribution_name: str) -> str | None:
    """Return the plugin distribution's declared ``raitap`` version specifier
    (from its pip ``Requires-Dist``), or ``None`` if it declares no raitap pin."""
    from packaging.requirements import Requirement

    reqs = importlib_metadata.requires(distribution_name) or []
    for raw in reqs:
        req = Requirement(raw)
        if req.name == "raitap":
            return str(req.specifier)
    return None


def _plugin_version_ok(distribution_name: str) -> tuple[bool, str]:
    """``(ok, message)``. Verify the running raitap version satisfies the
    plugin's declared raitap specifier. A plugin with no raitap pin is treated
    as malformed (not ok)."""
    from packaging.specifiers import SpecifierSet

    from raitap.__about__ import __version__

    spec = _plugin_raitap_specifier(distribution_name)
    if spec is None:
        return False, f"{distribution_name!r} declares no 'raitap' dependency pin"
    if __version__ not in SpecifierSet(spec, prereleases=True):
        return False, f"{distribution_name!r} requires raitap{spec}; running {__version__}"
    return True, ""


def discover_third_party_adapters() -> None:
    """Import every module under the ``raitap.adapters`` entry-point group so its
    decorators fire, populating :data:`_BUILDERS` like in-tree adapters.

    Default-allow; set ``RAITAP_DISABLE_PLUGINS=1`` to skip entirely. Each plugin
    is version-checked against its pip ``raitap`` pin and isolated: one failure
    is logged and skipped, never fatal.
    """
    if os.environ.get("RAITAP_DISABLE_PLUGINS"):
        return
    from raitap.utils.diagnostics import Module
    from raitap.utils.log import raitap_log

    for ep in importlib_metadata.entry_points(group="raitap.adapters"):
        dist_name = ep.dist.name if ep.dist is not None else ep.name
        try:
            ok, why = _plugin_version_ok(dist_name)
            if not ok:
                raitap_log.warn(f"Skipping plugin {ep.name!r}: {why}", module=Module.deps)
                continue
            ep.load()  # decorators fire as import side-effect
        except Exception as exc:
            raitap_log.warn(
                f"Plugin {ep.name!r} ({ep.value}) failed to load: {exc}",
                module=Module.deps,
            )


def lookup(group: str, name: str) -> Any:
    """Resolve a builder by (group, name). Used by module ``__getattr__``.

    Returns ``Any`` so callers can immediately invoke the result
    (``captum(algorithm="...")``) without pyright complaining that ``type``
    isn't callable in the user-facing sense — the builders are dataclass
    constructors but pyright can't infer that through the dict lookup.
    """
    try:
        return _BUILDERS[group][name]
    except KeyError:
        # Also check the unscoped pool (visualisers etc) when the caller
        # passed the family group — visualisers don't live under a Hydra
        # group, so look them up by name alone too.
        try:
            return _BUILDERS["_unscoped"][name]
        except KeyError:
            raise AttributeError(name) from None
