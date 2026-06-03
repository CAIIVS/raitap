"""Single point of truth for adapter registration.

Every concrete adapter (explainer, assessor, metric computer, reporter,
tracker, visualiser) is registered via its family decorator (e.g.
``@adapters.transparency``, ``@adapters.robustness``,
``@visualisers.transparency``). The decorator delegates to
:func:`_register_core` which:

* generates the hydra-zen builder (``builds(...)``)
* registers it with Hydra's ``ConfigStore`` (when the family owns a top-level
  config group such as ``transparency`` / ``robustness`` / ``metrics``)
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
import inspect
import os
import pkgutil
import re
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Required, TypedDict, Unpack

from hydra_zen import ZenStore, builds

from raitap.types import TaskKind

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import ModuleType

    from raitap.types import Capability

# Our own ``overwrite_ok=True`` store so re-importing a module — or pytest
# collecting the same class twice via slightly different paths — doesn't
# error on duplicate (group, name) entries.
store = ZenStore(overwrite_ok=True)

# group -> name -> hydra-zen-generated dataclass builder
_BUILDERS: dict[str, dict[str, type]] = {}
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


class _AllAlgorithmsSentinel:
    """Singleton type for the :data:`ALL` marker — pass
    ``onnx_compatible_algorithms=ALL`` to ``@adapters.transparency`` /
    ``@adapters.robustness`` to mark every algorithm in the adapter's
    ``algorithm_registry`` as ONNX-compatible without re-listing them."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "raitap.ALL"


ALL: Final[_AllAlgorithmsSentinel] = _AllAlgorithmsSentinel()


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

    def check_backend_compat(self, backend: object) -> None:
        """Default backend gate: every required capability must be provided.

        Override only for a non-capability contract (e.g. ``MarabouAssessor`` uses
        the hook for per-call setup; ``AutoLiRPAAssessor`` adds an XPU warning).
        """
        from raitap.utils.errors import BackendIncompatibilityError

        provided = getattr(backend, "provides", frozenset())
        missing = self.required_capabilities() - provided
        if missing:
            raise BackendIncompatibilityError(
                adapter=type(self).__name__,
                backend=type(backend).__name__,
                missing=sorted(str(c) for c in missing),
            )


def _build_schema_adapter(cls: type, schema: type) -> type:
    """Pick a hydra-zen builder shape based on whether ``cls.__init__`` can
    accept the schema's field kwargs.

    * ``**kwargs`` in init (Captum, torchattacks, …) → ``builds(cls, builds_bases=schema)``
      lifts every schema field onto the resulting dataclass and forwards them
      to the wrapped class at instantiate-time.
    * Narrow init (``MLFlowTracker(config: AppConfig)``, ``HTMLReporter(config)``)
      → schema-subclass with only ``_target_`` set; the wrapped class reads the
      remaining fields off the composed config blob itself.
    """
    sig = inspect.signature(cls.__init__)
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    init_params = {p for p in sig.parameters if p != "self"}
    schema_fields = {f.name for f in dataclasses.fields(schema) if f.name != "_target_"}
    accepts_schema = has_var_kw or schema_fields.issubset(init_params)

    if accepts_schema:
        return builds(cls, builds_bases=(schema,))

    # CI's ``pythonpath = ["src"]`` plus ``src/__init__.py`` makes ``src`` an
    # importable package too, so the same class can carry ``__module__ ==
    # "src.raitap.…"`` when discovered by pytest before any ``raitap.*`` import
    # canonicalises it. Strip the prefix so ``instantiate()`` resolves the
    # same module identity ``isinstance`` checks see.
    module = cls.__module__
    if module.startswith("src."):
        module = module[len("src.") :]
    fqn = f"{module}.{cls.__name__}"
    return dataclasses.make_dataclass(
        f"_{cls.__name__}Conf",
        [("_target_", str, dataclasses.field(default=fqn))],
        bases=(schema,),
    )


def _register_core(
    cls: type,
    *,
    family: FamilyConfig | None,
    **common: Unpack[AdapterDecoratorOptions],
) -> type:
    """Cross-family registration mechanics. Returns ``cls`` unchanged.

    Sets identity attrs on ``cls``, installs warning filters, validates
    family-required class-body attributes (e.g. ``algorithm_registry`` when
    ``family.has_algorithm_registry``), builds the hydra-zen builder
    (``_build_schema_adapter`` for schema-backed families, signature-based
    ``builds(...)`` for visualisers), and registers under
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
            builder = _build_schema_adapter(cls, schema_override or family.schema)
            package = (
                f"{family.group}.{registry_name}"
                if family.package_style == "nested"
                else family.group
            )
            store(builder, group=family.group, name=registry_name, package=package)
            _BUILDERS.setdefault(family.group, {})[registry_name] = builder
        else:
            builder = builds(
                cls,
                populate_full_signature=True,
                zen_meta={"call": {}, "raitap": {}},
            )
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
