"""Single point of truth for adapter registration.

Every concrete adapter (explainer, assessor, metric computer, reporter,
tracker, visualiser) inherits :class:`AdapterMixin` via its module's abstract
base class. ``__init_subclass__`` does the wiring:

* generates the hydra-zen builder (``builds(...)``)
* registers it with Hydra's ``ConfigStore`` (when the family owns a top-level
  config group such as ``transparency`` / ``robustness`` / ``metrics``)
* exposes the builder under ``raitap.<module>.<registry_name>`` (lazy
  ``__getattr__`` on each module looks it up in :data:`_BUILDERS`)
* records the ``extra`` dependency for :mod:`raitap.deps.inference`

The abstract base of each family declares ``group`` + ``schema`` via class
keyword arguments. Concrete adapters only declare ``registry_name`` (and
optionally ``extra``); nothing else needs editing to wire a new adapter in.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Required, TypedDict, Unpack

from hydra_zen import ZenStore, builds

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from types import ModuleType

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
    decorator, not the adapter site."""

    group: str
    schema: type
    package_style: Literal["nested", "flat"]
    strip_suffixes: tuple[str, ...]


class _CommonRegKwargs(TypedDict, total=False):
    """Cross-family registration kwargs. Forwarded into every family decorator
    via ``**common: Unpack[_CommonRegKwargs]`` so each decorator declares only
    its own family-specific required kwargs."""

    registry_name: Required[str]
    extra: str
    library: str
    error_patterns: "Mapping[re.Pattern[str], str]"
    suppress_warnings: "Sequence[tuple[str, type[Warning], str | None]]"


def _to_snake(name: str) -> str:
    """``CaptumExplainer`` -> ``captum_explainer``, ``HTMLReporter`` -> ``html_reporter``."""
    # Two passes so consecutive caps (``HTMLReporter``) collapse correctly.
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _default_registry_name(cls: type, *, strip_suffixes: tuple[str, ...] = ()) -> str:
    """Default registry name = snake-cased class name with adapter suffix stripped."""
    name = cls.__name__
    for suffix in strip_suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return _to_snake(name)


class AdapterMixin:
    """Inherit on every adapter family's abstract base.

    Class-kwarg API:

    * Abstract base declares the family contract once::

        class AttributionOnlyExplainer(
            BaseExplainer, AdapterMixin,
            abstract=True,
            group="transparency",
            schema=TransparencyConfig,
        ): ...

    * Concrete adapter only states its own identity::

        class SuperXAIExplainer(
            AttributionOnlyExplainer,
            registry_name="superxai",
            extra="superxai",
        ): ...

      Or omit ``registry_name`` to use the auto-snake-cased class name
      (``SuperXAIExplainer`` -> ``super_xai``).
    """

    _ADAPTER_GROUP: str | None = None
    _ADAPTER_SCHEMA: type | None = None
    _ADAPTER_STRIP_SUFFIXES: tuple[str, ...] = ()
    # ``"nested"`` (default) → package=``"<group>.<name>"``; the schema field is a
    # ``dict[str, Config]`` so multiple named entries can coexist. ``"flat"`` →
    # package=``"<group>"``; the schema field is a single config, names compete.
    _ADAPTER_PACKAGE_STYLE: str = "nested"
    registry_name: str | None = None
    extra: str | None = None
    # Wrapped third-party library (pip name). Drives ``self._lazy_import()``,
    # ``self._rethrow()``, and module-level :data:`THIRD_PARTY_LIBS`.
    library: str | None = None
    # Regex → friendly-message map applied automatically by ``self._rethrow()``.
    error_patterns: Mapping[re.Pattern[str], str] | None = None

    def __init_subclass__(
        cls,
        *,
        abstract: bool = False,
        group: str | None = None,
        schema: type | None = None,
        package_style: str | None = None,
        strip_suffixes: tuple[str, ...] | None = None,
        registry_name: str | None = None,
        extra: str | None = None,
        library: str | None = None,
        error_patterns: Mapping[re.Pattern[str], str] | None = None,
        suppress_warnings: tuple[tuple[str, type[Warning], str | None], ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)

        # Inherit / override family metadata from the abstract base.
        if group is not None:
            cls._ADAPTER_GROUP = group
        if schema is not None:
            cls._ADAPTER_SCHEMA = schema
        if package_style is not None:
            cls._ADAPTER_PACKAGE_STYLE = package_style
        if strip_suffixes is not None:
            cls._ADAPTER_STRIP_SUFFIXES = strip_suffixes
        if registry_name is not None:
            cls.registry_name = registry_name
        if extra is not None:
            cls.extra = extra
        if library is not None:
            cls.library = library
        if error_patterns is not None:
            cls.error_patterns = error_patterns

        # Silence noisy library warnings once at class load. Declared via
        # ``suppress_warnings=`` so each adapter file no longer needs a
        # module-level ``raitap_log.suppress(...)`` block.
        if suppress_warnings:
            from raitap.utils.log import raitap_log

            for pattern, category, module in suppress_warnings:
                raitap_log.suppress(message=pattern, category=category, module=module or "")

        if abstract or inspect.isabstract(cls):
            # ABCs and intermediates with unimplemented ``@abstractmethod``s
            # opt out automatically — only concrete leaves register.
            return

        name = cls.registry_name or _default_registry_name(
            cls, strip_suffixes=cls._ADAPTER_STRIP_SUFFIXES
        )

        try:
            if cls._ADAPTER_GROUP and cls._ADAPTER_SCHEMA:
                builder = _build_schema_adapter(cls, cls._ADAPTER_SCHEMA)
                if cls._ADAPTER_PACKAGE_STYLE == "nested":
                    package = f"{cls._ADAPTER_GROUP}.{name}"
                else:
                    package = cls._ADAPTER_GROUP
                store(
                    builder,
                    group=cls._ADAPTER_GROUP,
                    name=name,
                    package=package,
                )
                _BUILDERS.setdefault(cls._ADAPTER_GROUP, {})[name] = builder
            else:
                # Visualisers + anything without a top-level Hydra group. The
                # signature-based builder gives users typed kwargs (constructor
                # values) directly. ``zen_meta`` attaches ``call=`` / ``raitap=``
                # as metadata fields not forwarded to ``__init__`` — the adapter
                # factory peels them off and routes them to the render step.
                # ``image_pair(max_samples=4, call={"show_sample_names": True})``
                # becomes the canonical Python shape; no ``_target_`` strings
                # needed at any usage site.
                builder = builds(
                    cls,
                    populate_full_signature=True,
                    zen_meta={"call": {}, "raitap": {}},
                )
                _BUILDERS.setdefault("_unscoped", {})[name] = builder
        except (ModuleNotFoundError, TypeError):
            # Test fixtures define classes inline (no importable path) which
            # hydra-zen rejects. We silently skip — those classes aren't going
            # to be looked up by ``raitap.<module>.<name>`` anyway.
            return

        if cls.extra:
            ADAPTER_EXTRAS[cls.__name__] = cls.extra
        if cls.library and cls._ADAPTER_GROUP:
            THIRD_PARTY_LIBS.setdefault(cls._ADAPTER_GROUP, set()).add(cls.library)

    # ------------------------------------------------------------------
    # Instance helpers — concrete adapters use these instead of hand-rolling
    # the ``Module()`` / ``rethrow(module=..., third_party_lib=..., ...)``
    # boilerplate at every call site.
    # ------------------------------------------------------------------

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
        own class declaration.
        """
        from raitap.utils.diagnostics import Module
        from raitap.utils.errors import rethrow

        cls = type(self)
        with rethrow(
            module=Module(cls._ADAPTER_GROUP) if cls._ADAPTER_GROUP else Module.utils,
            third_party_lib=cls.library,
            message_map=cls.error_patterns or {},
            base_exc=base_exc,
        ):
            yield


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


def discover(package_path: list[str], package_name: str) -> None:
    """Import every submodule under ``package_name`` so ``__init_subclass__``
    fires for every adapter class declared anywhere in the tree.

    Call from each module's ``__init__.py``::

        from raitap._adapters import discover
        discover(__path__, __name__)
    """
    for _finder, name, _ispkg in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
        if "tests" in name.split("."):
            continue
        importlib.import_module(name)


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
