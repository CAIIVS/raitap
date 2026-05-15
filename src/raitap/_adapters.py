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
from typing import Any

from hydra_zen import ZenStore, builds

# Our own ``overwrite_ok=True`` store so re-importing a module — or pytest
# collecting the same class twice via slightly different paths — doesn't
# error on duplicate (group, name) entries.
store = ZenStore(overwrite_ok=True)

# group -> name -> hydra-zen-generated dataclass builder
_BUILDERS: dict[str, dict[str, type]] = {}
# adapter class name -> uv extra (consumed by raitap.deps.inference)
ADAPTER_EXTRAS: dict[str, str] = {}

_CAMEL_TO_KEBAB = re.compile(r"(?<!^)(?=[A-Z])")


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
                # signature-based builder gives users typed kwargs without
                # needing a schema dataclass.
                builder = builds(cls, populate_full_signature=True)
                _BUILDERS.setdefault("_unscoped", {})[name] = builder
        except (ModuleNotFoundError, TypeError):
            # Test fixtures define classes inline (no importable path) which
            # hydra-zen rejects. We silently skip — those classes aren't going
            # to be looked up by ``raitap.<module>.<name>`` anyway.
            return

        if cls.extra:
            ADAPTER_EXTRAS[cls.__name__] = cls.extra


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

    fqn = f"{cls.__module__}.{cls.__name__}"
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


def lookup(group: str, name: str) -> type:
    """Resolve a builder by (group, name). Used by module ``__getattr__``."""
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
