"""Generate a JSON Schema from the live adapter registry (refs #301).

:func:`build_config_schema` walks :data:`raitap._adapters._BUILDERS` — the
same ``group -> registry_name -> use-node dataclass`` map each family
decorator populates at import time — and emits a JSON-Schema dict whose
per-group ``use`` property is an ``enum`` of that group's valid ``use:``
selector values. A YAML editor pointed at the generated schema can then
autocomplete ``use:`` for ``transparency``, ``robustness``, ``metrics``,
``reporting``, ``tracking``, ``data.labels``, ``data.inputs`` and
visualiser entries.

Nested-style families (``transparency``, ``robustness``) allow multiple
named entries per group (``cfg.transparency.<name>.use``), so their group
schema wraps the ``use`` enum in ``additionalProperties``. Flat-style
families (``reporting``, ``metrics``, ``tracking``, ``data/labels``,
``data/inputs``) hold a single config per group (``cfg.<group>.use``).
Visualisers have no Hydra config group of their own and are registered
under ``_BUILDERS["_unscoped"]``; they surface here as a top-level
``visualiser`` property.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

# Families whose Hydra package style is "nested" (:class:`raitap._adapters.
# FamilyConfig`, ``package_style="nested"``) — multiple named entries share
# the group. Every other group (including ``data/labels`` / ``data/inputs``)
# is "flat": one config per group.
_NESTED_GROUPS = frozenset({"transparency", "robustness"})


def _use_enum_schema(registry_names: Iterable[str]) -> dict[str, Any]:
    return {"type": "string", "enum": sorted(registry_names)}


def _group_schema(registry_names: Iterable[str], *, nested: bool) -> dict[str, Any]:
    use_schema = _use_enum_schema(registry_names)
    if nested:
        return {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {"use": use_schema},
            },
        }
    return {"type": "object", "properties": {"use": use_schema}}


def _set_nested_property(
    properties: dict[str, Any], path: list[str], value: dict[str, Any]
) -> None:
    """Place ``value`` at ``path`` in ``properties``, splitting on Hydra's
    ``/``-nested group names (e.g. ``data/labels`` -> ``properties.data.
    properties.labels``, matching the composed OmegaConf ``cfg.data.labels``
    package)."""
    head, *rest = path
    if not rest:
        properties[head] = value
        return
    node = properties.setdefault(head, {"type": "object", "properties": {}})
    _set_nested_property(node["properties"], rest, value)


def build_config_schema() -> dict[str, Any]:
    """Build a JSON Schema whose ``use`` fields enumerate the live adapter
    registry.

    Triggers full family discovery (:func:`raitap.configs.register_configs`,
    idempotent) before reading :data:`raitap._adapters._BUILDERS`, so the
    schema reflects every in-tree and plugin adapter regardless of what the
    caller already imported.
    """
    from raitap._adapters import _BUILDERS
    from raitap.configs import register_configs

    register_configs()

    properties: dict[str, Any] = {}
    for group, entries in _BUILDERS.items():
        registry_names = entries.keys()
        if group == "_unscoped":
            properties["visualiser"] = _group_schema(registry_names, nested=False)
            continue
        schema = _group_schema(registry_names, nested=group in _NESTED_GROUPS)
        _set_nested_property(properties, group.split("/"), schema)

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": properties,
    }
