---
title: "Config API"
description: "Reference for RAITAP's public config helpers and top-level schema entry point."
---

The public config helpers live in `src/raitap/configs/__init__.py`. They provide the structured config root and a small set of utilities used by the CLI, factories, and external callers.

## Imports

```python
from raitap.configs import (
    AppConfig,
    cfg_to_dict,
    register_configs,
    resolve_run_dir,
    resolve_target,
    set_output_root,
)
```

## `AppConfig`

```python
@dataclass
class AppConfig:
    model: ModelConfig
    data: DataConfig
    transparency: dict[str, Any]
    metrics: MetricsConfig
    tracking: TrackingConfig
    reporting: ReportingConfig | None
    hardware: str = "gpu"
    experiment_name: str = "Experiment"
```

`AppConfig` is the structured root registered with Hydra's `ConfigStore`. The nested dataclasses are documented in `/docs/types`.

## `cfg_to_dict`

```python
def cfg_to_dict(cfg: Any) -> dict
```

Normalizes a Hydra `DictConfig`, dataclass instance, simple object, or plain mapping into a Python dict.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cfg` | `Any` | — | Config object to normalize. |

Return type: `dict`

## `register_configs`

```python
def register_configs() -> None
```

Registers `AppConfig` under the Hydra group `schema/config`. `raitap.run` calls this on import so CLI runs do not need to do it manually.

## `resolve_run_dir`

```python
def resolve_run_dir(
    config: AppConfig | None = None,
    *,
    output_root: str | Path | None = None,
    subdir: str | None = None,
) -> Path
```

Returns the active Hydra output directory when Hydra is running, or an explicit fallback path when it is not.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `AppConfig \| None` | `None` | Optional config whose `_output_root` can be used as a fallback. |
| `output_root` | `str \| Path \| None` | `None` | Explicit root when Hydra runtime metadata is unavailable. |
| `subdir` | `str \| None` | `None` | Optional child directory to append. |

Return type: `Path`

## `resolve_target`

```python
def resolve_target(target: str, prefix: str) -> str
```

Expands a short Hydra `_target_` value into a prefixed import path unless the string already contains a dot.

## `set_output_root`

```python
def set_output_root(config: Any, output_root: str | Path) -> None
```

Sets an `_output_root` attribute on a config object so non-Hydra code can still write artifacts to a predictable directory.

Example:

```python
from raitap.configs import set_output_root

set_output_root(cfg, "./outputs/manual")
```

These helpers are deliberately small. The larger config surface is defined by the schema dataclasses and the YAML groups under `src/raitap/configs/`.
