---
title: "Data API"
description: "Reference for dataset loading, metadata inference, and source-based loader helpers."
---

The data module is defined across `src/raitap/data/data.py` and `src/raitap/data/metadata.py`. Its public surface is designed around one object, `Data`, plus a few direct loader helpers.

## Imports

```python
from raitap.data import (
    Data,
    DataInputMetadata,
    infer_data_input_metadata,
    load_numpy_from_source,
    load_tensor_from_source,
)
```

## `Data`

Source: `src/raitap/data/data.py`

```python
class Data(Trackable):
    def __init__(self, cfg: AppConfig) -> None
```

`Data` loads the configured source into `self.tensor`, computes `self.sample_ids` for image sources, and optionally loads aligned labels into `self.labels`.

Constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cfg` | `AppConfig` | — | Run config containing `data.*` settings. |

Public methods:

```python
def describe(self) -> dict[str, Any]
def log(self, tracker: BaseTracker, **kwargs: Any) -> None
```

Example:

```python
data = Data(cfg)
print(data.describe())
```

## `load_tensor_from_source`

```python
def load_tensor_from_source(source: str, n_samples: int | None = None) -> torch.Tensor
```

Loads a tensor from a named sample set, local path, or URL, optionally subsampling rows.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | — | Demo sample name, URL, or local file/directory path. |
| `n_samples` | `int \| None` | `None` | Optional random subsample size. |

Return type: `torch.Tensor`

## `load_numpy_from_source`

```python
def load_numpy_from_source(source: str, n_samples: int | None = None) -> np.ndarray[Any, Any]
```

Equivalent to `load_tensor_from_source`, but prefers a NumPy-only path for file-based sources.

## `DataInputMetadata`

```python
@dataclass(frozen=True)
class DataInputMetadata:
    kind: str | None
    shape: tuple[int, ...] | None
    layout: str | None
    feature_names: list[str] | None = None
    metadata: dict[str, Any] | None = None
```

Returned by `infer_data_input_metadata()` and used by the transparency layer when building `InputSpec`.

## `infer_data_input_metadata`

```python
def infer_data_input_metadata(config: object, data: object) -> DataInputMetadata
```

Infers image or tabular metadata from explicit config, explicit `data.input_metadata`, or the source path and tensor shape.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `object` | — | Config-like object that may contain `data.input_metadata`. |
| `data` | `object` | — | Data-like object that may contain `source`, `tensor`, or explicit metadata. |

Return type: `DataInputMetadata`

## Common Pattern

Combine the module-level loader with a full `Data` object when you need both main inputs and an auxiliary background dataset:

```python
from raitap.data import Data, load_tensor_from_source

data = Data(cfg)
background = load_tensor_from_source("imagenet_samples", n_samples=2)
```

That pattern is exactly how SHAP background sources are resolved later by `raitap.transparency.factory`.
