"""
Data module, handles:

- loading data from various sources (local files, URLs, demo samples)
- converting to raw tensors for model input
- hosting a list of demo samples
"""

from .data import Data, load_tensor_from_source

__all__ = [
    "Data",
    "load_tensor_from_source",
]
