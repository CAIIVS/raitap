"""
Data module: format-agnostic data loading.
"""

from .loader import load_data, resolve_data_source
from .samples import SAMPLE_SOURCES

__all__ = [
    "SAMPLE_SOURCES",
    "load_data",
    "resolve_data_source",
]
