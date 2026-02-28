"""
Data module: format-agnostic data loading.
"""

from .loader import load_data, resolve_data_source
from .samples import SAMPLE_SOURCES

__all__ = [
    "load_data",
    "resolve_data_source",
    "SAMPLE_SOURCES",
]
