"""
Convenience re-export of RAITAP transparency methods registry.

Importing from this module provides both the registry classes and
common aliases used throughout the codebase.
"""

from __future__ import annotations

from .methods_registry import (
    FRAMEWORK_REGISTRY,
    SHAP,
    Captum,
    ExplainerMethod,
    get_framework_names,
)

# Convenience aliases - direct method references for common use cases
IntegratedGradients = Captum.IntegratedGradients
Saliency = Captum.Saliency
LayerGradCam = Captum.LayerGradCam
DeepLift = Captum.DeepLift
GuidedBackprop = Captum.GuidedBackprop

KernelShap = SHAP.KernelExplainer  # Alias matching common name

__all__ = [
    # Registry classes
    "Captum",
    "SHAP",
    "ExplainerMethod",
    "FRAMEWORK_REGISTRY",
    "get_framework_names",
    # Captum aliases
    "IntegratedGradients",
    "Saliency",
    "LayerGradCam",
    "DeepLift",
    "GuidedBackprop",
    # SHAP aliases
    "KernelShap",
]
