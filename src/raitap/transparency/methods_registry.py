"""
Method registry for RAITAP transparency module.

This module defines the explainer methods and visualisers supported by RAITAP.
Methods are curated to ensure they work with our visualization pipeline.
"""

from __future__ import annotations


class ExplainerMethod:
    """
    Descriptor that captures attribute name as algorithm name.

    Uses __set_name__ magic method to eliminate duplication.
    """

    def __init__(self):
        self.framework: str | None = None
        self.algorithm: str | None = None

    def __set_name__(self, owner, name):
        """Called when assigned as class attribute"""
        self.framework = owner.__framework__
        self.algorithm = name

    def __repr__(self):
        return f"ExplainerMethod({self.framework!r}, {self.algorithm!r})"


class Captum:
    """
    Captum attribution methods supported by RAITAP.

    All methods have been tested for compatibility with our visualisers.
    See: https://captum.ai/api/
    """

    __framework__ = "captum"

    # Zero duplication: algorithm name = attribute name
    IntegratedGradients = ExplainerMethod()
    Saliency = ExplainerMethod()
    LayerGradCam = ExplainerMethod()  # Note: Called "GradCAM" in papers, "LayerGradCam" in Captum
    DeepLift = ExplainerMethod()
    GuidedBackprop = ExplainerMethod()
    # Add new methods here after testing compatibility


class SHAP:
    """
    SHAP explainer types supported by RAITAP.

    All explainers have been tested for compatibility with our visualisers.
    See: https://shap.readthedocs.io/

    Note on background_data requirement:
        REQUIRED: GradientExplainer, DeepExplainer, KernelExplainer
        OPTIONAL: TreeExplainer (uses data from model)
    """

    __framework__ = "shap"

    GradientExplainer = ExplainerMethod()
    DeepExplainer = ExplainerMethod()
    KernelExplainer = ExplainerMethod()
    TreeExplainer = ExplainerMethod()
    # Add new explainers here after testing compatibility


# ---------------------------------------------------------------------------
# Visualiser compatibility registry
# ---------------------------------------------------------------------------
# Maps  framework -> visualiser_name -> frozenset of compatible algorithms
# ``None`` means the visualiser is compatible with ALL algorithms in that framework.
#
# Captum visualisers (from captum.attr.visualization):
#   "image"       - visualize_image_attr      (any Captum algorithm on image input)
#   "time_series" - visualize_timeseries_attr  (any Captum algorithm on time-series)
#   "text"        - visualize_text_attr        (any Captum algorithm on text input)
#
# SHAP visualisers:
#   "bar"       - shap.summary_plot(plot_type="bar")  (all explainers)
#   "beeswarm"  - shap.summary_plot()                 (all explainers)
#   "waterfall" - per-sample waterfall chart           (all explainers)
#   "force"     - per-sample force plot                (all explainers)
#   "image"     - shap.image_plot()  ONLY GradientExplainer / DeepExplainer
#                 (requires pixel-level SHAP values from gradient-based explainers)
# ---------------------------------------------------------------------------

VISUALISER_REGISTRY: dict[str, dict[str, frozenset[str] | None]] = {
    "captum": {
        "image": None,  # compatible with all Captum algorithms
        "time_series": None,  # compatible with all Captum algorithms
        "text": None,  # compatible with all Captum algorithms
    },
    "shap": {
        "bar": None,  # compatible with all SHAP explainers
        "beeswarm": None,  # compatible with all SHAP explainers
        "waterfall": None,  # compatible with all SHAP explainers
        "force": None,  # compatible with all SHAP explainers
        # Restricted: requires pixel-level SHAP values (gradient-based explainers only)
        "image": frozenset({"GradientExplainer", "DeepExplainer"}),
    },
}


def validate_visualiser_compatibility(framework: str, visualiser: str, algorithm: str) -> None:
    """
    Raise ``VisualiserIncompatibilityError`` if *visualiser* cannot be used
    with *algorithm* under *framework*.

    Args:
        framework:  e.g. ``"captum"`` or ``"shap"``
        visualiser: e.g. ``"image"`` or ``"beeswarm"``
        algorithm:  e.g. ``"IntegratedGradients"`` or ``"KernelExplainer"``

    Raises:
        VisualiserIncompatibilityError: When the combination is not supported.
        ValueError: When the visualiser name is unknown for the given framework.
    """
    fw_registry = VISUALISER_REGISTRY.get(framework)
    if fw_registry is None:
        raise ValueError(
            f"Unknown framework {framework!r}. Supported: {list(VISUALISER_REGISTRY)}."
        )

    if visualiser not in fw_registry:
        available = list(fw_registry)
        raise ValueError(
            f"Unknown visualiser {visualiser!r} for framework {framework!r}.\n"
            f"Supported: {available}"
        )

    compatible = fw_registry[visualiser]
    if compatible is not None and algorithm not in compatible:
        raise VisualiserIncompatibilityError(
            framework=framework,
            visualiser=visualiser,
            algorithm=algorithm,
            compatible_algorithms=sorted(compatible),
        )


class VisualiserIncompatibilityError(Exception):
    """Raised when a visualiser is not compatible with the chosen explainer algorithm."""

    def __init__(
        self,
        framework: str,
        visualiser: str,
        algorithm: str,
        compatible_algorithms: list[str],
    ):
        self.framework = framework
        self.visualiser = visualiser
        self.algorithm = algorithm
        self.compatible_algorithms = compatible_algorithms
        super().__init__(
            f"Visualiser {visualiser!r} is not compatible with "
            f"{framework}.{algorithm}.\n"
            f"Compatible algorithms: {', '.join(compatible_algorithms) or 'none'}."
        )


# Registry of all framework classes
# Add new framework classes here to make them available in config schema
FRAMEWORK_REGISTRY = [Captum, SHAP]


def get_framework_names() -> list[str]:
    """Get list of registered framework names for schema generation."""
    return [cls.__framework__ for cls in FRAMEWORK_REGISTRY]


# Convenience aliases for common methods
IntegratedGradients = Captum.IntegratedGradients
Saliency = Captum.Saliency
GradCAM = Captum.LayerGradCam  # Paper name → Captum class name
KernelShap = SHAP.KernelExplainer  # Popular name in literature
