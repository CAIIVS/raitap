"""SHAP visualisers for RAITAP.

Most visualisers wrap SHAP's plotting APIs (``shap.plots.*`` /
``shap.summary_plot``). ``ShapImageVisualiser`` is rendered manually with
Matplotlib so RAITAP can provide a consistent paired image/overlay layout,
titles, sample names, and colorbar handling. Its per-panel rendering recipe
reproduces ``shap.plots.image`` — a grayscale background under a
``red_transparent_blue`` diverging heatmap with a symmetric
``±np.nanpercentile(|attribution|, 99.9)`` colormap scale. The
``red_transparent_blue`` colormap is SHAP's own, imported lazily from
``shap.plots.colors`` (``shap`` is already a runtime dependency of this
visualiser).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np

from raitap import raitap_log
from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationScope,
    MethodFamily,
    ScopeDefinitionStep,
    VisualSummarySpec,
)
from raitap.transparency.visualisers.image_rendering import ShapNativeRenderer
from raitap.transparency.visualisers.registration import transparency_visualiser

from .base_visualiser import BaseVisualiser

if TYPE_CHECKING:
    import torch
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from raitap.transparency.contracts import VisualisationContext


def _red_transparent_blue() -> Colormap:
    """SHAP's ``red_transparent_blue`` diverging colormap, imported lazily.

    ``shap`` is a runtime dependency of this visualiser (``visualise`` already
    imports it), so the colormap is fetched from ``shap.plots.colors`` on demand
    rather than vendored — keeping RAITAP in sync with SHAP's own definition.
    """
    from shap.plots.colors import red_transparent_blue

    return red_transparent_blue


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor or array-like to numpy (float32)."""
    val: Any = x
    if hasattr(val, "detach"):
        val = val.detach()
    if hasattr(val, "cpu"):
        val = val.cpu()
    if hasattr(val, "numpy"):
        val = val.numpy()
    return np.asarray(val, dtype=np.float32)


def _close_and_return(fig: Figure) -> Figure:
    """Detach a figure from pyplot's interactive state then return it."""
    plt.close(fig)
    return fig


def _normalise_image(image: np.ndarray) -> np.ndarray:
    """Normalise an image array to the [0, 1] range when possible."""
    lo, hi = image.min(), image.max()
    if hi > lo:
        return (image - lo) / (hi - lo)
    return image


def _compose_title(base_title: str | None, sample_name: str | None = None) -> str | None:
    """Combine a base title and optional sample name.

    ``None`` means no base title is set and allows falling back to the sample
    name. An explicitly provided empty string is preserved as-is.
    """
    if base_title is None:
        return sample_name
    if base_title and sample_name:
        return f"{base_title}: {sample_name}"
    return base_title


def _resolve_title(
    *, explicit_title: str | None, fallback_title: str | None, sample_name: str | None = None
) -> str | None:
    """Resolve titles while preserving an explicitly provided empty string."""
    base_title = explicit_title if explicit_title is not None else fallback_title
    return _compose_title(base_title, sample_name)


def _display_image(ax: Any, image: np.ndarray) -> None:
    """Render an image on an axis with sensible defaults for grayscale/RGB input."""
    if image.ndim == 3 and image.shape[-1] == 1:
        ax.imshow(image[..., 0], cmap="gray")
        return
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)


def _default_image_title(algorithm: str | None) -> str:
    """Return the default SHAP attribution title for image plots."""
    return f"{algorithm} (SHAP)" if algorithm else "SHAP Image"


def _symmetric_vmin_vmax(values: np.ndarray, outlier_perc: float = 99.9) -> tuple[float, float]:
    """Symmetric ``(vmin, vmax)`` for ``imshow`` using a percentile of ``|values|``.

    Matches the normalisation used by ``shap.plots.image``:
    ``max_val = np.nanpercentile(|values|, outlier_perc)``. Falls back to
    ``(-1.0, 1.0)`` for degenerate inputs (empty, all-zero, or non-finite)
    so the heatmap stays renderable.
    """
    abs_vals = np.abs(values)
    if abs_vals.size == 0:
        return -1.0, 1.0
    max_val = float(np.nanpercentile(abs_vals, outlier_perc))
    if not np.isfinite(max_val) or max_val == 0.0:
        return -1.0, 1.0
    return -max_val, max_val


def _rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Reduce an image to a 2-D grayscale array for use as a background under a heatmap.

    For RGB inputs the standard luminosity weights are used
    (``0.2989·R + 0.5870·G + 0.1140·B``), matching ``shap.plots.image``. For
    other multi-channel inputs (including 1-channel ``(H, W, 1)``) the per-channel
    mean is used. 2-D inputs are returned unchanged.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        if image.shape[-1] == 3:
            return 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
        return image.mean(axis=-1)
    raise ValueError(f"_rgb_to_grayscale expected 2D or 3D image, got shape {image.shape}.")


def _image_heatmap(values: np.ndarray) -> np.ndarray:
    """Reduce channel-wise SHAP values to a 2-D heatmap for display."""
    if values.ndim == 3:
        if values.shape[-1] == 1:
            return values[..., 0]
        return values.sum(axis=-1)
    return values


def _input_kind(explanation: object) -> str:
    semantics = getattr(explanation, "semantics", None)
    input_spec = getattr(semantics, "input_spec", None)
    return str(getattr(input_spec, "kind", "") or "").lower()


def _input_layout(explanation: object) -> str:
    semantics = getattr(explanation, "semantics", None)
    input_spec = getattr(semantics, "input_spec", None)
    return str(getattr(input_spec, "layout", "") or "").upper().replace(" ", "")


def _input_metadata(explanation: object) -> dict[str, object]:
    semantics = getattr(explanation, "semantics", None)
    input_spec = getattr(semantics, "input_spec", None)
    metadata = getattr(input_spec, "metadata", None)
    return dict(metadata) if metadata is not None else {}


def _has_explicit_image_metadata(explanation: object) -> bool:
    kind = _input_kind(explanation)
    if kind == "image":
        return True
    if kind:
        return False
    metadata = _input_metadata(explanation)
    return any(
        str(metadata.get(key, "")).lower() == "image"
        for key in ("modality", "input_kind", "data_kind", "data_type")
    )


def _output_layout(explanation: object) -> str:
    semantics = getattr(explanation, "semantics", None)
    output_space = getattr(semantics, "output_space", None)
    return str(getattr(output_space, "layout", "") or "").upper().replace(" ", "")


def _output_shape(
    explanation: object, attributions: object | None = None
) -> tuple[int, ...] | None:
    semantics = getattr(explanation, "semantics", None)
    output_space = getattr(semantics, "output_space", None)
    shape = getattr(output_space, "shape", None)
    if shape is None and attributions is not None:
        shape = getattr(attributions, "shape", None)
    return None if shape is None else tuple(int(dim) for dim in shape)


def _has_image_layout(explanation: object, attributions: object) -> bool:
    layouts = {_input_layout(explanation), _output_layout(explanation)}
    if any(layout and layout != "NCHW" for layout in layouts):
        return False
    shape = _output_shape(explanation, attributions)
    return shape is not None and len(shape) >= 3


def _is_tabular_output(explanation: object) -> bool:
    kind = _input_kind(explanation)
    if kind in {"image", "text", "time_series", "timeseries"}:
        return False
    return (
        kind == "tabular"
        or _input_layout(explanation) in {"B,F", "(B,F)"}
        or _output_layout(explanation)
        in {
            "B,F",
            "(B,F)",
        }
    )


class _TabularSummaryContractMixin(BaseVisualiser):
    supported_scopes: ClassVar[frozenset[ExplanationScope]] = frozenset({ExplanationScope.LOCAL})
    supported_output_spaces: ClassVar[frozenset[ExplanationOutputSpace]] = frozenset(
        {
            ExplanationOutputSpace.INPUT_FEATURES,
            ExplanationOutputSpace.INTERPRETABLE_FEATURES,
        }
    )
    supported_method_families: ClassVar[frozenset[MethodFamily]] = frozenset({MethodFamily.SHAPLEY})
    produces_scope: ClassVar[ExplanationScope | None] = ExplanationScope.AGGREGATED
    scope_definition_step: ClassVar[ScopeDefinitionStep | None] = (
        ScopeDefinitionStep.VISUALISER_SUMMARY
    )

    def validate_explanation(
        self,
        explanation: object,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None,
    ) -> None:
        super().validate_explanation(explanation, attributions, inputs)
        if not _is_tabular_output(explanation):
            actual_layout = _input_kind(explanation)
            actual_layout = actual_layout or _input_layout(explanation)
            actual_layout = actual_layout or _output_layout(explanation)
            self._raise_incompatibility(
                "tabular layout",
                actual_layout,
                "(B, F) tabular/interpretable attributions",
            )


# ---------------------------------------------------------------------------
# Tabular / general visualisers
# Compatible with all SHAP explainer algorithms.
# ---------------------------------------------------------------------------


@transparency_visualiser(
    registry_name="shap_bar",
    visual_summary=VisualSummarySpec(
        summary_type="bar",
        aggregation="mean_absolute_attribution",
        description="Mean absolute attribution by feature.",
    ),
)
class ShapBarVisualiser(_TabularSummaryContractMixin):
    """
    Mean absolute SHAP value bar chart via ``shap.summary_plot(plot_type='bar')``.

    Compatible with all SHAP explainer algorithms.
    """

    def __init__(self, feature_names: list[str] | None = None, max_display: int = 20):
        self.feature_names = feature_names
        self.max_display = max_display

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        **kwargs: Any,
    ) -> Figure:
        """
        Args:
            attributions: ``(B, F)`` SHAP values tensor / array.
            inputs:       Original feature values ``(B, F)`` (used for colouring).
            context:      Standard RAITAP metadata (not used by this visualiser).
            **kwargs:     Forwarded to ``shap.summary_plot``.
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "SHAP visualiser is enabled but shap is not installed. "
                "Install it with `uv sync --extra shap`."
            ) from e

        values = _to_numpy(attributions)
        feats = _to_numpy(inputs) if inputs is not None else None

        fig = plt.figure()
        shap.summary_plot(
            values,
            features=feats,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=self.max_display,
            show=False,
            **kwargs,
        )
        return fig


@transparency_visualiser(
    registry_name="shap_beeswarm",
    visual_summary=VisualSummarySpec(
        summary_type="beeswarm",
        aggregation="distribution_summary",
        description="Distribution of local attributions by feature.",
    ),
)
class ShapBeeswarmVisualiser(_TabularSummaryContractMixin):
    """
    SHAP beeswarm summary plot via ``shap.summary_plot()``.

    Compatible with all SHAP explainer algorithms.
    """

    def __init__(self, feature_names: list[str] | None = None, max_display: int = 20):
        self.feature_names = feature_names
        self.max_display = max_display

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        **kwargs: Any,
    ) -> Figure:
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "SHAP visualiser is enabled but shap is not installed. "
                "Install it with `uv sync --extra shap`."
            ) from e

        values = _to_numpy(attributions)
        feats = _to_numpy(inputs) if inputs is not None else None

        fig = plt.figure()
        shap.summary_plot(
            values,
            features=feats,
            feature_names=self.feature_names,
            max_display=self.max_display,
            show=False,
            **kwargs,
        )
        return fig


@transparency_visualiser(registry_name="shap_waterfall")
class ShapWaterfallVisualiser(BaseVisualiser):
    """
    Per-sample SHAP waterfall chart via ``shap.plots.waterfall``.

    Compatible with all SHAP explainer algorithms.
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        expected_value: float = 0.0,
        sample_index: int = 0,
        max_display: int = 10,
    ):
        """
        Args:
            feature_names:  Optional list of feature labels.
            expected_value: Model baseline / expected output value.
                            Most SHAP explainers expose this as
                            ``explainer.expected_value``.
            sample_index:   Which sample from the batch to visualise.
            max_display:    Maximum number of features to show.
        """
        self.feature_names = feature_names
        self.expected_value = expected_value
        self.sample_index = sample_index
        self.max_display = max_display

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        **kwargs: Any,
    ) -> Figure:
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "SHAP visualiser is enabled but shap is not installed. "
                "Install it with `uv sync --extra shap`."
            ) from e

        values = _to_numpy(attributions)
        sample_vals = values[self.sample_index]

        feats = None
        if inputs is not None:
            feats = _to_numpy(inputs)[self.sample_index]

        feature_names = (
            self.feature_names if self.feature_names else [f"f{i}" for i in range(len(sample_vals))]
        )

        explanation = shap.Explanation(
            values=sample_vals,
            base_values=self.expected_value,
            data=feats,
            feature_names=feature_names,
        )

        shap.plots.waterfall(explanation, max_display=self.max_display, show=False, **kwargs)
        fig = plt.gcf()
        return _close_and_return(fig)


@transparency_visualiser(registry_name="shap_force")
class ShapForceVisualiser(BaseVisualiser):
    """
    Per-sample SHAP force plot via ``shap.plots.force`` (matplotlib backend).

    Compatible with all SHAP explainer algorithms.
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        expected_value: float = 0.0,
        sample_index: int = 0,
    ):
        self.feature_names = feature_names
        self.expected_value = expected_value
        self.sample_index = sample_index

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        **kwargs: Any,
    ) -> Figure:
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "SHAP visualiser is enabled but shap is not installed. "
                "Install it with `uv sync --extra shap`."
            ) from e

        values = _to_numpy(attributions)
        sample_vals = values[self.sample_index]

        feats = None
        if inputs is not None:
            feats = _to_numpy(inputs)[self.sample_index]

        feature_names = (
            self.feature_names if self.feature_names else [f"f{i}" for i in range(len(sample_vals))]
        )

        shap.plots.force(
            base_value=self.expected_value,
            shap_values=sample_vals,
            features=feats,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            **kwargs,
        )
        fig = plt.gcf()
        return _close_and_return(fig)


# ---------------------------------------------------------------------------
# Image visualiser — RESTRICTED to GradientExplainer / DeepExplainer
# ---------------------------------------------------------------------------


@transparency_visualiser(
    registry_name="shap_image",
    compatible_algorithms=frozenset({"GradientExplainer", "DeepExplainer"}),
    supported_scopes=frozenset({ExplanationScope.LOCAL}),
    supported_output_spaces=frozenset({ExplanationOutputSpace.INPUT_FEATURES}),
    supported_method_families=frozenset({MethodFamily.GRADIENT}),
    embeds_original_input=True,
)
class ShapImageVisualiser(BaseVisualiser):
    """
    Render image-level SHAP attributions with Matplotlib.

    This visualiser does not call ``shap.image_plot`` directly; instead it
    renders a RAITAP-managed figure that follows the same rendering recipe
    as ``shap.plots.image`` — a grayscale background at
    ``overlay_alpha=0.15``, a ``red_transparent_blue`` diverging colormap,
    and a symmetric ``±np.nanpercentile(|attribution|, outlier_perc)``
    colormap scale (``outlier_perc=99.9`` by default). RAITAP layers
    sample-aware titles, the paired-original panel, and the colorbar on
    top of that recipe.

    .. warning::
       **Only compatible with ``GradientExplainer`` and ``DeepExplainer``.**
       These are the only SHAP explainers that compute pixel-level SHAP
       values suitable for image visualisation. Passing attributions from
       other explainers will produce meaningless plots.

    Positive contributions are shown in red and negative contributions in
    blue with a transparent mid-range, matching SHAP's native presentation.
    """

    def validate_explanation(
        self,
        explanation: object,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None,
    ) -> None:
        super().validate_explanation(explanation, attributions, inputs)
        if not _has_explicit_image_metadata(explanation) or not _has_image_layout(
            explanation, attributions
        ):
            self._raise_incompatibility("input metadata", _input_kind(explanation), "image")

    def __init__(
        self,
        max_samples: int = 4,
        title: str | None = None,
        include_original_image: bool = True,
        show_colorbar: bool = True,
        cmap: Any = None,
        overlay_alpha: float = 0.15,
        outlier_perc: float = 99.9,
    ):
        """
        Args:
            max_samples: Maximum number of images to display side by side.
            title: Optional attribution panel title.
            include_original_image: Whether to render the original image next to
                the attribution heatmap when ``inputs`` are provided.
            show_colorbar: Whether to add a SHAP colorbar in the paired layout.
            cmap: Matplotlib colormap for the SHAP heatmap overlay.
                ``None`` (default) uses SHAP's ``red_transparent_blue`` diverging
                colormap, matching ``shap.plots.image`` (resolved at render time).
            overlay_alpha: Alpha of the grayscale background drawn under the
                colored SHAP heatmap. Default ``0.15`` matches
                ``shap.plots.image``.
            outlier_perc: Percentile used to compute the symmetric
                ``±nanpercentile(|attribution|, outlier_perc)`` colormap scale.
                Default ``99.9`` matches ``shap.plots.image``.
        """
        self.max_samples = max_samples
        self.title = title
        self.include_original_image = include_original_image
        self.show_colorbar = show_colorbar
        self.cmap = cmap
        self.overlay_alpha = overlay_alpha
        self.outlier_perc = outlier_perc

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        max_samples: int | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Figure:
        """
        Args:
            attributions: ``(B, C, H, W)`` SHAP values tensor / array.
            inputs:        Original images ``(B, C, H, W)`` for background.
            context:       Standard RAITAP metadata (optional).
            max_samples:   Maximum number of images to display.
            title:         Optional attribution panel title. Overrides the algorithm-based
                           default title, even when set to an empty string.
            **kwargs:      Optional visual styling overrides.

        Returns:
            Matplotlib Figure.
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "SHAP visualiser is enabled but shap is not installed. "
                "Install it with `uv sync --extra shap`."
            ) from e

        del shap

        shap_vals = _to_numpy(attributions)  # (B, C, H, W)
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[np.newaxis]
        n = min(shap_vals.shape[0], self.max_samples if max_samples is None else max_samples)
        shap_vals = shap_vals[:n]

        pixel_values = None
        if inputs is not None:
            pv = _to_numpy(inputs)[:n]  # (B, C, H, W)
            pixel_values = np.transpose(pv, (0, 2, 3, 1))  # (B, H, W, C)
            pixel_values = np.stack([_normalise_image(image) for image in pixel_values], axis=0)

        sample_names = context.sample_names if context is not None else None
        show_sample_names = context.show_sample_names if context is not None else False
        algorithm = context.algorithm if context is not None else None

        names = [] if sample_names is None else [str(name) for name in sample_names[:n]]
        cmap = kwargs.pop("cmap", self.cmap)
        if cmap is None:
            cmap = _red_transparent_blue()
        overlay_alpha = kwargs.pop("overlay_alpha", self.overlay_alpha)
        outlier_perc = float(kwargs.pop("outlier_perc", self.outlier_perc))
        show_colorbar = bool(kwargs.pop("show_colorbar", self.show_colorbar))
        if "include_original_input" in kwargs:
            include_original_image = bool(kwargs.pop("include_original_input"))
            kwargs.pop("include_original_image", None)
        elif "include_original_image" in kwargs:
            raitap_log.warn(
                "`include_original_image` as a render-time kwarg is deprecated; "
                "use `include_original_input` instead.",
                category=DeprecationWarning,
                stacklevel=3,
            )
            include_original_image = bool(kwargs.pop("include_original_image"))
        else:
            include_original_image = self.include_original_image

        show_original_panel = include_original_image and pixel_values is not None
        if show_original_panel and show_colorbar:
            fig, axes = plt.subplots(
                n,
                3,
                figsize=(8.5, 4 * n),
                squeeze=False,
                gridspec_kw={"width_ratios": [1, 1, 0.05]},
            )
        elif show_original_panel:
            fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n), squeeze=False)
        elif show_colorbar:
            fig, axes = plt.subplots(
                n,
                2,
                figsize=(4.5, 4 * n),
                squeeze=False,
                gridspec_kw={"width_ratios": [1, 0.05]},
            )
        else:
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
            axes = np.array([axes], dtype=object) if n == 1 else np.asarray(axes, dtype=object)

        fallback_attr_title = (
            self.title if self.title is not None else _default_image_title(algorithm)
        )

        for i in range(n):
            shap_i = (
                np.transpose(shap_vals[i], (1, 2, 0)) if shap_vals[i].ndim == 3 else shap_vals[i]
            )
            sample_name = names[i] if show_sample_names and i < len(names) else None
            original_title = _compose_title("Original Image", sample_name)
            attr_title = _resolve_title(
                explicit_title=title,
                fallback_title=fallback_attr_title,
                sample_name=sample_name,
            )

            image_i = None if pixel_values is None else pixel_values[i]

            if show_original_panel:
                colorbar_ax = None
                if show_colorbar:
                    original_ax, attr_ax, colorbar_ax = axes[i]
                else:
                    original_ax, attr_ax = axes[i]

                if image_i is not None:
                    _display_image(original_ax, image_i)
                original_ax.set_title(original_title or "")
                original_ax.axis("off")
            else:
                if show_colorbar:
                    attr_ax, colorbar_ax = axes[i]
                else:
                    attr_ax = axes[i]
                    colorbar_ax = None

            im = ShapNativeRenderer().draw(
                attr_ax,
                shap_i,
                image_i,
                cmap=cmap,
                overlay_alpha=overlay_alpha,
                outlier_perc=outlier_perc,
            )
            attr_ax.set_title(attr_title or "")
            attr_ax.axis("off")

            if colorbar_ax is not None:
                fig.colorbar(im, cax=colorbar_ax)
                colorbar_ax.set_ylabel("SHAP value")

        return fig
