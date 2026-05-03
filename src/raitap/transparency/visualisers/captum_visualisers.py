"""Captum-native visualisers (wrapping captum.attr.visualization)"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np

from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationScope,
    MethodFamily,
)

from .base_visualiser import BaseVisualiser

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure

    from raitap.transparency.contracts import VisualisationContext


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor or array-like to numpy."""
    val: Any = x
    if hasattr(val, "detach"):
        val = val.detach()
    if hasattr(val, "cpu"):
        val = val.cpu()
    if hasattr(val, "numpy"):
        val = val.numpy()
    return np.asarray(val)


def _resize_attr_to_hw(attr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize attribution map to target (H, W) while preserving channels-last shape."""
    import torch
    import torch.nn.functional as f

    if attr.ndim == 2:
        tensor = torch.from_numpy(attr).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
        resized = f.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
        return resized.squeeze(0).squeeze(0).cpu().numpy()

    if attr.ndim == 3:
        tensor = torch.from_numpy(np.transpose(attr, (2, 0, 1))).unsqueeze(0).float()  # (1,C,H,W)
        resized = f.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
        return np.transpose(resized.squeeze(0).cpu().numpy(), (1, 2, 0))

    return attr


def _normalise_image(image: np.ndarray) -> np.ndarray:
    """Normalise an image array to the [0, 1] range when possible."""
    lo, hi = image.min(), image.max()
    if hi > lo:
        return (image - lo) / (hi - lo)
    return image


def _method_label(method: str) -> str:
    """Convert a Captum method key into a human-readable plot title."""
    return method.replace("_", " ").title()


def _compose_title(base_title: str | None, sample_name: str | None = None) -> str | None:
    """Combine a base title and optional sample name."""
    if base_title and sample_name:
        return f"{base_title}: {sample_name}"
    return base_title or sample_name


def _resolve_title(
    *, explicit_title: str | None, fallback_title: str | None, sample_name: str | None = None
) -> str | None:
    """Resolve titles while preserving an explicitly provided empty string."""
    base_title = explicit_title if explicit_title is not None else fallback_title
    return _compose_title(base_title, sample_name)


_CAPTUM_SEQUENCE_METHOD_FAMILIES = frozenset(
    {
        MethodFamily.GRADIENT,
        MethodFamily.PERTURBATION,
        MethodFamily.SHAPLEY,
        MethodFamily.MODEL_AGNOSTIC,
        MethodFamily.SURROGATE,
    }
)


def _last_mappable(ax: Any) -> Any:
    """Return the last Matplotlib mappable drawn on an axis, if any."""
    if getattr(ax, "images", None):
        return ax.images[-1]
    if getattr(ax, "collections", None):
        return ax.collections[-1]
    return None


def _input_kind(explanation: object) -> str:
    semantics = getattr(explanation, "semantics", None)
    input_spec = getattr(semantics, "input_spec", None)
    return str(getattr(input_spec, "kind", "") or "").lower()


def _input_layout(explanation: object) -> str:
    semantics = getattr(explanation, "semantics", None)
    input_spec = getattr(semantics, "input_spec", None)
    return str(getattr(input_spec, "layout", "") or "").upper().replace(" ", "")


def _output_layout(explanation: object) -> str:
    semantics = getattr(explanation, "semantics", None)
    output_space = getattr(semantics, "output_space", None)
    return str(getattr(output_space, "layout", "") or "").upper().replace(" ", "")


def _output_shape(explanation: object, attributions: object) -> tuple[int, ...] | None:
    semantics = getattr(explanation, "semantics", None)
    output_space = getattr(semantics, "output_space", None)
    shape = getattr(output_space, "shape", None)
    if shape is None:
        shape = getattr(attributions, "shape", None)
    return None if shape is None else tuple(int(dim) for dim in shape)


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


def _has_image_layout(explanation: object, attributions: object) -> bool:
    layouts = {_input_layout(explanation), _output_layout(explanation)}
    if any(layout and layout != "NCHW" for layout in layouts):
        return False
    shape = _output_shape(explanation, attributions)
    return shape is not None and len(shape) >= 3


class CaptumImageVisualiser(BaseVisualiser):
    """
    Visualise image attributions using ``captum.attr.visualization.visualize_image_attr``.

    Wraps the Captum native function so the output is a Matplotlib Figure
    that can be saved or returned by ``explain()``.

    Compatible with ALL Captum attribution algorithms.
    """

    supported_scopes: ClassVar[frozenset[ExplanationScope]] = frozenset({ExplanationScope.LOCAL})
    supported_output_spaces: ClassVar[frozenset[ExplanationOutputSpace]] = frozenset(
        {
            ExplanationOutputSpace.INPUT_FEATURES,
            ExplanationOutputSpace.IMAGE_SPATIAL_MAP,
        }
    )
    supported_method_families: ClassVar[frozenset[MethodFamily]] = frozenset(
        {
            MethodFamily.GRADIENT,
            MethodFamily.PERTURBATION,
            MethodFamily.SHAPLEY,
            MethodFamily.CAM,
            MethodFamily.MODEL_AGNOSTIC,
            MethodFamily.SURROGATE,
        }
    )

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
        method: str = "blended_heat_map",
        sign: str = "all",
        show_colorbar: bool = True,
        title: str | None = None,
        include_original_image: bool = True,
    ):
        """
        Args:
            method:       Captum viz method. Options:
                          ``"blended_heat_map"``, ``"heat_map"``,
                          ``"original_image"``, ``"masked_image"``,
                          ``"alpha_scaling"``.  Default: ``"blended_heat_map"``.
            sign:         Which signs to visualise.  Options:
                          ``"all"``, ``"positive"``, ``"negative"``,
                          ``"absolute_value"``.  Default: ``"all"``.
            show_colorbar: Whether to add a colorbar.  Default: ``True``.
            title:        Optional subplot title forwarded to Captum.
            include_original_image: Whether to render the original image next to
                          the attribution plot when ``inputs`` are provided.
                          Default: ``True``.
        """
        self.method = method
        self.sign = sign
        self.show_colorbar = show_colorbar
        self.title = title
        self.include_original_image = include_original_image

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        max_samples: int = 8,
        **kwargs: Any,
    ) -> Figure:
        """
        Args:
            attributions: ``(B, C, H, W)`` or ``(B, H, W)`` tensor / array.
            inputs:       Original images ``(B, C, H, W)`` for overlay.
            context:      Standard RAITAP metadata (optional).
            max_samples:  Maximum number of samples to display (default: 8).
            **kwargs:     Forwarded to ``visualize_image_attr``.

        Returns:
            Matplotlib Figure with one column per sample.
        """
        try:
            from captum.attr import visualization as viz
        except ImportError as e:
            raise ImportError(
                "Captum visualiser is enabled but captum is not installed. "
                "Install it with `uv sync --extra captum`."
            ) from e

        attrs = _to_numpy(attributions)
        # Ensure batch dimension
        if attrs.ndim == 3:
            attrs = attrs[np.newaxis]  # (1, C, H, W)
        n = min(attrs.ndim and attrs.shape[0], max_samples)
        attrs = attrs[:n]

        origs = None
        if inputs is not None:
            origs = _to_numpy(inputs)
            if origs.ndim == 3:
                origs = origs[np.newaxis]
            origs = origs[:n]

        sample_names = context.sample_names if context is not None else None
        show_sample_names = context.show_sample_names if context is not None else False
        names = [] if sample_names is None else [str(name) for name in sample_names[:n]]

        show_original_panel = (
            self.include_original_image and origs is not None and self.method != "original_image"
        )

        if show_original_panel and self.show_colorbar:
            fig, axes = plt.subplots(
                n,
                3,
                figsize=(8.5, 4 * n),
                squeeze=False,
                gridspec_kw={"width_ratios": [1, 1, 0.05]},
            )
        elif show_original_panel:
            fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n), squeeze=False)
        else:
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
            axes = np.atleast_1d(axes)

        for i in range(n):
            # (C, H, W) -> (H, W, C)
            attr_i = np.transpose(attrs[i], (1, 2, 0)) if attrs[i].ndim == 3 else attrs[i]

            orig_i = None
            if origs is not None:
                orig_i = np.transpose(origs[i], (1, 2, 0)) if origs[i].ndim == 3 else origs[i]
                orig_i = _normalise_image(orig_i)

                # Layer methods (e.g., LayerGradCam) can yield low-res maps;
                # masked modes need same HxW.
                if self.method in {"masked_image", "alpha_scaling"}:
                    attr_hw = attr_i.shape[:2]
                    orig_hw = orig_i.shape[:2]
                    if attr_hw != orig_hw:
                        attr_i = _resize_attr_to_hw(attr_i, orig_hw)

            sample_name = names[i] if show_sample_names and i < len(names) else None

            if show_original_panel:
                colorbar_ax = None
                if self.show_colorbar:
                    original_ax, attr_ax, colorbar_ax = axes[i]
                else:
                    original_ax, attr_ax = axes[i]
                original_title = _compose_title("Original Image", sample_name)
                explicit_title = kwargs.get("title")
                attr_title = _resolve_title(
                    explicit_title=explicit_title,
                    fallback_title=self.title or _method_label(self.method),
                    sample_name=sample_name,
                )

                _, _ = viz.visualize_image_attr(
                    attr_i,
                    orig_i,
                    method="original_image",
                    sign="all",
                    show_colorbar=False,
                    plt_fig_axis=(fig, original_ax),
                    use_pyplot=False,
                    title=original_title,
                )

                attr_viz_kwargs = dict(kwargs)
                attr_viz_kwargs["title"] = attr_title
                _, _ = viz.visualize_image_attr(
                    attr_i,
                    orig_i,
                    method=self.method,
                    sign=self.sign,
                    show_colorbar=False,
                    plt_fig_axis=(fig, attr_ax),
                    use_pyplot=False,
                    **attr_viz_kwargs,
                )
                if self.show_colorbar and colorbar_ax is not None:
                    mappable = _last_mappable(attr_ax)
                    if mappable is None:
                        colorbar_ax.set_visible(False)
                    else:
                        fig.colorbar(mappable, cax=colorbar_ax)
                continue

            ax = axes[i]
            viz_kwargs = dict(kwargs)
            if self.title is not None and "title" not in viz_kwargs:
                viz_kwargs["title"] = self.title

            # Let Captum render into our existing axes
            _, _ = viz.visualize_image_attr(
                attr_i,
                orig_i,
                method=self.method,
                sign=self.sign,
                show_colorbar=self.show_colorbar,
                plt_fig_axis=(fig, ax),
                use_pyplot=False,
                **viz_kwargs,
            )
            if show_sample_names and i < len(names):
                base_title = ax.get_title().strip()
                label = f"{base_title}: {names[i]}" if base_title else names[i]
                ax.set_title(label, fontsize=9)

        fig.tight_layout()
        return fig


class CaptumTimeSeriesVisualiser(BaseVisualiser):
    """
    Visualise time-series attributions via
    ``captum.attr.visualization.visualize_timeseries_attr``.

    Compatible with ALL Captum attribution algorithms.
    """

    supported_scopes: ClassVar[frozenset[ExplanationScope]] = frozenset({ExplanationScope.LOCAL})
    supported_output_spaces: ClassVar[frozenset[ExplanationOutputSpace]] = frozenset(
        {ExplanationOutputSpace.INPUT_FEATURES}
    )
    supported_method_families: ClassVar[frozenset[MethodFamily]] = _CAPTUM_SEQUENCE_METHOD_FAMILIES

    def validate_explanation(
        self,
        explanation: object,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None,
    ) -> None:
        super().validate_explanation(explanation, attributions, inputs)
        if _input_kind(explanation) not in {"time_series", "timeseries"}:
            self._raise_incompatibility(
                "input metadata",
                _input_kind(explanation),
                "time_series",
            )

    def __init__(
        self,
        method: str = "overlay_individual",
        sign: str = "absolute_value",
    ):
        """
        Args:
            method: One of ``"overlay_individual"``, ``"overlay_combined"``,
                    ``"colored_graph"``.  Default: ``"overlay_individual"``.
            sign:   One of ``"positive"``, ``"negative"``, ``"absolute_value"``,
                    ``"all"``.  Default: ``"absolute_value"``.
        """
        self.method = method
        self.sign = sign

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
            attributions: ``(N, C)`` numpy array / tensor  (channels-last).
                          If a batch of shape ``(B, N, C)`` is given, the first
                          sample is used.
            inputs:       Matching ``(N, C)`` time-series data (optional).
            context:      Standard RAITAP metadata (not used by this visualiser).
            **kwargs:     Forwarded to ``visualize_timeseries_attr``.

        Returns:
            Matplotlib Figure.
        """
        try:
            from captum.attr import visualization as viz
        except ImportError as e:
            raise ImportError(
                "Captum visualiser is enabled but captum is not installed. "
                "Install it with `uv sync --extra captum`."
            ) from e

        attr = _to_numpy(attributions)
        # If batch dimension present, take first sample
        if attr.ndim == 3:
            attr = attr[0]

        data = None
        if inputs is not None:
            data = _to_numpy(inputs)
            if data.ndim == 3:
                data = data[0]

        if data is None:
            raise ValueError(
                "CaptumTimeSeriesVisualiser requires `inputs` (the original time-series "
                "data) to be passed alongside attributions."
            )

        fig, _ = viz.visualize_timeseries_attr(
            attr=attr,
            data=data,
            method=self.method,
            sign=self.sign,
            use_pyplot=False,
            **kwargs,
        )
        return fig


class CaptumTextVisualiser(BaseVisualiser):
    """
    Visualise per-token text attributions as a horizontal bar chart.

    This is a lightweight matplotlib-based implementation since Captum's native
    text visualisation renders HTML (not a Matplotlib Figure).

    Compatible with ALL Captum attribution algorithms on text/sequence inputs.

    Note: ``attributions`` should be a 1-D array of per-token scores for a
    single input. Pass ``token_labels`` via kwargs for readable output.
    """

    supported_scopes: ClassVar[frozenset[ExplanationScope]] = frozenset({ExplanationScope.LOCAL})
    supported_output_spaces: ClassVar[frozenset[ExplanationOutputSpace]] = frozenset(
        {ExplanationOutputSpace.TOKEN_SEQUENCE}
    )
    supported_method_families: ClassVar[frozenset[MethodFamily]] = _CAPTUM_SEQUENCE_METHOD_FAMILIES

    def validate_explanation(
        self,
        explanation: object,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None,
    ) -> None:
        super().validate_explanation(explanation, attributions, inputs)
        kind = _input_kind(explanation)
        has_token_metadata = _input_layout(explanation) in {"TOKENS", "TOKEN_SEQUENCE"}
        if kind and kind != "text":
            self._raise_incompatibility(
                "input metadata",
                kind,
                "text/token sequence",
            )
        if kind != "text" and not has_token_metadata:
            self._raise_incompatibility(
                "input metadata",
                kind,
                "text/token sequence",
            )

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        token_labels: list[str] | None = None,
        **kwargs,
    ) -> Figure:
        """
        Args:
            attributions:  1-D attribution scores (one per token).
            inputs:        Ignored.
            token_labels:  List of token strings (optional).
            **kwargs:      Ignored (for API consistency).

        Returns:
            Matplotlib Figure with a horizontal bar chart of token importance.
        """
        attr = _to_numpy(attributions).ravel()
        n = len(attr)
        labels = token_labels if token_labels else [f"tok_{i}" for i in range(n)]

        fig, ax = plt.subplots(figsize=(8, max(4, n * 0.35)))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in attr]
        y = np.arange(n)
        ax.barh(y, attr, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Attribution")
        ax.set_title("Token Attribution")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig
