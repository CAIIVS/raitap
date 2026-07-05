"""Captum-native visualisers (wrapping captum.attr.visualization)"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from raitap import raitap_log
from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationScope,
    MethodFamily,
    TensorLayout,
)
from raitap.transparency.visualisers.image_rendering import IMAGE_RENDERER_REGISTRY
from raitap.transparency.visualisers.registration import transparency_visualiser

from .base_visualiser import BaseVisualiser

if TYPE_CHECKING:
    import torch
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
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


# Captum renders the heat overlay through ``_normalize_image_attr`` (outlier-
# clipped percentile scaling), so the colorbar shows relative intensity, not raw
# attribution units: 0..1 for single-sign maps, -1..1 for ``sign="all"``.
_ATTR_COLORBAR_LABEL = "Normalised attribution"


def _last_mappable(ax: Any) -> Any:
    """Return the last Matplotlib mappable drawn on an axis, if any."""
    if getattr(ax, "images", None):
        return ax.images[-1]
    if getattr(ax, "collections", None):
        return ax.collections[-1]
    return None


def _captum_normalisation_degenerate(
    attr: np.ndarray, sign: str, outlier_perc: float = 2.0
) -> bool:
    """Whether Captum's ``_normalize_attr`` would hit ``scale_factor == 0``.

    An all-zero (or sign-empty, e.g. a non-positive map under ``sign="positive"``)
    slice is a *valid* explainer output — it means the method found no attribution
    of that sign — but Captum's normaliser asserts ``Cannot normalize by scale
    factor = 0`` on it rather than rendering it. We mirror Captum's threshold maths
    here (channel-summed reduction, sign filter, cumulative-sum percentile) so the
    slice can be rendered flat instead of crashing the figure.
    """
    combined = np.sum(attr, axis=2) if attr.ndim == 3 else attr
    if sign == "positive":
        values = (combined > 0) * combined
    elif sign == "negative":
        values = np.abs((combined < 0) * combined)
    else:  # "all" / "absolute_value"
        values = np.abs(combined)
    sorted_vals = np.sort(values.flatten())
    if sorted_vals.size == 0:
        return True
    cum_sums = np.cumsum(sorted_vals)
    total = float(cum_sums[-1])
    if total == 0:
        return True
    # NB: this is *not* equivalent to the ``total == 0`` check above and must not be
    # simplified to it. For the default ``outlier_perc=2`` the percentile lands deep
    # in the non-zero tail, so the two agree — but Captum scales by the
    # ``(100 - outlier_perc)``-percentile value, which is itself 0 when enough mass
    # sits at zero (e.g. ``outlier_perc=100`` → 0th percentile → the min, which is 0
    # whenever any pixel is zero). In those cases ``total > 0`` yet Captum still hits
    # ``scale_factor == 0``; replicating its threshold here is what catches them.
    # ``np.where`` is non-empty for any valid ``outlier_perc`` in [0, 100] because the
    # final cum-sum (== total) always satisfies ``total >= total * 0.01 * percentile``.
    threshold_id = int(np.where(cum_sums >= total * 0.01 * (100.0 - outlier_perc))[0][0])
    return bool(sorted_vals[threshold_id] == 0)


def _degenerate_note(sign: str) -> str:
    """Sign-aware annotation for a valid all-zero attribution map."""
    if sign == "positive":
        return "no positive attribution"
    if sign == "negative":
        return "no negative attribution"
    return "no attribution"


def _render_flat_attribution(ax: Any, sign: str, base_title: str | None) -> None:
    """Render a valid all-zero attribution as a flat map with an explanatory note.

    The map is zero everywhere, so we show a uniform field (faithful to the data)
    and annotate it so a reader reads it as a finding — the method assigned no
    attribution of this sign — rather than a rendering glitch.
    """
    ax.imshow(np.zeros((1, 1)), cmap="gray", vmin=-1.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    note = _degenerate_note(sign)
    ax.set_title(f"{base_title}\n({note})" if base_title else f"({note})")


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
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _input_metadata(explanation: object) -> dict[str, object]:
    semantics = getattr(explanation, "semantics", None)
    input_spec = getattr(semantics, "input_spec", None)
    metadata = getattr(input_spec, "metadata", None)
    if metadata is None:
        return {}
    return {str(k): v for k, v in dict(metadata).items()}


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


def _has_time_series_layout(explanation: object, attributions: object) -> bool:
    valid_layouts = {"B,T,C", "(B,T,C)", "T,C", "(T,C)"}
    layouts = {_input_layout(explanation), _output_layout(explanation)}
    if any(layout and layout not in valid_layouts for layout in layouts):
        return False
    shape = _output_shape(explanation, attributions)
    return shape is not None and len(shape) in {2, 3}


def _has_token_layout(explanation: object, attributions: object) -> bool:
    valid_layouts = {TensorLayout.TOKEN_SEQUENCE.value, TensorLayout.TOKEN_SEQUENCE.name}
    layouts = {_input_layout(explanation), _output_layout(explanation)}
    if any(layout and layout not in valid_layouts for layout in layouts):
        return False
    shape = _output_shape(explanation, attributions)
    # 1-D ``(T,)`` for a single token sequence, or 2-D ``(B, T)`` for a batch of
    # per-token scores (one row per sample) — both are token-sequence layouts.
    return shape is not None and len(shape) in {1, 2}


@transparency_visualiser(
    registry_name="captum_image",
    supported_scopes={ExplanationScope.LOCAL},
    supported_output_spaces={
        ExplanationOutputSpace.INPUT_FEATURES,
        ExplanationOutputSpace.IMAGE_SPATIAL_MAP,
    },
    supported_method_families={
        MethodFamily.GRADIENT,
        MethodFamily.PERTURBATION,
        MethodFamily.SHAPLEY,
        MethodFamily.CAM,
        MethodFamily.MODEL_AGNOSTIC,
        MethodFamily.SURROGATE,
    },
    embeds_original_input=True,
)
class CaptumImageVisualiser(BaseVisualiser):
    """
    Visualise image attributions using ``captum.attr.visualization.visualize_image_attr``.

    Wraps the Captum native function so the output is a Matplotlib Figure
    that can be saved or returned by ``explain()``.

    Compatible with ALL Captum attribution algorithms.
    """

    def renders_attribution_only_when_original_hidden(self) -> bool:
        return self.method != "masked_image"

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
            include_original_image and origs is not None and self.method != "original_image"
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
                # bilinear-upsample so heat overlays align with the original
                # image extent in matplotlib (also required for masked modes).
                if self.method != "original_image":
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

                if _captum_normalisation_degenerate(attr_i, "all"):
                    # ``original_image`` only displays the input, but Captum
                    # normalises attr first and would assert on this slice.
                    original_ax.imshow(orig_i)
                    original_ax.set_xticks([])
                    original_ax.set_yticks([])
                    if original_title is not None:
                        original_ax.set_title(original_title)
                else:
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

                CaptumNativeRenderer().draw(
                    attr_ax,
                    attr_i,
                    orig_i,
                    sign=self.sign,
                    method=self.method,
                    title=attr_title,
                    **{k: v for k, v in kwargs.items() if k != "title"},
                )
                if self.show_colorbar and colorbar_ax is not None:
                    mappable = _last_mappable(attr_ax)
                    if mappable is None:
                        colorbar_ax.set_visible(False)
                    else:
                        fig.colorbar(mappable, cax=colorbar_ax).set_label(_ATTR_COLORBAR_LABEL)
                continue

            ax = axes[i]
            viz_kwargs = dict(kwargs)
            if self.title is not None and "title" not in viz_kwargs:
                viz_kwargs["title"] = self.title

            mappable = CaptumNativeRenderer().draw(
                ax,
                attr_i,
                orig_i,
                sign=self.sign,
                method=self.method,
                show_colorbar=self.show_colorbar,
                title=viz_kwargs.get("title"),
                **{k: v for k, v in viz_kwargs.items() if k != "title"},
            )
            if self.show_colorbar and mappable is not None and mappable.colorbar is not None:
                mappable.colorbar.set_label(_ATTR_COLORBAR_LABEL)
            if show_sample_names and i < len(names):
                base_title = ax.get_title().strip()
                label = f"{base_title}: {names[i]}" if base_title else names[i]
                ax.set_title(label, fontsize=9)

        fig.tight_layout()
        return fig


@transparency_visualiser(
    registry_name="captum_time_series",
    supported_scopes={ExplanationScope.LOCAL},
    supported_output_spaces={ExplanationOutputSpace.INPUT_FEATURES},
    supported_method_families=_CAPTUM_SEQUENCE_METHOD_FAMILIES,
)
class CaptumTimeSeriesVisualiser(BaseVisualiser):
    """
    Visualise time-series attributions via
    ``captum.attr.visualization.visualize_timeseries_attr``.

    Compatible with ALL Captum attribution algorithms.
    """

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
        if not _has_time_series_layout(explanation, attributions):
            self._raise_incompatibility(
                "time-series layout",
                _input_layout(explanation)
                or _output_layout(explanation)
                or str(_output_shape(explanation, attributions)),
                "(B, T, C) or (T, C) attributions",
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


@transparency_visualiser(
    registry_name="captum_text",
    supported_scopes={ExplanationScope.LOCAL},
    supported_output_spaces={ExplanationOutputSpace.TOKEN_SEQUENCE},
    supported_method_families=_CAPTUM_SEQUENCE_METHOD_FAMILIES,
)
class CaptumTextVisualiser(BaseVisualiser):
    """
    Visualise per-token text attributions as a horizontal bar chart.

    This is a lightweight matplotlib-based implementation since Captum's native
    text visualisation renders HTML (not a Matplotlib Figure).

    Compatible with ALL Captum attribution algorithms on text/sequence inputs.

    Note: ``attributions`` should be a 1-D array of per-token scores for a
    single input. Pass ``token_labels`` via kwargs for readable output.
    """

    def validate_explanation(
        self,
        explanation: object,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None,
    ) -> None:
        super().validate_explanation(explanation, attributions, inputs)
        kind = _input_kind(explanation)
        if kind and kind != "text":
            self._raise_incompatibility(
                "input metadata",
                kind,
                "text/token sequence",
            )
        if not _has_token_layout(explanation, attributions):
            self._raise_incompatibility(
                "token-sequence layout",
                _input_layout(explanation)
                or _output_layout(explanation)
                or str(_output_shape(explanation, attributions)),
                "1-D token attributions with TOKENS/TOKEN_SEQUENCE layout",
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
            A 2-D ``(B, T)`` batch renders one bar panel per sample (row).
        """
        attr = _to_numpy(attributions)
        if attr.ndim > 2:
            # A token attribution is 1-D ``(T,)`` or 2-D ``(B, T)``. A higher-rank
            # tensor (e.g. an un-reduced ``(B, T, H)`` layer output) would be
            # silently flattened into the wrong token grid; reject it instead.
            raise ValueError(
                f"CaptumTextVisualiser expects a 1-D (T,) or 2-D (B, T) token attribution, "
                f"got shape {tuple(attr.shape)}. Reduce the embedding dimension first."
            )
        # Normalise to a batch of rows: 1-D ``(T,)`` is a single sample.
        rows = attr.reshape(1, -1) if attr.ndim == 1 else attr.reshape(attr.shape[0], -1)
        n_samples, n_tokens = rows.shape
        if token_labels and n_samples > 1:
            # One ``list[str]`` reused across samples would mislabel every token
            # after the first sample. Per-sample labels are follow-on work (#99).
            raise ValueError(
                "token_labels cannot be applied to a batched (B, T) attribution: a single "
                "label list would be reused across all samples and mislabel tokens. Omit "
                "token_labels for batched text (per-sample labels are tracked in GH #99)."
            )

        fig, axes = plt.subplots(
            n_samples,
            1,
            figsize=(8, max(4, n_tokens * 0.35) * n_samples),
            squeeze=False,
        )
        y = np.arange(n_tokens)
        labels = token_labels if token_labels else [f"tok_{i}" for i in range(n_tokens)]
        for row_index in range(n_samples):
            ax = axes[row_index][0]
            values = rows[row_index]
            colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]
            ax.barh(y, values, color=colors)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Attribution")
            title = (
                "Token Attribution" if n_samples == 1 else f"Token Attribution (sample {row_index})"
            )
            ax.set_title(title)
            ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig


class CaptumNativeRenderer:
    """Captum-native recipe via ``captum.attr.visualization.visualize_image_attr``.

    ``attr`` channels-last (H,W,C); ``image`` normalised (H,W,C). ``method``
    (default ``"blended_heat_map"``) and other Captum styling are forwarded via
    ``**style``. Returns the drawn mappable, or ``None`` when the slice is a valid
    all-zero map (rendered flat instead of crashing — see #206/#207).
    """

    honours_method = True
    honoured_signs = frozenset({"all", "positive", "negative", "absolute_value"})

    def draw(
        self,
        ax: Axes,
        attr: np.ndarray,
        image: np.ndarray | None,
        *,
        sign: str = "all",
        **style: Any,
    ) -> ScalarMappable | None:
        from captum.attr import visualization as viz
        from matplotlib.figure import Figure

        method = style.pop("method", "blended_heat_map")
        title = style.pop("title", None)
        show_colorbar = bool(style.pop("show_colorbar", False))
        outlier_perc = float(style.get("outlier_perc", 2.0))
        if _captum_normalisation_degenerate(np.asarray(attr), sign, outlier_perc):
            _render_flat_attribution(ax, sign, title)
            return None
        # ``ax.figure`` is typed ``Figure | SubFigure``; visualisers always pass a
        # top-level ``Figure``'s axes, and visualize_image_attr's stub requires ``Figure``.
        fig = ax.figure
        assert isinstance(fig, Figure)
        viz.visualize_image_attr(
            attr,
            image,
            method=method,
            sign=sign,
            show_colorbar=show_colorbar,
            plt_fig_axis=(fig, ax),
            use_pyplot=False,
            **({"title": title} if title is not None else {}),
            **style,
        )
        return _last_mappable(ax)


IMAGE_RENDERER_REGISTRY["captum"] = CaptumNativeRenderer()
