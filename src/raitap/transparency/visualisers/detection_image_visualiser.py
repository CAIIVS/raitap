"""Detection image visualiser — one figure per detected box.

Renders the original image with the reference box outlined and the per-pixel
attribution heatmap overlaid. Compatible with all attribution method families
that produce per-pixel maps (gradient / perturbation / shapley / cam /
model_agnostic / surrogate). Issue #146 Phase 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from raitap import raitap_log
from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    MethodFamily,
)
from raitap.transparency.visualisers.registration import transparency_visualiser
from raitap.types import TaskKind

from .base_visualiser import BaseVisualiser
from .captum_visualisers import _resize_attr_to_hw

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure

    from raitap.transparency.contracts import VisualisationContext


@transparency_visualiser(
    registry_name="detection_image",
    supported_payload_kinds={ExplanationPayloadKind.ATTRIBUTIONS},
    supported_output_spaces={ExplanationOutputSpace.DETECTION_BOXES},
    supported_scopes={ExplanationScope.LOCAL},
    supported_method_families={
        MethodFamily.GRADIENT,
        MethodFamily.PERTURBATION,
        MethodFamily.SHAPLEY,
        MethodFamily.CAM,
        MethodFamily.MODEL_AGNOSTIC,
        MethodFamily.SURROGATE,
    },
    supported_tasks={TaskKind.detection},
    embeds_original_input=True,
)
class DetectionImageVisualiser(BaseVisualiser):
    """Render one fig per box: original image + bbox rectangle + heatmap.

    Reads the per-box metadata from ``context.detection_box`` (populated by
    the detection explain phase). The visualiser does not know which box k
    it is rendering until the context is passed in.
    """

    def __init__(
        self,
        *,
        method: str | None = None,
        sign: str | None = None,
        show_colorbar: bool | None = None,
        title: str | None = None,
    ) -> None:
        """Optional Captum render-style knobs, mirroring CaptumImageVisualiser.

        Defaults are ``None``-sentinels (not classification's concrete defaults)
        so unset fields reproduce the current default detection figure:

        - ``method``: ``None`` -> renderer default (``blended_heat_map``).
          Options: ``blended_heat_map`` | ``heat_map`` | ``masked_image`` |
          ``alpha_scaling``. Honoured only by the captum-sourced renderer.
        - ``sign``: ``None`` -> family-auto (``positive`` for CAM, else ``all``).
          Options: ``all`` | ``positive`` | ``negative`` | ``absolute_value``.
        - ``show_colorbar``: gates the attribution colorbar (renderer-agnostic).
          ``None``/``True`` -> shown; ``False`` -> suppressed.
        - ``title``: ``None`` -> report falls back to ``ClassName_index``.
          When set, surfaces as the report group name (also covers #225's
          detection half). Does NOT change the per-box matplotlib title.
        """
        self.method = method
        self.sign = sign
        self.show_colorbar = show_colorbar
        self.title = title

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        **kwargs: Any,
    ) -> Figure:
        del kwargs  # no per-call options yet
        if context is None or context.detection_box is None:
            raise ValueError(
                "DetectionImageVisualiser requires VisualisationContext.detection_box "
                "to be set; got None. This usually means the pipeline detection phase "
                "did not attach the per-box metadata to the ExplanationResult."
            )
        if inputs is None:
            raise ValueError("DetectionImageVisualiser requires inputs (original images).")

        box = context.detection_box

        img = inputs.detach().cpu()
        attr = attributions.detach().cpu()
        if img.ndim == 4:
            img = img[0]
        if attr.ndim == 4:
            attr = attr[0]
        if img.shape[0] == 3:
            img_hwc = img.permute(1, 2, 0).numpy()
        elif img.shape[0] == 1:
            img_hwc = img[0].numpy()
        else:
            img_hwc = img.permute(1, 2, 0).numpy()
        if img_hwc.dtype != np.uint8:
            img_hwc = np.clip(img_hwc, 0.0, 1.0)

        from raitap.transparency.visualisers.image_rendering import resolve_image_renderer

        renderer, auto_sign = resolve_image_renderer(
            context.source_library,
            frozenset(context.method_families),
        )
        final_sign = self.sign if self.sign is not None else auto_sign
        # ``show_colorbar`` is NOT forwarded to the renderer: it gates the
        # figure-level colorbar drawn below (renderer-agnostic), so forwarding it
        # to the captum renderer too would draw a second colorbar. Forward
        # ``method`` only.
        style = {key: value for key, value in (("method", self.method),) if value is not None}

        source = context.source_library or "the attribution's source library"
        if self.method is not None and not getattr(renderer, "honours_method", True):
            raitap_log.warn(
                "DetectionImageVisualiser method=%r is ignored: attributions from "
                "%s have no selectable overlay method. Set method only for "
                "captum-sourced detections, or leave it unset.",
                self.method,
                source,
            )
        if self.sign is not None and self.sign not in getattr(
            renderer, "honoured_signs", frozenset({self.sign})
        ):
            raitap_log.warn(
                "DetectionImageVisualiser sign=%r is ignored: attributions from %s "
                "do not support that sign and fall back to the default. Use a sign "
                "that source supports, or leave it unset.",
                self.sign,
                source,
            )

        fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
        attr_np = attr.numpy() if hasattr(attr, "numpy") else np.asarray(attr)
        if attr_np.ndim == 3:
            attr_np = np.transpose(attr_np, (1, 2, 0))
        # Layer methods (e.g. LayerGradCam) yield low-res spatial maps; bilinear-
        # upsample to the image so the heat overlay spans the full frame instead
        # of a top-left corner patch. Mirrors the classification path
        # (captum_visualisers._resize_attr_to_hw). Issue #203.
        if attr_np.shape[:2] != img_hwc.shape[:2]:
            attr_np = _resize_attr_to_hw(attr_np, img_hwc.shape[:2])
        heat = renderer.draw(ax, attr_np, img_hwc, sign=final_sign, **style)

        x1, y1, x2, y2 = box.xyxy
        rect = mpatches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            label="reference box",
        )
        ax.add_patch(rect)
        # Keys so the overlay is legible standalone: a colorbar for the attribution
        # heat (the renderer returns its mappable) and a legend naming the green
        # reference box. Both sit outside the axes so they never cover the image.
        # ``show_colorbar`` gates the colorbar: unset/None/True -> shown,
        # False -> suppressed.
        if heat is not None and self.show_colorbar is not False:
            fig.colorbar(heat, ax=ax).set_label("Attribution")
        fig.legend(handles=[rect], loc="outside upper right", fontsize=8, framealpha=0.9)

        # No per-box title: the predicted label, score, and ground-truth match
        # are shown on the thumbnail overlay (every box on the original image)
        # and in the report heading, so a title here would only duplicate them.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, img_hwc.shape[1])
        ax.set_ylim(img_hwc.shape[0], 0)

        return fig
