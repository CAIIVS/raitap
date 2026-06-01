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
    supported_payload_kinds=frozenset({ExplanationPayloadKind.ATTRIBUTIONS}),
    supported_output_spaces=frozenset({ExplanationOutputSpace.DETECTION_BOXES}),
    supported_scopes=frozenset({ExplanationScope.LOCAL}),
    supported_method_families=frozenset(
        {
            MethodFamily.GRADIENT,
            MethodFamily.PERTURBATION,
            MethodFamily.SHAPLEY,
            MethodFamily.CAM,
            MethodFamily.MODEL_AGNOSTIC,
            MethodFamily.SURROGATE,
        }
    ),
    supported_tasks=frozenset({TaskKind.detection}),
    embeds_original_input=True,
)
class DetectionImageVisualiser(BaseVisualiser):
    """Render one fig per box: original image + bbox rectangle + heatmap.

    Reads the per-box metadata from ``context.detection_box`` (populated by
    the detection explain phase). The visualiser does not know which box k
    it is rendering until the context is passed in.
    """

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

        attr_arr = attr.numpy() if hasattr(attr, "numpy") else np.asarray(attr)
        if attr_arr.ndim == 3:
            attr_2d = np.abs(attr_arr).sum(axis=0)
        elif attr_arr.ndim == 2:
            attr_2d = attr_arr
        else:
            raise ValueError(
                f"DetectionImageVisualiser expected (C, H, W) or (H, W) attribution; "
                f"got shape {attr_arr.shape!r}."
            )
        # Layer methods (e.g. LayerGradCam) yield low-res spatial maps; bilinear-
        # upsample to the image so the heat overlay spans the full frame instead
        # of a top-left corner patch. Mirrors the classification path
        # (captum_visualisers._resize_attr_to_hw). Issue #203.
        target_hw = img_hwc.shape[:2]
        if attr_2d.shape != target_hw:
            attr_2d = _resize_attr_to_hw(attr_2d, target_hw)
        attr_max = float(np.max(np.abs(attr_2d))) if attr_2d.size else 0.0
        if attr_max > 0:
            attr_2d = attr_2d / attr_max

        fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
        ax.imshow(img_hwc, interpolation="nearest")
        heat = ax.imshow(attr_2d, cmap="seismic", alpha=0.45, vmin=-1.0, vmax=1.0)

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
        # Keys so the overlay is legible standalone: colorbar maps red/blue to
        # signed attribution; the legend names the green reference box. The legend
        # sits outside the axes so it never covers the image.
        bar = fig.colorbar(heat, ax=ax)
        bar.set_label("attribution (normalised)")
        fig.legend(handles=[rect], loc="outside upper right", fontsize=8, framealpha=0.9)

        label_str = box.label_name if box.label_name else f"class {box.label_index}"
        ax.set_title(
            f"{label_str}: {box.score:.2f}    [box {box.display_index} (raw {box.raw_index})]"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, img_hwc.shape[1])
        ax.set_ylim(img_hwc.shape[0], 0)

        return fig
