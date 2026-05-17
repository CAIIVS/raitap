"""Side-by-side clean / perturbed / diff renderer for empirical attacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np

from raitap.robustness.visualisers.registration import register_robustness_visualiser
from raitap.transparency.contracts import InputKind
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch
else:
    torch = lazy_import("torch")

from ...contracts import MethodKind
from ..base_visualiser import BaseRobustnessVisualiser

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ...contracts import RobustnessVisualisationContext
    from ...results import RobustnessResult


@register_robustness_visualiser(registry_name="image_pair")
class ImagePairVisualiser(BaseRobustnessVisualiser):
    """Render N rows by 3 columns: clean, perturbed, signed perturbation."""

    supported_method_kinds: ClassVar[frozenset[MethodKind]] = frozenset(
        {MethodKind.EMPIRICAL_ATTACK}
    )
    embeds_clean_input: ClassVar[bool] = True
    embeds_perturbation_map: ClassVar[bool] = True

    def __init__(
        self,
        *,
        max_samples: int = 4,
        cmap: str = "RdBu_r",
        diff_scale: float | None = None,
    ) -> None:
        self.max_samples = max(int(max_samples), 1)
        self.cmap = cmap
        self.diff_scale = diff_scale

    def visualise(
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
        **kwargs: Any,
    ) -> Figure:
        include_clean_input = bool(kwargs.pop("include_clean_input", True))
        include_perturbation_map = bool(kwargs.pop("include_perturbation_map", True))
        del kwargs
        if result.perturbed_inputs is None:
            raise ValueError(
                "ImagePairVisualiser requires perturbed_inputs to be present on the result."
            )
        _require_image_modality(result, type(self).__name__)
        clean = _to_image_batch(result.clean_inputs)
        perturbed = _to_image_batch(result.perturbed_inputs)
        n = min(int(clean.shape[0]), self.max_samples)
        columns = []
        if include_clean_input:
            columns.append("clean")
        columns.append("perturbed")
        if include_perturbation_map:
            columns.append("perturbation")
        fig, axes = plt.subplots(n, len(columns), figsize=(3 * len(columns), 3 * n), squeeze=False)

        scale = self.diff_scale
        diff_extreme = float(
            (perturbed[:n] - clean[:n]).abs().max().item() if scale is None else scale
        )
        diff_extreme = max(diff_extreme, 1e-6)

        # Use one display range across both clean and perturbed so the eye can
        # compare them; per-cell normalisation would shift the perturbed image's
        # scale relative to the clean image and visually exaggerate the attack.
        display_lo, display_hi = _shared_display_range(clean[:n], perturbed[:n])

        sample_names = context.sample_names or []

        for row in range(n):
            clean_image = _as_displayable(clean[row], lo=display_lo, hi=display_hi)
            perturbed_image = _as_displayable(perturbed[row], lo=display_lo, hi=display_hi)
            # imshow treats RGB-shaped arrays as literal colors and ignores cmap/vmin/vmax;
            # reduce to a 2D scalar map so the diverging cmap actually applies.
            diff = _signed_perturbation_heatmap(perturbed[row] - clean[row])

            target_label = int(result.targets[row].item())
            clean_pred = int(result.clean_predictions[row].item())
            adv_pred = (
                int(result.perturbed_predictions[row].item())
                if result.perturbed_predictions is not None
                else -1
            )
            sample_title = (
                sample_names[row] if context.show_sample_names and row < len(sample_names) else ""
            )

            for col, column in enumerate(columns):
                axis = axes[row][col]
                if column == "clean":
                    axis.imshow(clean_image, cmap="gray" if clean_image.ndim == 2 else None)
                    axis.set_title(_format_title("clean", clean_pred, target_label, sample_title))
                elif column == "perturbed":
                    axis.imshow(
                        perturbed_image,
                        cmap="gray" if perturbed_image.ndim == 2 else None,
                    )
                    axis.set_title(_format_title("perturbed", adv_pred, target_label, sample_title))
                else:
                    axis.imshow(diff, cmap=self.cmap, vmin=-diff_extreme, vmax=diff_extreme)
                    axis.set_title("perturbation")
                axis.set_axis_off()
        fig.suptitle(f"{context.algorithm} — clean vs perturbed", fontsize=12)
        fig.tight_layout()
        return fig


def _to_image_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    return tensor


def _as_displayable(
    sample: torch.Tensor,
    *,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Convert a (C, H, W) or (H, W) tensor to a HWC / HW float32 array in [0, 1].

    ``lo`` / ``hi`` are the display range used for normalisation; pass values
    derived once across the full clean+perturbed pair so both images share a
    scale (per-cell normalisation would visually exaggerate the attack).
    """
    array = sample.detach().cpu().to(torch.float32).numpy()
    if array.ndim == 3:
        if array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.shape[-1] == 1:
            array = array[..., 0]
    span = hi - lo
    if span < 1e-6:
        return np.zeros_like(array)
    return np.clip((array - lo) / span, 0.0, 1.0)


def _shared_display_range(clean: torch.Tensor, perturbed: torch.Tensor) -> tuple[float, float]:
    """Return a single ``(lo, hi)`` covering both batches.

    Tensors loaded by RAITAP are in ``[0, 1]`` (see :mod:`raitap.data`); we keep
    that as the canonical range for inputs that haven't drifted out of it (an
    attack can clip-overflow by an FGSM step). When something is clearly outside
    ``[0, 1]`` (e.g. uint8 ``[0, 255]`` or normalised model inputs), we fall back
    to the observed min/max so the visualiser still produces a readable image.
    """
    combined = torch.cat(
        [clean.detach().to(torch.float32).flatten(), perturbed.detach().to(torch.float32).flatten()]
    )
    sample_min = float(combined.min().item())
    sample_max = float(combined.max().item())
    if sample_min >= -1e-3 and sample_max <= 1.0 + 1e-3:
        return 0.0, 1.0
    return sample_min, sample_max


def _require_image_modality(result: RobustnessResult, visualiser_name: str) -> None:
    """Refuse to render a non-image robustness result through an image visualiser.

    ``BaseRobustnessVisualiser.validate_result`` only enforces ``method_kind``;
    image visualisers additionally need the result's input modality to be IMAGE
    so the (B, C, H, W) layout assumption holds. Without this guard a tabular
    or time-series result would silently feed garbage into ``imshow``.
    """
    spec = getattr(result.semantics, "input_spec", None)
    kind = getattr(spec, "kind", None)
    if kind is not None and kind != InputKind.IMAGE:
        raise ValueError(
            f"{visualiser_name} only renders image-modality results; "
            f"got input_spec.kind={kind.value!r}. Configure an image-aware "
            "visualiser or omit this entry from the assessor's visualisers list."
        )


def _signed_perturbation_heatmap(delta: torch.Tensor) -> np.ndarray:
    """Reduce a signed per-channel perturbation to one scalar per pixel.

    Picks the channel with the largest absolute deviation per pixel and keeps
    its signed value; for grayscale (C=1) just drops the channel dim. The
    returned 2D array can be passed to ``imshow(cmap=..., vmin=..., vmax=...)``
    so the diverging colormap actually applies (matplotlib treats 3-channel
    arrays as literal RGB and ignores ``cmap`` / ``vmin`` / ``vmax``).
    """
    array = delta.detach().cpu().to(torch.float32).numpy()
    if array.ndim == 3:
        if array.shape[0] in (1, 3):
            if array.shape[0] == 1:
                return array[0]
            channel_indices = np.abs(array).argmax(axis=0)
            return np.take_along_axis(array, channel_indices[None, ...], axis=0)[0]
        if array.shape[-1] in (1, 3):
            if array.shape[-1] == 1:
                return array[..., 0]
            channel_indices = np.abs(array).argmax(axis=-1)
            return np.take_along_axis(array, channel_indices[..., None], axis=-1)[..., 0]
    return array


def _format_title(label: str, predicted: int, target: int, sample_id: str) -> str:
    parts = [label, f"pred={predicted}", f"target={target}"]
    if sample_id:
        parts.insert(0, sample_id)
    return " | ".join(parts)
