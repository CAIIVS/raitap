"""POC visualiser rendering additive structured payloads (#101).

Renders per-sample structured diagnostics (convergence deltas, SHAP base values)
delivered via ``VisualisationContext.structured_payloads`` as one bar panel per
matching payload. This is the end-to-end proof of the structured-payload
consumption seam; richer structured visualisers (AIX360) build on the same seam
in #289.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from raitap.transparency.contracts import StructuredPayloadKind
from raitap.transparency.visualisers.registration import transparency_visualiser

from .base_visualiser import BaseVisualiser

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure

    from raitap.transparency.contracts import StructuredPayload, VisualisationContext


@transparency_visualiser(
    registry_name="structured_payload_summary",
    supported_structured_payload_kinds={
        StructuredPayloadKind.CONVERGENCE_DELTA,
        StructuredPayloadKind.BASE_VALUE,
    },
)
class StructuredPayloadSummaryVisualiser(BaseVisualiser):
    """Render per-sample structured payloads as bar panels."""

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        **kwargs: Any,
    ) -> Figure:
        del attributions, inputs, kwargs
        payloads = self._matching_payloads(context)
        if not payloads:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No structured payloads to display", ha="center", va="center")
            return fig

        fig, axes = plt.subplots(len(payloads), 1, figsize=(8, 2.5 * len(payloads)), squeeze=False)
        for ax, payload in zip(axes[:, 0], payloads, strict=True):
            values = self._to_1d_numpy(payload.data)
            readable_kind = payload.kind.value.replace("_", " ").capitalize()
            # ``name`` is usually the kind value; only disambiguate when it differs.
            title = (
                payload.name
                if payload.name == payload.kind.value
                else f"{payload.name} ({payload.kind.value})"
            )
            ax.bar(np.arange(values.shape[0]), values)
            ax.set_title(title)
            ax.set_xlabel("Sample")
            ax.set_ylabel(readable_kind)
            ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    def _matching_payloads(self, context: VisualisationContext | None) -> list[StructuredPayload]:
        payloads = getattr(context, "structured_payloads", ()) if context is not None else ()
        return [p for p in payloads if p.kind in self.supported_structured_payload_kinds]

    @staticmethod
    def _to_1d_numpy(data: Any) -> np.ndarray:
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        return np.asarray(data, dtype="float32").reshape(-1)
