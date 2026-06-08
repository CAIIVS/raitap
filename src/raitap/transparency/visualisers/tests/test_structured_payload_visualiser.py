"""Tests for the POC structured-payload summary visualiser (#101)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from raitap.transparency.contracts import (
    StructuredPayload,
    StructuredPayloadKind,
    VisualisationContext,
)
from raitap.transparency.visualisers import StructuredPayloadSummaryVisualiser


def _context(payloads: tuple[StructuredPayload, ...]) -> VisualisationContext:
    return VisualisationContext(
        algorithm="IntegratedGradients",
        sample_names=None,
        show_sample_names=False,
        structured_payloads=payloads,
    )


def test_declares_supported_structured_kinds() -> None:
    declared = StructuredPayloadSummaryVisualiser.supported_structured_payload_kinds
    assert StructuredPayloadKind.CONVERGENCE_DELTA in declared
    assert StructuredPayloadKind.BASE_VALUE in declared


def test_renders_per_sample_payload_from_context() -> None:
    payload = StructuredPayload(
        "convergence_delta",
        StructuredPayloadKind.CONVERGENCE_DELTA,
        torch.tensor([0.1, 0.2, 0.3]),
    )
    vis = StructuredPayloadSummaryVisualiser()
    fig = vis.visualise(torch.zeros(3, 4), inputs=None, context=_context((payload,)))

    assert fig.axes  # at least one axis was drawn
    # The matching payload contributes bars equal to its per-sample length.
    assert any(len(ax.patches) == 3 for ax in fig.axes)
    plt.close(fig)
