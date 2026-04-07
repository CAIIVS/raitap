"""Base class for visualization (modality-agnostic interface)"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure


class BaseVisualiser(ABC):
    """
    Abstract base class for all visualiser implementations.

    visualisers ONLY handle visualization - they do NOT compute attributions.
    Attribution computation is handled by separate Explainer classes.

    Class attribute
    ---------------
    compatible_algorithms:
        Frozenset of algorithm names this visualiser supports.
        An empty frozenset (the default) means *compatible with all algorithms*.
    """

    compatible_algorithms: frozenset[str] = frozenset()

    @abstractmethod
    def visualise(
        self, attributions: torch.Tensor, inputs: torch.Tensor | None = None, **kwargs: Any
    ) -> Figure:
        """
        Create visualization from attributions.

        Args:
            attributions: Attribution values (numpy array or tensor)
            inputs: Original inputs for overlay (optional)
            **kwargs: visualiser-specific arguments

        Returns:
            Matplotlib figure
        """
        pass

    def save(
        self,
        attributions: torch.Tensor,
        output_path: str | Path,
        inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Save visualization to file.

        Default implementation using visualise(). Subclasses can override for efficiency.

        Args:
            attributions: Attribution values
            output_path: Path to save image
            inputs: Original inputs for overlay (optional)
            **kwargs: visualiser-specific arguments
        """
        fig = self.visualise(attributions, inputs, **kwargs)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
