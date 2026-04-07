from __future__ import annotations


class VisualiserIncompatibilityError(Exception):
    """Raised when a visualiser is not compatible with the chosen explainer algorithm."""

    def __init__(
        self,
        framework: str,
        visualiser: str,
        algorithm: str,
        compatible_algorithms: list[str],
    ) -> None:
        self.framework = framework
        self.visualiser = visualiser
        self.algorithm = algorithm
        self.compatible_algorithms = compatible_algorithms
        super().__init__(
            f"Visualiser {visualiser!r} is not compatible with "
            f"{framework}/{algorithm}.\n"
            f"Compatible algorithms: {', '.join(compatible_algorithms) or 'none'}."
        )


class ExplainerBackendIncompatibilityError(Exception):
    """Raised when an explainer algorithm is not supported by the selected backend."""

    def __init__(
        self,
        explainer: str,
        backend: str,
        algorithm: str,
        compatible_algorithms: list[str],
    ) -> None:
        self.explainer = explainer
        self.backend = backend
        self.algorithm = algorithm
        self.compatible_algorithms = compatible_algorithms
        super().__init__(
            f"Explainer {explainer!r} with algorithm {algorithm!r} is not compatible with "
            f"backend {backend!r}.\n"
            f"Compatible algorithms: {', '.join(compatible_algorithms) or 'none'}."
        )
