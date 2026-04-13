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


class PayloadVisualiserIncompatibilityError(Exception):
    """Raised when a visualiser does not accept the explainer's output payload kind."""

    def __init__(
        self,
        *,
        explainer_target: str,
        visualiser: str,
        output_payload_kind: str,
        supported_payload_kinds: list[str],
    ) -> None:
        self.explainer_target = explainer_target
        self.visualiser = visualiser
        self.output_payload_kind = output_payload_kind
        self.supported_payload_kinds = supported_payload_kinds
        supported = ", ".join(supported_payload_kinds) if supported_payload_kinds else "none"
        super().__init__(
            f"Visualiser {visualiser!r} does not support explainer payload kind "
            f"{output_payload_kind!r} (from {explainer_target}). "
            f"That visualiser's supported_payload_kinds are: {supported}."
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
