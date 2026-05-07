from __future__ import annotations


class AssessorBackendIncompatibilityError(Exception):
    """Raised when an assessor's algorithm is not supported by the selected backend."""

    def __init__(
        self,
        assessor: str,
        backend: str,
        algorithm: str,
        reason: str,
    ) -> None:
        self.assessor = assessor
        self.backend = backend
        self.algorithm = algorithm
        self.reason = reason
        super().__init__(
            f"Assessor {assessor!r} with algorithm {algorithm!r} is not compatible with "
            f"backend {backend!r}.\nReason: {reason}"
        )


class MethodKindVisualiserIncompatibilityError(Exception):
    """Raised when a visualiser does not support the assessor's method kind."""

    def __init__(
        self,
        *,
        assessor_target: str,
        visualiser: str,
        assessor_method_kind: str,
        supported_method_kinds: list[str],
    ) -> None:
        self.assessor_target = assessor_target
        self.visualiser = visualiser
        self.assessor_method_kind = assessor_method_kind
        self.supported_method_kinds = supported_method_kinds
        supported = ", ".join(supported_method_kinds) if supported_method_kinds else "none"
        super().__init__(
            f"Visualiser {visualiser!r} does not support assessor method kind "
            f"{assessor_method_kind!r} (from {assessor_target}). "
            f"That visualiser's supported_method_kinds are: {supported}."
        )


class RobustnessVisualiserIncompatibilityError(Exception):
    """Raised when a visualiser is not compatible with the assessor's algorithm."""

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


class MissingTargetsError(Exception):
    """Raised when a robustness assessment is requested but no labels are available."""

    def __init__(self, assessor_name: str) -> None:
        self.assessor_name = assessor_name
        super().__init__(
            f"Robustness assessor {assessor_name!r} requires per-sample target labels, "
            "but the data pipeline produced no labels. Configure ``data.labels.source`` "
            "(or otherwise expose ``data.labels``) before enabling robustness."
        )
