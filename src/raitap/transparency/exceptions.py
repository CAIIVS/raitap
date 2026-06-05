from __future__ import annotations


class VisualiserIncompatibilityError(Exception):
    """A visualiser cannot render an explanation: one declared axis is unsupported.

    Single typed failure for every transparency visualiser-compat gate — algorithm
    allowlist, payload kind, and the typed §4.3 semantic axes (output space, scope,
    method family). The predicate that decides incompatibility (subset / intersection
    / membership) stays at the call site; this only carries the structured result.

    Fields
    ------
    visualiser:
        Class name of the rejecting visualiser.
    axis:
        Human label of the contract axis that failed (``"algorithm"``,
        ``"payload kind"``, ``"output space"``, ``"scope"``, ``"method family"``,
        ``"input metadata"``, …).
    declared:
        The explanation/explainer value(s) the visualiser rejects.
    accepted:
        What the visualiser supports on that axis (``"any"`` when wildcard).
    """

    def __init__(self, *, visualiser: str, axis: str, declared: str, accepted: str) -> None:
        self.visualiser = visualiser
        self.axis = axis
        self.declared = declared
        self.accepted = accepted
        super().__init__(
            f"Visualiser {visualiser!r} is incompatible with this explanation: "
            f"{axis} {declared!r} is not supported; expected {accepted}."
        )
