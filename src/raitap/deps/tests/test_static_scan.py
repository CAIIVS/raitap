"""Guard the AST-based ``extra=`` scanner against drift from runtime registration.

:func:`raitap.deps.static_scan.scan_adapter_extras` walks the raitap source
tree without importing modules so the deps bootstrap can resolve adapter
extras in partial-extras venvs. This test asserts the runtime
``ADAPTER_EXTRAS`` populated by the family decorators (``_register_core``) is a
*subset* of what the scanner finds — drift in either direction (a missing
``extra=`` kwarg on a new adapter, or scanner missing a new declaration
syntax) shows up here.

Tolerant of partial-extras venvs: when ``register_zen_groups`` can't load
every adapter module (because torch / Captum / ... is missing), the
runtime dict is partial, but it should still be a subset of the scanner's
output — that is the whole point of the scanner.
"""

from __future__ import annotations

from raitap._adapters import ADAPTER_EXTRAS
from raitap.deps.static_scan import scan_adapter_extras


def test_runtime_extras_subset_of_static_scan() -> None:
    # Best-effort registration — wrapped in try/except so a partial-extras
    # venv where some adapter module fails to import does not abort the test.
    try:
        from raitap.configs.zen import register_zen_groups

        register_zen_groups()
    except Exception:  # pragma: no cover — depends on venv state
        pass

    scanned = scan_adapter_extras()
    for class_name, extra in ADAPTER_EXTRAS.items():
        assert scanned.get(class_name) == extra, (
            f"AST scanner missed {class_name!r} (or disagreed on its extra)."
        )


def test_static_scan_finds_canonical_set() -> None:
    """Scanner must always find the full set, regardless of import state."""
    scanned = scan_adapter_extras()
    assert set(scanned) >= {
        "CaptumExplainer",
        "ShapExplainer",
        "TorchattacksAssessor",
        "FoolboxAssessor",
        "MarabouAssessor",
        "HTMLReporter",
        "PDFReporter",
        "MLFlowTracker",
        "BinaryClassificationMetrics",
        "MulticlassClassificationMetrics",
        "MultilabelClassificationMetrics",
        "DetectionMetrics",
    }, f"Scanner missed canonical adapters. Found: {sorted(scanned)!r}."
