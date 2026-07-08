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

from raitap._adapters import _TARGET_FQN, ADAPTER_EXTRAS
from raitap.deps.static_scan import (
    scan_adapter_extras,
    scan_adapter_registry,
    scan_backend_extras,
)
from raitap.types import ResolvedHardware


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


def test_runtime_target_fqn_subset_of_scan_adapter_registry() -> None:
    # Same drift guard as ``test_runtime_extras_subset_of_static_scan``, but for
    # the ``(group, registry_name) -> extra`` map ``_extra_for_use`` (#301)
    # falls back on when ``_TARGET_FQN`` hasn't been populated yet (adapter
    # module not imported in a partial-extras venv).
    try:
        from raitap.configs.zen import register_zen_groups

        register_zen_groups()
    except Exception:  # pragma: no cover — depends on venv state
        pass

    # Only the groups deps inference actually walks — ``data/inputs`` and
    # ``data/labels`` use a different family decorator, not covered by
    # :data:`raitap.deps.static_scan._DECORATOR_GROUP`, and are irrelevant to
    # :func:`raitap.deps.inference.infer_extras`.
    relevant_groups = {"transparency", "robustness", "metrics", "reporting", "tracking"}
    scanned = scan_adapter_registry()
    for group, names in _TARGET_FQN.items():
        if group not in relevant_groups:
            continue
        for registry_name, fqn in names.items():
            # Sibling tests register stub adapters (e.g. ``_stub_metric``) into
            # the runtime registry; the AST scanner intentionally skips ``tests``
            # dirs, so those never appear in ``scanned``. Guard only in-tree
            # adapters — the drift this test protects against.
            if ".tests." in fqn:
                continue
            assert registry_name in scanned.get(group, {}), (
                f"AST scanner missed {group}/{registry_name!r}."
            )
    # "_unscoped" mixes evaluators (scanned) with visualisers (not scanned —
    # the deps walker never reads a "visualisers" list's ``use``, see
    # ``test_visualisers_do_not_contribute`` in test_inference.py); only guard
    # the evaluator registered there.
    if "quantus" in _TARGET_FQN.get("_unscoped", {}):
        assert scanned["_unscoped"]["quantus"] == "quantus"


def test_scan_adapter_registry_finds_canonical_set() -> None:
    """Scanner must always find the full set, regardless of import state."""
    scanned = scan_adapter_registry()
    assert scanned["transparency"] == {"captum": "captum", "shap": "shap"}
    assert scanned["reporting"] == {"html": "html", "pdf": "pdf"}
    assert scanned["tracking"] == {"mlflow": "mlflow"}
    assert scanned["_unscoped"]["quantus"] == "quantus"
    assert {"auto_lirpa", "foolbox", "imagecorruptions", "marabou", "torchattacks"} <= set(
        scanned["robustness"]
    )
    assert {
        "binary_classification",
        "multiclass_classification",
        "multilabel_classification",
        "detection",
    } <= set(scanned["metrics"])


def test_scan_backend_extras_maps_extension_to_extra_and_hardware() -> None:
    """Import-free scan of backend ``@register`` decorators: extension ->
    (extra, supported_hardware). Guards drift between the decorators and the
    deps inference path."""
    scanned = scan_backend_extras()
    full = frozenset(ResolvedHardware)
    # Accelerator runtimes split per hardware.
    assert scanned[".pt"] == ("torch", full)
    assert scanned[".pth"] == ("torch", full)
    assert scanned[".onnx"] == ("onnx", full)
    # Single-wheel runtime: bare extra, no hardware split.
    assert scanned[".ubj"] == ("xgboost", frozenset())
