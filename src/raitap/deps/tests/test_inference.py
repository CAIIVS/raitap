"""Tests for inference.infer_extras — operates on dict (post-compose) inputs."""

from __future__ import annotations

import pytest

from raitap.deps.inference import (
    UnknownAdapterTargetError,
    _extra_for_spec,
    backend_extra,
    infer_extras,
)
from raitap.types import ResolvedHardware


def test_torch_pt_cuda() -> None:
    cfg = {"model": {"source": "foo.pt"}, "hardware": "gpu"}
    extras, origins = infer_extras(cfg, hardware=ResolvedHardware.cuda)
    assert "torch-cuda" in extras
    assert "model.source" in origins["torch-cuda"]


def test_torch_pt_cpu() -> None:
    cfg = {"model": {"source": "foo.pt"}, "hardware": "cpu"}
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "torch-cpu" in extras


def test_torch_pt_intel() -> None:
    cfg = {"model": {"source": "foo.pt"}}
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.xpu)
    assert "torch-intel" in extras


def test_onnx_backend() -> None:
    cfg = {"model": {"source": "model.onnx"}}
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cuda)
    assert "onnx-cuda" in extras
    assert "torch-cuda" not in extras


def test_ubj_backend_maps_to_xgboost() -> None:
    cfg = {"model": {"source": "model.ubj"}}
    # XGBoost has no per-accelerator wheel split: same extra on every hardware,
    # and never a torch backend extra.
    for hardware in (ResolvedHardware.cpu, ResolvedHardware.cuda, ResolvedHardware.xpu):
        extras, _ = infer_extras(cfg, hardware=hardware)
        assert "xgboost" in extras
        assert not any(e.startswith("torch-") for e in extras)


def test_backend_extra_pure() -> None:
    assert backend_extra("a.pt", ResolvedHardware.cuda) == "torch-cuda"
    assert backend_extra("a.onnx", ResolvedHardware.xpu) == "onnx-intel"
    assert backend_extra("a.pth", ResolvedHardware.cpu) == "torch-cpu"
    assert backend_extra("a.ubj", ResolvedHardware.cuda) == "xgboost"
    assert backend_extra("a.ubj", ResolvedHardware.cpu) == "xgboost"


def test_builtin_name_falls_back_to_torch() -> None:
    # Built-in torchvision models are extensionless (e.g. "resnet50"); they have
    # no backend registration, so deps inference defaults to the torch runtime.
    # The Bare-bootstrap demo relies on this.
    assert backend_extra("resnet50", ResolvedHardware.cpu) == "torch-cpu"
    assert backend_extra("resnet50", ResolvedHardware.xpu) == "torch-intel"


def test_extra_for_spec_bare_and_split() -> None:
    # Single-wheel runtime: bare extra on every hardware.
    bare: frozenset[ResolvedHardware] = frozenset()
    assert _extra_for_spec("xgboost", bare, ResolvedHardware.cuda) == "xgboost"
    # Hardware-split: suffix from ResolvedHardware (xpu -> intel).
    full = frozenset(ResolvedHardware)
    assert _extra_for_spec("torch", full, ResolvedHardware.xpu) == "torch-intel"


def test_extra_for_spec_partial_matrix_errors_on_unsupported_hw() -> None:
    # A backend shipping only cpu+cuda wheels, asked for xpu -> clear error.
    partial = frozenset({ResolvedHardware.cpu, ResolvedHardware.cuda})
    with pytest.raises(ValueError, match="no xpu build"):
        _extra_for_spec("foo", partial, ResolvedHardware.xpu)
    # Supported hardware still resolves.
    assert _extra_for_spec("foo", partial, ResolvedHardware.cuda) == "foo-cuda"


def test_captum_explainer_block_adds_extra() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "transparency": {
            "ig": {"_target_": "CaptumExplainer", "algorithm": "IntegratedGradients"},
        },
    }
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "captum" in extras


def test_shap_explainer_block_adds_extra() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "transparency": {
            "shap_block": {"_target_": "ShapExplainer", "algorithm": "GradientExplainer"}
        },
    }
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "shap" in extras


def test_robustness_extras() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "robustness": {
            "ta": {"_target_": "TorchattacksAssessor", "algorithm": "PGD"},
            "fb": {"_target_": "FoolboxAssessor", "algorithm": "LinfPGD"},
            "mb": {"_target_": "MarabouAssessor", "algorithm": "linf-box"},
        },
    }
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert {"torchattacks", "foolbox", "marabou"} <= extras


def test_reporting_html_uses_html_extra() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "reporting": {"_target_": "HTMLReporter", "filename": "r"},
    }
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "html" in extras
    assert "pdf" not in extras


def test_reporting_pdf_uses_pdf_extra() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "reporting": {"_target_": "PDFReporter", "filename": "r"},
    }
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "pdf" in extras
    assert "html" not in extras


def test_reporting_disabled() -> None:
    cfg = {"model": {"source": "x.pt"}, "reporting": {"_target_": None}}
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "pdf" not in extras and "html" not in extras


def test_tracking_mlflow() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "tracking": {"_target_": "MLFlowTracker"},
    }
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "mlflow" in extras


def test_metrics_block_adds_extra() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "metrics": {"_target_": "MulticlassClassificationMetrics", "num_classes": 3},
    }
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "metrics" in extras


def test_unknown_target_raises() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "transparency": {"bad": {"_target_": "TotallyMadeUpAdapter"}},
    }
    with pytest.raises(UnknownAdapterTargetError):
        infer_extras(cfg, hardware=ResolvedHardware.cpu)


def test_unknown_target_message_lists_known_adapters() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "transparency": {"bad": {"_target_": "djd"}},
    }
    with pytest.raises(UnknownAdapterTargetError) as excinfo:
        infer_extras(cfg, hardware=ResolvedHardware.cpu)
    msg = str(excinfo.value)
    # End-user-facing: names the bad target and enumerates valid ones.
    assert "djd" in msg
    assert "Known adapters:" in msg
    assert "CaptumExplainer" in msg


def test_unknown_target_message_suggests_close_match() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        # One-char typo of a real adapter -> difflib should suggest it.
        "transparency": {"bad": {"_target_": "CaptumExplaner"}},
    }
    with pytest.raises(UnknownAdapterTargetError) as excinfo:
        infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "Did you mean 'CaptumExplainer'?" in str(excinfo.value)


def test_visualisers_do_not_contribute() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "transparency": {
            "ig": {
                "_target_": "CaptumExplainer",
                "visualisers": [{"_target_": "CaptumImageVisualiser"}],
            }
        },
    }
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "captum" in extras


def test_launcher_extra() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "hydra": {
            "launcher": {
                "_target_": "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
            }
        },
    }
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "launcher" in extras


def test_fully_qualified_target_still_resolves() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "transparency": {
            "ig": {"_target_": "raitap.transparency.explainers.captum_explainer.CaptumExplainer"}
        },
    }
    extras, _ = infer_extras(cfg, hardware=ResolvedHardware.cpu)
    assert "captum" in extras


def test_mapping_table_lists_all_known_targets() -> None:
    # The canonical class-name → extra map is now sourced from the AST scanner
    # (:mod:`raitap.deps.static_scan`) so the deps bootstrap stays usable in
    # partial-extras venvs. Runtime ``ADAPTER_EXTRAS`` is still populated
    # lazily by ``AdapterMixin.__init_subclass__`` but can be partial when an
    # adapter module fails to import — see
    # ``raitap.deps.tests.test_static_scan`` for the scanner-side guard.
    from raitap.deps.static_scan import scan_adapter_extras

    assert set(scan_adapter_extras()) == {
        "CaptumExplainer",
        "ShapExplainer",
        "TorchattacksAssessor",
        "FoolboxAssessor",
        "MarabouAssessor",
        "AutoLiRPAAssessor",
        "ImageCorruptionsAssessor",
        "HTMLReporter",
        "PDFReporter",
        "MLFlowTracker",
        "BinaryClassificationMetrics",
        "MulticlassClassificationMetrics",
        "MultilabelClassificationMetrics",
        "DetectionMetrics",
        "QuantusEvaluator",
    }
