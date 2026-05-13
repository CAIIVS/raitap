"""Tests for inference.infer_extras — operates on dict (post-compose) inputs."""

from __future__ import annotations

import pytest

from raitap.configs.extras.inference import (
    ADAPTER_EXTRAS,
    UnknownAdapterTargetError,
    backend_extra,
    infer_extras,
)


def test_torch_pt_cuda() -> None:
    cfg = {"model": {"source": "foo.pt"}, "hardware": "gpu"}
    extras, origins = infer_extras(cfg, hardware="cuda")
    assert "torch-cuda" in extras
    assert "model.source" in origins["torch-cuda"]


def test_torch_pt_cpu() -> None:
    cfg = {"model": {"source": "foo.pt"}, "hardware": "cpu"}
    extras, _ = infer_extras(cfg, hardware="cpu")
    assert "torch-cpu" in extras


def test_torch_pt_intel() -> None:
    cfg = {"model": {"source": "foo.pt"}}
    extras, _ = infer_extras(cfg, hardware="xpu")
    assert "torch-intel" in extras


def test_onnx_backend() -> None:
    cfg = {"model": {"source": "model.onnx"}}
    extras, _ = infer_extras(cfg, hardware="cuda")
    assert "onnx-cuda" in extras
    assert "torch-cuda" not in extras


def test_backend_extra_pure() -> None:
    assert backend_extra("a.pt", "cuda") == "torch-cuda"
    assert backend_extra("a.onnx", "xpu") == "onnx-intel"
    assert backend_extra("a.pth", "cpu") == "torch-cpu"


def test_captum_explainer_block_adds_extra() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "transparency": {
            "ig": {"_target_": "CaptumExplainer", "algorithm": "IntegratedGradients"},
        },
    }
    extras, _ = infer_extras(cfg, hardware="cpu")
    assert "captum" in extras


def test_shap_explainer_block_adds_extra() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "transparency": {"shap_block": {"_target_": "ShapExplainer", "algorithm": "GradientExplainer"}},
    }
    extras, _ = infer_extras(cfg, hardware="cpu")
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
    extras, _ = infer_extras(cfg, hardware="cpu")
    assert {"torchattacks", "foolbox", "marabou"} <= extras


def test_reporting_html_uses_jinja() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "reporting": {"_target_": "HTMLReporter", "filename": "r"},
    }
    extras, _ = infer_extras(cfg, hardware="cpu")
    assert "jinja" in extras
    assert "borb" not in extras


def test_reporting_pdf_uses_borb() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "reporting": {"_target_": "PDFReporter", "filename": "r"},
    }
    extras, _ = infer_extras(cfg, hardware="cpu")
    assert "borb" in extras
    assert "jinja" not in extras


def test_reporting_disabled() -> None:
    cfg = {"model": {"source": "x.pt"}, "reporting": {"_target_": None}}
    extras, _ = infer_extras(cfg, hardware="cpu")
    assert "borb" not in extras and "jinja" not in extras


def test_tracking_mlflow() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "tracking": {"_target_": "MLFlowTracker"},
    }
    extras, _ = infer_extras(cfg, hardware="cpu")
    assert "mlflow" in extras


def test_metrics_block_adds_extra() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "metrics": {"_target_": "ClassificationMetrics", "task": "multiclass"},
    }
    extras, _ = infer_extras(cfg, hardware="cpu")
    assert "metrics" in extras


def test_unknown_target_raises() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "transparency": {"bad": {"_target_": "TotallyMadeUpAdapter"}},
    }
    with pytest.raises(UnknownAdapterTargetError):
        infer_extras(cfg, hardware="cpu")


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
    extras, _ = infer_extras(cfg, hardware="cpu")
    assert "captum" in extras


def test_launcher_extra() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "hydra": {"launcher": {"_target_": "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"}},
    }
    extras, _ = infer_extras(cfg, hardware="cpu")
    assert "launcher" in extras


def test_fully_qualified_target_still_resolves() -> None:
    cfg = {
        "model": {"source": "x.pt"},
        "transparency": {
            "ig": {"_target_": "raitap.transparency.explainers.captum_explainer.CaptumExplainer"}
        },
    }
    extras, _ = infer_extras(cfg, hardware="cpu")
    assert "captum" in extras


def test_mapping_table_lists_all_known_targets() -> None:
    assert set(ADAPTER_EXTRAS) == {
        "CaptumExplainer",
        "ShapExplainer",
        "TorchattacksAssessor",
        "FoolboxAssessor",
        "MarabouAssessor",
        "HTMLReporter",
        "PDFReporter",
        "MLFlowTracker",
        "ClassificationMetrics",
        "DetectionMetrics",
    }
