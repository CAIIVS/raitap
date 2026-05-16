"""End-to-end inference checks against representative composed configs.

These tests bundle config snippets inline so they remain runnable in CI
without the (gitignored) ``examples/`` artifacts. The shapes mirror the
real ``examples/lwise-ham10000/assessment*.yaml`` files; if the inference
mapping ever drifts away from what those examples expect, these tests
fail loudly.
"""

from __future__ import annotations

from raitap.deps.inference import infer_extras

_LWISE_ASSESSMENT: dict = {
    "experiment_name": "lwise-ham10000-dermoscopy-demo",
    "hardware": "gpu",
    "model": {
        "source": "lwise_ham10000_inner_resnet50_state_dict.pt",
        "arch": "resnet50",
        "num_classes": 7,
    },
    "data": {"name": "ham10000-presentation-balanced"},
    "metrics": {"_target_": "ClassificationMetrics", "task": "multiclass"},
    "transparency": {
        "gradcam": {"_target_": "CaptumExplainer", "algorithm": "LayerGradCam"},
        "saliency": {"_target_": "CaptumExplainer", "algorithm": "Saliency"},
    },
    "robustness": {
        "fgsm": {"_target_": "TorchattacksAssessor", "algorithm": "FGSM"},
        "pgd": {"_target_": "TorchattacksAssessor", "algorithm": "PGD"},
        "marabou_linf": {"_target_": "MarabouAssessor", "algorithm": "linf-box"},
    },
    "reporting": {"_target_": "HTMLReporter", "filename": "lwise_ham10000_report.pdf"},
}


_LWISE_ASSESSMENT_MLFLOW: dict = {
    **_LWISE_ASSESSMENT,
    "tracking": {"_target_": "MLFlowTracker"},
}


def test_lwise_ham10000_assessment_extras() -> None:
    extras, _ = infer_extras(_LWISE_ASSESSMENT, hardware="cpu")
    assert {"torch-cpu", "captum", "metrics", "html", "torchattacks", "marabou"} <= extras


def test_lwise_ham10000_assessment_mlflow_extras() -> None:
    extras, _ = infer_extras(_LWISE_ASSESSMENT_MLFLOW, hardware="cpu")
    assert {
        "torch-cpu",
        "captum",
        "metrics",
        "html",
        "torchattacks",
        "marabou",
        "mlflow",
    } <= extras


def test_lwise_ham10000_with_xpu_picks_torch_intel() -> None:
    extras, _ = infer_extras(_LWISE_ASSESSMENT, hardware="xpu")
    assert "torch-intel" in extras
    assert "torch-cpu" not in extras


def test_lwise_ham10000_with_cuda_picks_torch_cuda() -> None:
    extras, _ = infer_extras(_LWISE_ASSESSMENT, hardware="cuda")
    assert "torch-cuda" in extras
    assert "torch-cpu" not in extras


_MARABOU_MNIST_DEMO: dict = {
    "model": {"source": "mlp_mnist.onnx"},
    "data": {"name": "mnist_samples"},
    "metrics": {"_target_": "ClassificationMetrics", "task": "multiclass"},
    "robustness": {
        "marabou_linf": {"_target_": "MarabouAssessor", "algorithm": "linf-box"},
    },
    "reporting": {"_target_": "HTMLReporter"},
}


def test_marabou_mnist_demo_picks_onnx_backend() -> None:
    extras, _ = infer_extras(_MARABOU_MNIST_DEMO, hardware="cpu")
    assert "onnx-cpu" in extras
    assert "torch-cpu" not in extras
    assert "marabou" in extras
    assert "metrics" in extras
    assert "html" in extras
