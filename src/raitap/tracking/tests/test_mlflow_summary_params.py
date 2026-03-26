"""Tests for MLflow summary param extraction from resolved config dicts."""

from __future__ import annotations

from raitap.tracking.mlflow_tracker import _mlflow_summary_params


def test_summary_params_includes_explainers_and_per_explainer_fields() -> None:
    cfg = {
        "experiment_name": "demo",
        "model": {"source": "resnet50"},
        "data": {"name": "isic2018", "source": "/data"},
        "transparency": {
            "captum_ig": {
                "_target_": "CaptumExplainer",
                "algorithm": "IntegratedGradients",
                "visualisers": [],
            },
            "captum_saliency": {
                "_target_": "CaptumExplainer",
                "algorithm": "Saliency",
                "visualisers": [],
            },
        },
    }
    params = _mlflow_summary_params(cfg)

    assert params["assessment.name"] == "demo"
    assert params["model.source"] == "resnet50"
    assert params["data.name"] == "isic2018"
    assert params["data.source"] == "/data"
    assert params["transparency.explainers"] == "captum_ig,captum_saliency"
    assert params["transparency.captum_ig.algorithm"] == "IntegratedGradients"
    assert params["transparency.captum_ig._target_"] == "CaptumExplainer"
    assert params["transparency.captum_saliency.algorithm"] == "Saliency"


def test_summary_params_skips_empty_and_none_values() -> None:
    cfg = {
        "experiment_name": "  ",
        "model": {},
        "data": {"name": None, "source": ""},
        "transparency": {},
    }
    params = _mlflow_summary_params(cfg)

    assert params == {}


def test_summary_params_ignores_non_dict_explainer_entries() -> None:
    cfg = {
        "experiment_name": "x",
        "model": {},
        "data": {},
        "transparency": {"bad": "not-a-dict", "ok": {"algorithm": "IG", "_target_": "T"}},
    }
    params = _mlflow_summary_params(cfg)

    assert params["transparency.explainers"] == "bad,ok"
    assert params["transparency.ok.algorithm"] == "IG"
    assert "transparency.bad.algorithm" not in params
