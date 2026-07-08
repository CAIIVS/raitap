"""Tests for MLflow summary param extraction from resolved config dicts."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from raitap.tracking.mlflow_tracker import _mlflow_summary_params

if TYPE_CHECKING:
    from pathlib import Path


def test_summary_params_includes_explainers_and_per_explainer_fields() -> None:
    cfg = {
        "experiment_name": "demo",
        "model": {"source": "resnet50"},
        "data": {"name": "isic2018", "source": "/data"},
        "transparency": {
            "captum_ig": {
                "use": "captum",
                "algorithm": "IntegratedGradients",
                "visualisers": [],
            },
            "captum_saliency": {
                "use": "captum",
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
    assert params["transparency.captum_ig.use"] == "captum"
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
        "transparency": {"bad": "not-a-dict", "ok": {"algorithm": "IG", "use": "captum"}},
    }
    params = _mlflow_summary_params(cfg)

    assert params["transparency.explainers"] == "bad,ok"
    assert params["transparency.ok.algorithm"] == "IG"
    assert "transparency.bad.algorithm" not in params


def test_summary_params_records_custom_preprocessing_file_identity(
    tmp_path: Path,
) -> None:
    preprocessing_file = tmp_path / "preprocessing.py"
    contents = b"raise RuntimeError('tracking must not import this file')\n"
    preprocessing_file.write_bytes(contents)

    params = _mlflow_summary_params(
        {
            "data": {
                "preprocessing": str(preprocessing_file),
            },
        }
    )

    assert params["data.preprocessing"] == str(preprocessing_file)
    assert params["data.preprocessing.origin"] == "custom-file"
    assert params["data.preprocessing.file_path"] == str(preprocessing_file.resolve())
    assert params["data.preprocessing.file_sha256"] == hashlib.sha256(contents).hexdigest()
