"""Tests for the public :mod:`raitap.api` surface.

These guard the user-facing contract that documentation and tutorials
reference: ``from raitap import run, AppConfig`` and the hydra-zen builders
under ``raitap.api`` must keep working, and ``raitap.run(cfg, verbose=False)``
must drive the orchestrator end-to-end equivalently to the bundled
``--demo`` Hydra config.
"""

from __future__ import annotations

import dataclasses
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir

import raitap
from raitap import AppConfig, run
from raitap.api import instantiate
from raitap.configs.schema import (
    DataConfig,
    LabelsConfig,
    MetricsConfig,
    ModelConfig,
    RobustnessConfig,
    TransparencyConfig,
)
from raitap.data.preprocessing import resolve_preprocessing
from raitap.metrics import classification as classification_metrics
from raitap.models.model import Model
from raitap.pipeline.outputs import RunOutputs
from raitap.robustness import foolbox, torchattacks
from raitap.transparency import captum, shap
from raitap.types import Hardware, Task

FIXTURE = (
    Path(__file__).resolve().parents[1] / "data" / "tests" / "fixtures" / "preproc_imagenet.py"
)


def _configs_dir() -> Path:
    return (Path(__file__).resolve().parents[1] / "configs").resolve()


def _demo_app_config() -> AppConfig:
    """Mirror of ``src/raitap/configs/demo.yaml`` as a Python ``AppConfig``."""
    return AppConfig(
        hardware=Hardware.cpu,
        experiment_name="demo",
        model=ModelConfig(source="vit_b_32"),
        data=DataConfig(
            name="imagenet_samples",
            source="imagenet_samples",
            forward_batch_size=4,
            labels=LabelsConfig(
                source="imagenet_samples",
                id_column="image",
                column="label",
            ),
        ),
        metrics=MetricsConfig(_target_="ClassificationMetrics", task=Task.multiclass),
        transparency={
            "default": TransparencyConfig(
                _target_="CaptumExplainer",
                algorithm="IntegratedGradients",
                call={"target": 0},
                visualisers=[{"_target_": "CaptumImageVisualiser"}],
            )
        },
        robustness={
            "pgd": RobustnessConfig(
                _target_="TorchattacksAssessor",
                algorithm="PGD",
                constructor={"eps": 0.03, "alpha": 0.005, "steps": 10},
                visualisers=[{"_target_": "ImagePairVisualiser"}],
            )
        },
        # ``reporting`` left ``None`` so the smoke test stays fast and writes
        # nothing to disk beyond the Hydra output dir.
        reporting=None,
    )


def test_api_surface_exports_what_docs_will_reference() -> None:
    """`raitap.run` + `AppConfig` are top-level; builders are dataclass types."""
    # Top-level re-exports
    assert raitap.run is run
    assert raitap.AppConfig is AppConfig

    # Hydra-zen builders are exposed under ``raitap.api`` as dataclass *types*
    # (calling them with kwargs yields a config instance the orchestrator can
    # instantiate). Each builder must expose ``_target_`` so Hydra can resolve it.
    for name, builder in [
        ("captum", captum),
        ("shap", shap),
        ("torchattacks", torchattacks),
        ("foolbox", foolbox),
        ("classification_metrics", classification_metrics),
    ]:
        assert isinstance(builder, type), f"{name} should be a dataclass type"
        assert dataclasses.is_dataclass(builder), f"{name} should be a dataclass"
        field_names = {f.name for f in dataclasses.fields(builder)}
        assert "_target_" in field_names, f"{name} should carry a ``_target_`` field"

    # ``instantiate`` re-exported so users don't have to pip install hydra-zen.
    assert callable(instantiate)


def test_classification_metrics_builder_instantiates_round_trip() -> None:
    """Sanity: the builder produces a config that Hydra can instantiate."""
    from raitap.metrics.classification_metrics import ClassificationMetrics

    cfg = classification_metrics(task="multiclass", num_classes=3)
    instance = instantiate(cfg)
    assert isinstance(instance, ClassificationMetrics)


def test_explainer_builders_accept_schema_fields() -> None:
    """``captum`` / ``shap`` must accept every ``TransparencyConfig`` field.

    Guards against a hydra-zen pitfall: ``builds(Cls, populate_full_signature=True)``
    lifts only ``__init__`` params. Wrapped explainers take ``**kwargs``, so
    without ``builds_bases=(TransparencyConfig,)`` the builders would reject
    ``call=`` / ``visualisers=`` / etc. — exactly the kwargs docs and Hydra
    YAML configs use.
    """
    captum_cfg = captum(
        algorithm="IntegratedGradients",
        call={"target": 0},
        visualisers=[{"_target_": "CaptumImageVisualiser"}],
    )
    assert captum_cfg.algorithm == "IntegratedGradients"
    assert captum_cfg.call == {"target": 0}

    shap_cfg = shap(
        algorithm="GradientExplainer",
        constructor={"local_smoothing": 0.0},
        raitap={"batch_size": 1},
    )
    assert shap_cfg.constructor == {"local_smoothing": 0.0}
    assert shap_cfg.raitap == {"batch_size": 1}


def test_assessor_builders_accept_schema_fields() -> None:
    """``torchattacks`` / ``foolbox`` must accept every ``RobustnessConfig`` field."""
    torchattacks_cfg = torchattacks(
        algorithm="PGD",
        constructor={"eps": 0.03, "alpha": 0.005, "steps": 10},
        visualisers=[{"_target_": "ImagePairVisualiser"}],
    )
    assert torchattacks_cfg.algorithm == "PGD"
    assert torchattacks_cfg.constructor == {"eps": 0.03, "alpha": 0.005, "steps": 10}

    foolbox_cfg = foolbox(
        algorithm="LinfPGD",
        constructor={"rel_stepsize": 0.025, "steps": 40},
        call={"eps": 0.03},
    )
    assert foolbox_cfg.constructor["rel_stepsize"] == 0.025
    assert foolbox_cfg.call == {"eps": 0.03}


def test_programmatic_custom_file_refuses_without_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Programmatic ``preprocessing: <path>.py`` must refuse without consent.

    The CLI bootstrap sets ``RAITAP_ALLOW_PREPROCESSING_EXEC`` from
    ``--allow-preprocessing-exec``; ``raitap.run(config)`` skips that
    bootstrap, so consent must come from the
    ``acknowledge_preprocessing_exec`` kwarg on :func:`raitap.run`. With
    neither set, the resolver must raise rather than silently exec
    arbitrary Python from disk.
    """
    monkeypatch.delenv("RAITAP_ALLOW_PREPROCESSING_EXEC", raising=False)
    data_cfg = DataConfig(
        name="imagenet_samples",
        source="imagenet_samples",
        preprocessing=str(FIXTURE),
    )
    model_cfg = ModelConfig(source="vit_b_32")

    with pytest.raises(PermissionError) as excinfo:
        resolve_preprocessing(model_cfg, data_cfg)
    assert "acknowledge_preprocessing_exec" in str(excinfo.value)


def test_programmatic_custom_file_accepts_with_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resolve_preprocessing(..., acknowledge_exec=True)`` unlocks the custom-file path.

    Guards parity with the CLI's env-var consent: the Python API must accept
    consent expressed via the explicit kwarg even when the env var is unset.
    """
    monkeypatch.delenv("RAITAP_ALLOW_PREPROCESSING_EXEC", raising=False)
    data_cfg = DataConfig(
        name="imagenet_samples",
        source="imagenet_samples",
        preprocessing=str(FIXTURE),
    )
    model_cfg = ModelConfig(source="vit_b_32")

    resolved = resolve_preprocessing(model_cfg, data_cfg, acknowledge_exec=True)
    assert resolved.origin == "custom-file"


def test_model_accepts_externally_resolved_custom_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Model`` accepts a :class:`ResolvedPreprocessing` built externally.

    The orchestrator resolves preprocessing once with consent and hands the
    result to :class:`Model`; this guards that ``Model(cfg,
    resolved_preprocessing=…)`` honours the supplied resolution instead of
    re-resolving without consent. Exercises the propagation surface
    :func:`raitap.run` relies on.
    """
    monkeypatch.delenv("RAITAP_ALLOW_PREPROCESSING_EXEC", raising=False)
    cfg = _demo_app_config()
    cfg.data.preprocessing = str(FIXTURE)

    resolved = resolve_preprocessing(cfg.model, cfg.data, acknowledge_exec=True)
    model = Model(cfg, resolved_preprocessing=resolved)

    assert model.resolved_preprocessing.origin == "custom-file"
    sha = model.resolved_preprocessing.file_sha256
    assert sha is not None and len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)


@pytest.fixture(scope="module")
def _demo_run() -> RunOutputs:
    """Run the programmatic pipeline once and share results across tests.

    Skipped on environments without the ``torchattacks`` extra: the demo
    config exercises ``TorchattacksAssessor`` to mirror what ``--demo`` runs
    on the CLI. Surface tests above stay unconditional.
    """
    pytest.importorskip("torchattacks")
    cfg = _demo_app_config()
    buf = io.StringIO()
    with redirect_stdout(buf):
        outputs = run(cfg, verbose=False)
    # ``verbose=False`` must suppress the summary panel. The orchestrator
    # still routes log records through Python ``logging``; we only assert the
    # rich console banner stays silent (it writes to stdout).
    assert "Run summary" not in buf.getvalue()
    return outputs


def test_run_smoke_with_verbose_false_drives_full_pipeline(_demo_run: RunOutputs) -> None:
    """`raitap.run(cfg, verbose=False)` exits cleanly with non-empty outputs."""
    assert isinstance(_demo_run, RunOutputs)
    assert len(_demo_run.explanations) >= 1
    assert len(_demo_run.robustness_results) >= 1
    assert _demo_run.metrics is not None
    assert _demo_run.metrics.result.metrics  # at least one scalar metric


def test_run_parity_with_yaml_demo(_demo_run: RunOutputs) -> None:
    """Composing ``demo.yaml`` and our Python ``AppConfig`` produce equivalent runs.

    Bit-identical parity is too brittle (PyTorch determinism varies across
    threads/BLAS). Asserting that the Python path drives the same *shape* of
    outputs as the YAML path is enough to catch drift in either direction.
    """
    with initialize_config_dir(version_base="1.3", config_dir=str(_configs_dir())):
        yaml_cfg = compose(config_name="demo", overrides=["reporting._target_=null"])

    yaml_outputs = run(cast("AppConfig", yaml_cfg), verbose=False)

    assert len(_demo_run.explanations) == len(yaml_outputs.explanations)
    assert len(_demo_run.robustness_results) == len(yaml_outputs.robustness_results)
    assert (_demo_run.metrics is None) == (yaml_outputs.metrics is None)
    if _demo_run.metrics is not None and yaml_outputs.metrics is not None:
        assert set(_demo_run.metrics.result.metrics.keys()) == set(
            yaml_outputs.metrics.result.metrics.keys()
        )
