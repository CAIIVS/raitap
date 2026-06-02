"""Python equivalent of ``assessment.yaml`` — same run, programmatic config.

Run with::

    uv run python assessment.py

Mirrors ``assessment.yaml`` field-for-field via the per-module hydra-zen
builders (see ``docs/using-raitap/configuration/python-api.md``). The two
paths drive identical pipelines (modulo PyTorch determinism).

Adapter modules use lazy imports of their wrapped libraries
(``raitap.utils.lazy.lazy_import``), so this file can be imported in a venv
that has only the base raitap dep — ``run(cfg, auto_install_deps=True)`` walks
the cfg and installs the missing extras on first run, then re-execs the
script (same flow as the CLI's ``--allow-project-edit`` / ``-y``).
"""

from __future__ import annotations

import logging

from raitap import AppConfig, Hardware, run
from raitap.data import DataConfig, LabelsConfig
from raitap.metrics import multiclass_classification
from raitap.metrics.factory import MetricsEvaluation
from raitap.models import ModelConfig
from raitap.reporting import html
from raitap.robustness import image_pair, torchattacks
from raitap.robustness.report import RobustnessPhaseResult
from raitap.transparency import captum, captum_image
from raitap.transparency.report import TransparencyPhaseResult


def build_config() -> AppConfig:
    return AppConfig(
        hardware=Hardware.gpu,
        experiment_name="example",
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
        metrics=multiclass_classification(num_classes=1000),
        transparency={
            "default": captum(
                algorithm="IntegratedGradients",
                call={"target": 0},
                visualisers=[captum_image()],
            ),
        },
        robustness={
            "pgd": torchattacks(
                algorithm="PGD",
                constructor={"eps": 0.03, "alpha": 0.005, "steps": 10},
                visualisers=[image_pair()],
            ),
        },
        reporting=html(filename="report"),
    )


if __name__ == "__main__":
    cfg = build_config()

    # ``raitap.run`` delegates step-by-step progress to Python ``logging`` —
    # without a handler the run is silent apart from the summary panel.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ``auto_install_deps=True`` runs the same auto-deps flow as the CLI's
    # ``--allow-project-edit`` / ``-y``: walk the cfg, install missing
    # extras via ``uv add`` (or ``pip install``), re-exec the script.
    outputs = run(cfg, auto_install_deps=True)

    # Programmatic access demo — ``outputs`` is a ``RunOutputs`` dataclass
    # (see ``raitap.pipeline.outputs``). All artefacts the report consumes
    # are already in memory.
    # Programmatic access: results are keyed by phase name in ``phase_results``
    # (each value is a ``PhaseResult``). Only configured phases are present.
    print("\n--- programmatic access demo -----------------------------------")
    transparency = outputs.phase_results.get("transparency")
    robustness = outputs.phase_results.get("robustness")
    metrics = outputs.phase_results.get("metrics")
    assert transparency is None or isinstance(transparency, TransparencyPhaseResult)
    assert robustness is None or isinstance(robustness, RobustnessPhaseResult)
    assert metrics is None or isinstance(metrics, MetricsEvaluation)

    explanations = transparency.explanations if transparency is not None else []
    robustness_results = robustness.robustness_results if robustness is not None else []
    print(f"explanations:          {len(explanations)}")
    print(f"robustness results:    {len(robustness_results)}")
    print(f"forward batch size:    {outputs.forward_output.batch_size}")

    if metrics is not None:
        scalars = metrics.result.metrics  # dict[str, float]
        print("metrics (scalar):")
        for name, value in sorted(scalars.items()):
            print(f"  {name:30s} {value:.4f}")

    correct = sum(1 for p in outputs.prediction_summaries if p.correct)
    total = len(outputs.prediction_summaries)
    if total:
        print(f"accuracy (recomputed): {correct}/{total} = {correct / total:.2%}")

    for rr in robustness_results:
        rate = rr.metrics.attack_success_rate
        if rate is not None:
            print(f"{rr.assessor_name or rr.algorithm:22s} attack success: {rate:.2%}")

    for er in explanations:
        attrs = er.attributions  # torch.Tensor, shape (N, C, H, W) for images
        magnitudes = attrs.abs().flatten(1).mean(dim=1)
        print(f"mean |attr| per sample: {[round(m.item(), 4) for m in magnitudes]}")
