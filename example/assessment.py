"""Python equivalent of ``assessment.yaml`` — same run, programmatic config.

Run with::

    uv run python assessment.py

Uses raw schema dataclasses + ``_target_`` strings rather than the
:mod:`raitap.<module>` hydra-zen builders. Reason: builders live next to
their wrapped third-party library (``classification`` lives in
``raitap.metrics.classification_metrics``, which imports ``torchmetrics``),
so importing a builder requires the very extras :func:`install_raitap_deps`
is about to install. The schema dataclass + ``_target_`` form is the only
import-free way to express the config before the first install. The
builder shape (see ``docs/using-raitap/configuration/python-api.md``) is
fine for everyday scripts where extras are already pinned.
"""

from __future__ import annotations

import logging

# All imports below are intentionally light — none of them pull a wrapped
# adapter library (torch / Captum / torchattacks / ...). This is what lets
# ``install_raitap_deps`` walk the config and install the missing extras on
# first run before the heavy imports happen.
from raitap.configs.schema import (
    AppConfig,
    DataConfig,
    LabelsConfig,
    MetricsConfig,
    ModelConfig,
    ReportingConfig,
    RobustnessConfig,
    TransparencyConfig,
)
from raitap.deps import install_raitap_deps
from raitap.types import Hardware, Task


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
        metrics=MetricsConfig(_target_="ClassificationMetrics", task=Task.multiclass),
        transparency={
            "default": TransparencyConfig(
                _target_="CaptumExplainer",
                algorithm="IntegratedGradients",
                call={"target": 0},
                visualisers=[{"_target_": "CaptumImageVisualiser"}],
            ),
        },
        robustness={
            "pgd": RobustnessConfig(
                _target_="TorchattacksAssessor",
                algorithm="PGD",
                constructor={"eps": 0.03, "alpha": 0.005, "steps": 10},
                visualisers=[{"_target_": "ImagePairVisualiser"}],
            ),
        },
        reporting=ReportingConfig(_target_="HTMLReporter", filename="report"),
    )


if __name__ == "__main__":
    cfg = build_config()
    install_raitap_deps(cfg, allow_project_edit=True)

    # ``raitap.run`` delegates step-by-step progress to Python ``logging`` —
    # without a handler the run is silent apart from the summary panel.
    # Configure the root logger before ``run`` to see per-explainer / per-
    # assessor messages.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Deferred so adapter backends are pulled only after the bootstrap has
    # installed them.
    from raitap import run

    outputs = run(cfg)

    # ``outputs`` is a ``RunOutputs`` dataclass (see
    # ``raitap.pipeline.outputs``). All artefacts the report consumes are
    # already in memory — no need to re-read the files under ``outputs/``.
    print("\n--- programmatic access demo -----------------------------------")
    print(f"explanations:          {len(outputs.explanations)}")
    print(f"robustness results:    {len(outputs.robustness_results)}")
    print(f"forward output shape:  {tuple(outputs.forward_output.shape)}")

    if outputs.metrics is not None:
        scalars = outputs.metrics.result.metrics  # dict[str, float]
        print("metrics (scalar):")
        for name, value in sorted(scalars.items()):
            print(f"  {name:30s} {value:.4f}")

    # Per-sample predictions: each entry carries (index, predicted_class,
    # confidence, sample_id, target_class, correct).
    correct = sum(1 for p in outputs.prediction_summaries if p.correct)
    total = len(outputs.prediction_summaries)
    if total:
        print(f"accuracy (recomputed): {correct}/{total} = {correct / total:.2%}")

    # Robustness: how often the PGD attack succeeded per assessor run.
    for rr in outputs.robustness_results:
        rate = rr.metrics.attack_success_rate
        if rate is not None:
            print(f"{rr.assessor_name or rr.algorithm:22s} attack success: {rate:.2%}")

    # Dummy downstream op: average attribution magnitude per sample.
    for er in outputs.explanations:
        attrs = er.attributions  # torch.Tensor, shape (N, C, H, W) for images
        magnitudes = attrs.abs().flatten(1).mean(dim=1)  # per-sample mean |attr|
        print(f"mean |attr| per sample: {[round(m.item(), 4) for m in magnitudes]}")
