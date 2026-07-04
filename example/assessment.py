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
from raitap.data import DataConfig, TabularLabelsConfig
from raitap.metrics import multiclass_classification
from raitap.models import ModelConfig
from raitap.reporting import html
from raitap.robustness import image_pair, torchattacks
from raitap.transparency import captum, captum_image


def build_config() -> AppConfig:
    return AppConfig(
        hardware=Hardware.gpu,
        experiment_name="example",
        model=ModelConfig(source="vit_b_32"),
        data=DataConfig(
            name="imagenet_samples",
            source="imagenet_samples",
            forward_batch_size=4,
            labels=TabularLabelsConfig(
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
    # (see ``raitap.pipeline.outputs``). Typed convenience views read straight
    # off the keyed ``phase_results``: ``outputs.transparency`` / ``.robustness``
    # are lists of per-adapter results (empty when the phase wasn't configured),
    # ``outputs.metrics`` is the ``MetricResult`` (or ``None``).
    print("\n--- programmatic access demo -----------------------------------")
    print(f"explanations:          {len(outputs.transparency)}")
    print(f"robustness results:    {len(outputs.robustness)}")
    print(f"forward batch size:    {outputs.forward_output.batch_size}")

    if outputs.metrics is not None:
        scalars = outputs.metrics.scalars
        print("metrics (scalar):")
        for name, value in sorted(scalars.items()):
            print(f"  {name:30s} {value:.4f}")

    correct = sum(1 for p in outputs.prediction_summaries if p.correct)
    total = len(outputs.prediction_summaries)
    if total:
        print(f"accuracy (recomputed): {correct}/{total} = {correct / total:.2%}")

    for rr in outputs.robustness:
        rate = rr.metrics.attack_success_rate
        if rate is not None:
            print(f"{rr.name or rr.algorithm:22s} attack success: {rate:.2%}")

    for er in outputs.transparency:
        attrs = er.attributions  # torch.Tensor, shape (N, C, H, W) for images
        magnitudes = attrs.abs().flatten(1).mean(dim=1)
        print(f"mean |attr| per sample: {[round(m.item(), 4) for m in magnitudes]}")
