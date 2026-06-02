"""End-to-end smoke tests for every recipe under ``docs/using-raitap/examples/``.

Each recipe ships a ``:python:`` block inside a ``{recipe}`` directive. The
tests below build the same ``AppConfig`` programmatically (mirror of the
recipe's Python tab) and run it through :func:`raitap.run`, so a broken
recipe — wrong builder kwarg, mistyped target, removed adapter — fails CI
immediately rather than only at the next manual try.

Recipes share most of the config (vit_b_32 on imagenet_samples); only the
transparency / robustness / visualisers dicts vary. We re-state the cfg in
Python rather than parsing the markdown so this file is self-contained and
the cfg stays type-checkable.

The tests are wall-clock slow (one full pipeline run each) so the suite is
gated behind the ``e2e`` marker (declared in ``[tool.pytest.ini_options]``)
— run on CI but skipped in the default ``pytest`` invocation. Update
:func:`raitap.tests.test_api` if you change the baseline shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import pytest

pytest.importorskip("torchattacks")  # the multi-assessor recipe needs it
pytest.importorskip("captum")  # every recipe uses Captum
pytest.importorskip("torchmetrics")  # metrics adapter

from raitap import AppConfig, Hardware, run
from raitap.data import DataConfig, LabelsConfig
from raitap.metrics import multiclass_classification as classification
from raitap.models import ModelConfig
from raitap.pipeline.outputs import RunOutputs
from raitap.robustness import image_pair, torchattacks
from raitap.robustness.report import RobustnessPhaseResult
from raitap.transparency import captum, captum_image
from raitap.transparency.report import TransparencyPhaseResult

if TYPE_CHECKING:
    from raitap.configs.schema import MetricsConfig, ReportingConfig


def _transparency(outputs: RunOutputs) -> TransparencyPhaseResult | None:
    result = outputs.phase_results.get("transparency")
    assert result is None or isinstance(result, TransparencyPhaseResult)
    return result


def _explanations(outputs: RunOutputs) -> list:
    result = _transparency(outputs)
    return list(result.explanations) if result is not None else []


def _visualisations(outputs: RunOutputs) -> list:
    result = _transparency(outputs)
    return list(result.visualisations) if result is not None else []


def _robustness_results(outputs: RunOutputs) -> list:
    result = outputs.phase_results.get("robustness")
    assert result is None or isinstance(result, RobustnessPhaseResult)
    return list(result.robustness_results) if result is not None else []


class _BaseKwargs(TypedDict):
    """Typed shape so ``**_base_kwargs(...)`` keeps pyright happy at unpack
    sites — without this, the dict-of-object would erase every field's type."""

    hardware: Hardware
    experiment_name: str
    model: ModelConfig
    data: DataConfig
    metrics: MetricsConfig | None
    reporting: ReportingConfig | None


def _base_kwargs(experiment_name: str) -> _BaseKwargs:
    return {
        "hardware": Hardware.cpu,  # CI may have no GPU; recipes show ``gpu`` but tests use cpu
        "experiment_name": experiment_name,
        "model": ModelConfig(source="vit_b_32"),
        "data": DataConfig(
            name="imagenet_samples",
            source="imagenet_samples",
            forward_batch_size=4,
            labels=LabelsConfig(
                source="imagenet_samples",
                id_column="image",
                column="label",
            ),
        ),
        "metrics": classification(num_classes=1000),
        "reporting": None,  # skip report rendering for speed; recipes show ``html(...)``
    }


@pytest.mark.e2e
def test_recipe_imagenet_captum_ig_pgd() -> None:
    cfg = AppConfig(
        **_base_kwargs("imagenet-captum-ig-pgd"),
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
    )
    outputs = run(cfg, verbose=False)
    assert isinstance(outputs, RunOutputs)
    assert len(_explanations(outputs)) == 1
    assert len(_robustness_results(outputs)) == 1


@pytest.mark.e2e
def test_recipe_multi_explainer() -> None:
    cfg = AppConfig(
        **_base_kwargs("multi-explainer"),
        transparency={
            "ig": captum(
                algorithm="IntegratedGradients",
                call={"target": 0},
                visualisers=[captum_image()],
            ),
            "saliency": captum(
                algorithm="Saliency",
                call={"target": 0},
                visualisers=[captum_image()],
            ),
        },
    )
    outputs = run(cfg, verbose=False)
    assert len(_explanations(outputs)) == 2
    assert {er.explainer_name for er in _explanations(outputs)} == {"ig", "saliency"}


@pytest.mark.e2e
def test_recipe_multi_visualiser() -> None:
    cfg = AppConfig(
        **_base_kwargs("multi-visualiser"),
        transparency={
            "default": captum(
                algorithm="IntegratedGradients",
                call={"target": 0},
                visualisers=[
                    captum_image(
                        method="blended_heat_map",
                        sign="all",
                        title="Integrated gradients (blended)",
                    ),
                    captum_image(
                        method="heat_map",
                        sign="absolute_value",
                        title="Integrated gradients (absolute)",
                    ),
                ],
            ),
        },
    )
    outputs = run(cfg, verbose=False)
    assert len(_explanations(outputs)) == 1
    # Each visualiser renders one figure per sample-batch.
    assert len(_visualisations(outputs)) >= 2


@pytest.mark.e2e
def test_recipe_multi_assessor() -> None:
    cfg = AppConfig(
        **_base_kwargs("multi-assessor"),
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
            "fgsm": torchattacks(
                algorithm="FGSM",
                constructor={"eps": 0.03},
                visualisers=[image_pair()],
            ),
        },
    )
    outputs = run(cfg, verbose=False)
    assert len(_robustness_results(outputs)) == 2
    assert {rr.assessor_name for rr in _robustness_results(outputs)} == {"pgd", "fgsm"}
