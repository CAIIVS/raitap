"""End-to-end proof for the text-classification (SST-2) use case (#340).

Runs the ``contributor-configs/text-classification-sst2`` config through the
real pipeline: load HF sentiment model + tokenizer, tokenise a small text CSV
into ``input_ids`` + ``attention_mask``, forward pass, metrics, then
LayerIntegratedGradients token attribution rendered by ``CaptumTextVisualiser``.

Network + heavy optional deps are required (the model downloads from the HF
hub); skipped when ``transformers``/``captum`` are absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig

pytest.importorskip("transformers")
pytest.importorskip("captum")

_CONFIG_DIR = (
    Path(__file__).resolve().parents[4] / "contributor-configs" / "text-classification-sst2"
)
_ARTIFACTS = _CONFIG_DIR / "artifacts"


@pytest.mark.slow
@pytest.mark.e2e
def test_text_classification_config_runs_end_to_end(tmp_path: Path) -> None:
    from hydra import compose, initialize_config_dir

    from raitap.api import run

    reviews = _ARTIFACTS / "reviews.csv"
    labels = _ARTIFACTS / "labels.csv"
    if not reviews.exists() or not labels.exists():
        pytest.skip("fixtures missing; run build_data.py")

    # Override the ${hydra:runtime.cwd} paths with absolute ones: the hydra
    # resolver is not active under bare ``compose`` (no ``hydra.main``).
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        cfg = compose(
            config_name="assessment",
            overrides=[
                f"data.source={reviews}",
                f"data.labels.source={labels}",
            ],
        )

    try:
        outputs = run(
            cast("AppConfig", cfg),
            verbose=False,
            output_root=tmp_path,
            acknowledge_preprocessing_off=True,
        )
    except OSError as error:  # hub download failure (offline CI) => skip, not fail
        pytest.skip(f"HF hub unreachable: {error}")

    # Transparency produced a token-attribution explanation with a rendered
    # visualisation (the CaptumTextVisualiser panel).
    explanations = outputs.transparency
    assert explanations, "no transparency explanations produced"
    assert any(exp.visualisations for exp in explanations), "no rendered token-attribution panel"

    # Attributions are one score per token (1-D per sample, shape (N, T)).
    attributions = explanations[0].attributions
    assert attributions.ndim >= 1
    assert attributions.shape[0] == 6  # one row per review

    # An HTML report landed under the run dir.
    html_reports = list(tmp_path.rglob("*.html"))
    assert html_reports, "no HTML report generated"
    report_text = html_reports[0].read_text(encoding="utf-8")
    # The report embeds the token-attribution figure (base64 PNG <img>) and
    # names the explainer.
    assert "token_ig" in report_text
    assert "<img" in report_text
