from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from raitap.configs.schema import AppConfig, ReportingConfig
from raitap.reporting.html_reporter import HTMLReporter
from raitap.reporting.sections import ReportGroup, ReportSection

if TYPE_CHECKING:
    from pathlib import Path


def test_html_reporter_generates_browser_html_with_expected_anchors(
    tmp_path: Path,
) -> None:
    config = _config()

    output_path = HTMLReporter(config).generate(_synthetic_sections(), report_dir=tmp_path)

    html_path = tmp_path / "report.html"
    assert output_path == html_path
    assert html_path.exists()
    assert not (tmp_path / "report.pdf").exists()

    html = html_path.read_text(encoding="utf-8")
    assert 'href="#sample-3"' in html or 'id="sample-3"' in html
    assert 'href="#explainer-gradcam_localisation"' in html or (
        'id="explainer-gradcam_localisation"' in html
    )
    assert 'href="#robustness-fgsm_linf_fast"' in html or (
        'id="robustness-fgsm_linf_fast"' in html
    )
    assert '<link rel="stylesheet" href="report.css">' in html
    assert (tmp_path / "report.css").exists()


@pytest.mark.parametrize(
    ("configured_filename", "expected_html_name"),
    [
        ("report.pdf", "report.html"),
        ("lwise_ham10000_report.pdf", "lwise_ham10000_report.html"),
        ("custom", "custom.html"),
    ],
)
def test_html_reporter_uses_configured_basename_with_html_suffix(
    tmp_path: Path,
    configured_filename: str,
    expected_html_name: str,
) -> None:
    config = _config(filename=configured_filename)

    output_path = HTMLReporter(config).generate(_synthetic_sections(), report_dir=tmp_path)

    assert output_path == tmp_path / expected_html_name
    assert output_path.exists()
    assert not (tmp_path / configured_filename).exists()


def test_html_reporter_omits_missing_sections(
    tmp_path: Path,
) -> None:
    sections = (
        ReportSection.from_groups(
            "Local Explanations",
            [],
            metadata={"section_role": "local"},
        ),
    )

    HTMLReporter(_config()).generate(sections, report_dir=tmp_path)

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert 'id="metrics-panel"' not in html
    assert 'id="robustness-section"' not in html
    assert 'class="card empty-card"' not in html
    assert "No local explanations" in html


def _config(*, filename: str = "report.pdf") -> AppConfig:
    config = AppConfig(experiment_name="html-report-test")
    config.reporting = ReportingConfig(
        _target_="raitap.reporting.HTMLReporter",
        filename=filename,
    )
    return config


def _synthetic_sections() -> tuple[ReportSection, ...]:
    return (
        ReportSection.from_groups(
            "Metrics",
            [
                ReportGroup(
                    heading="Performance Metrics",
                    table_rows=(("accuracy", "0.9000"),),
                    metadata={"role": "metrics"},
                )
            ],
            metadata={"section_role": "metrics"},
        ),
        ReportSection.from_groups(
            "Local Explanations",
            [
                ReportGroup(
                    heading="Sample - wrong",
                    table_rows=(
                        ("bucket", "wrong"),
                        ("sample_index", "3"),
                        ("predicted_class", "5"),
                        ("confidence", "0.8968"),
                        ("target_class", "4"),
                        ("correct", "False"),
                    ),
                    metadata={"role": "sample_header", "bucket": "wrong", "sample_index": 3},
                ),
                ReportGroup(
                    heading="Explainer: gradcam_localisation - Visualiser: Grad-CAM",
                    table_rows=(
                        ("explainer", "gradcam_localisation"),
                        ("algorithm", "LayerGradCam"),
                        ("method_families", "cam, gradient"),
                        ("targets", "3: 5"),
                        ("layer_path", "1.layer4.2.conv3"),
                        ("visualiser_sign", "positive"),
                        ("call.relu_attributions", "true"),
                    ),
                    metadata={
                        "role": "local_visualiser",
                        "bucket": "wrong",
                        "sample_index": 3,
                        "explainer_name": "gradcam_localisation",
                        "algorithm": "LayerGradCam",
                        "visualiser_name": "Grad-CAM lesion localisation",
                    },
                ),
            ],
            metadata={"section_role": "local"},
        ),
        ReportSection.from_groups(
            "Robustness",
            [
                ReportGroup(
                    heading="Adversarial attack - FGSM (fgsm_linf_fast)",
                    table_rows=(
                        ("assessor", "fgsm_linf_fast"),
                        ("algorithm", "FGSM"),
                        ("method_kind", "empirical_attack"),
                        ("clean_accuracy", "0.9000"),
                        ("adversarial_accuracy", "0.5000"),
                        ("attack_success_rate", "0.4000"),
                    ),
                    metadata={
                        "role": "robustness",
                        "assessor_name": "fgsm_linf_fast",
                        "algorithm": "FGSM",
                        "method_kind": "empirical_attack",
                        "sample_indices": [3],
                    },
                )
            ],
            metadata={"section_role": "robustness"},
        ),
    )
