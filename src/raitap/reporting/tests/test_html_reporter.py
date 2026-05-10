from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pytest

from raitap.configs.schema import AppConfig, ReportingConfig
from raitap.reporting.html_reporter import HTMLReporter
from raitap.reporting.sections import ReportGroup, ReportSection


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
    assert 'href="#robustness-fgsm_linf_fast"' in html or ('id="robustness-fgsm_linf_fast"' in html)
    assert '<link rel="stylesheet" href="report.css">' in html
    assert (tmp_path / "report.css").exists()


def test_html_reporter_renders_reviewed_browser_structure(
    tmp_path: Path,
) -> None:
    HTMLReporter(_config()).generate(_synthetic_sections(), report_dir=tmp_path)

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    css = (tmp_path / "report.css").read_text(encoding="utf-8")

    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert "Clean → adversarial accuracy" in html
    assert "Model Prediction: 5" in html
    assert "Ground Truth: 4" in html
    assert "Confidence: 0.897" in html
    assert "File: ISIC_0025964.jpg" in html
    assert html.index("<h3>Local</h3>") < html.index("<h3>Explainer Reference</h3>")
    assert html.index('class="explainer-card original-card"') < html.index(
        'alt="Grad-CAM lesion localisation"'
    )
    assert 'src="_assets/sample_3_original.png"' in html
    assert ".section-heading" in css
    assert "@media (max-width: 900px)" in css


def test_html_reporter_generates_valid_fragment_links_and_image_alts(
    tmp_path: Path,
) -> None:
    HTMLReporter(_config()).generate(_synthetic_sections(), report_dir=tmp_path)

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    parser = _ReportHTMLParser()
    parser.feed(html)

    ids = Counter(parser.ids)
    assert [identifier for identifier, count in ids.items() if count > 1] == []
    assert [href for href in parser.fragment_hrefs if href not in ids] == []
    assert parser.image_count > 0
    assert parser.missing_alt_count == 0


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


class _ReportHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.fragment_hrefs: list[str] = []
        self.image_count = 0
        self.missing_alt_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if identifier := attributes.get("id"):
            self.ids.append(identifier)
        if tag == "a":
            href = attributes.get("href")
            if href is not None and href.startswith("#"):
                self.fragment_hrefs.append(href[1:])
        if tag == "img":
            self.image_count += 1
            if "alt" not in attributes:
                self.missing_alt_count += 1


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
                        ("sample_id", "ISIC_0025964.jpg"),
                        ("predicted_class", "5"),
                        ("confidence", "0.8968"),
                        ("target_class", "4"),
                        ("correct", "False"),
                    ),
                    metadata={"role": "sample_header", "bucket": "wrong", "sample_index": 3},
                    images=(Path("reports/_assets/sample_3_original.png"),),
                ),
                ReportGroup(
                    heading="Explainer: gradcam_localisation - Visualiser: Grad-CAM",
                    images=(Path("reports/_assets/sample_3_gradcam.png"),),
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
