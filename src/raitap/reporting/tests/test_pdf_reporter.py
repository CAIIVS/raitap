"""Tests for PDF report generation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from raitap.configs import set_output_root
from raitap.configs.schema import AppConfig, ReportingConfig
from raitap.reporting.pdf_reporter import (
    _MAX_PDF_TEXT_LEN,
    _MAX_PDF_TOKEN_LEN,
    PDFReporter,
    _pdf_display_text,
    _prepare_raster_for_pdf,
)
from raitap.reporting.sections import ReportGroup, ReportSection

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def mock_config(tmp_path: Path) -> AppConfig:
    """Create a mock config for testing."""
    cfg = AppConfig(experiment_name="test_experiment")
    set_output_root(cfg, tmp_path)
    cfg.reporting = ReportingConfig(
        _target_="PDFReporter",
        filename="test.pdf",
    )
    return cfg


def test_pdf_reporter_instantiation(mock_config: AppConfig) -> None:
    """Test that PDFReporter can be instantiated."""
    reporter = PDFReporter(mock_config)
    assert reporter.config == mock_config


def _mock_borb() -> SimpleNamespace:
    return SimpleNamespace(
        Document=MagicMock(return_value=MagicMock()),
        PDF=SimpleNamespace(write=MagicMock()),
        Page=MagicMock(return_value=MagicMock()),
        Paragraph=MagicMock(),
        SingleColumnLayout=MagicMock(return_value=MagicMock(append_layout_element=MagicMock())),
        FixedColumnWidthTable=MagicMock(return_value=MagicMock(append_layout_element=MagicMock())),
        Image=MagicMock(),
        Chart=MagicMock(),
    )


def test_pdf_reporter_generates_file(mock_config: AppConfig, tmp_path: Path) -> None:
    """Test basic PDF generation."""
    reporter = PDFReporter(mock_config)

    with patch("raitap.reporting.pdf_reporter._borb_pdf_ns", return_value=_mock_borb()):
        output_path = reporter.generate(())

    # Verify output path
    assert output_path.exists() is False or output_path.parent.exists()
    assert output_path.suffix == ".pdf"
    assert "test.pdf" in str(output_path)


def test_pdf_reporter_creates_report_directory(mock_config: AppConfig, tmp_path: Path) -> None:
    """Test that report directory is created."""
    with patch("raitap.reporting.pdf_reporter._borb_pdf_ns", return_value=_mock_borb()):
        reporter = PDFReporter(mock_config)
        output_path = reporter.generate(())

        assert output_path.parent.name == "reports"


def test_pdf_display_text_shortens_path_like_tokens() -> None:
    long_path = (
        "/cluster/home/vondejon/reporting-summary/raitap/usecases/classification/"
        "isic2018/isic2018_efficientNet4_fullmodule.pt"
    )

    assert _pdf_display_text(f"Model: {long_path}") == "Model: isic2018_efficientNet4_fullmodule.pt"


def test_pdf_display_text_truncates_long_unbreakable_tokens() -> None:
    result = _pdf_display_text("x" * 120)

    assert "..." in result
    assert max(len(token) for token in result.split()) <= _MAX_PDF_TOKEN_LEN


def test_pdf_display_text_caps_long_multi_word_text() -> None:
    result = _pdf_display_text(" ".join(["safe-token"] * 30))

    assert "..." in result
    assert len(result) <= _MAX_PDF_TEXT_LEN


def test_pdf_display_text_keeps_short_text_and_handles_none() -> None:
    assert _pdf_display_text("Metrics") == "Metrics"
    assert _pdf_display_text(None) == "N/A"


def test_pdf_reporter_sanitizes_all_paragraph_text(mock_config: AppConfig) -> None:
    """Long paths/headings/table cells should be shortened before borb sees them."""
    long_model_path = (
        "/cluster/home/vondejon/reporting-summary/raitap/usecases/classification/"
        "isic2018/isic2018_efficientNet4_fullmodule.pt"
    )
    mock_config.model.source = long_model_path
    borb = _mock_borb()
    section = ReportSection.from_groups(
        "Local Explanations " + ("s" * 120),
        [
            ReportGroup(
                heading="Explainer: " + ("h" * 120),
                table_rows=(("Long metric", "v" * 120),),
            )
        ],
    )

    with patch("raitap.reporting.pdf_reporter._borb_pdf_ns", return_value=borb):
        PDFReporter(mock_config).generate((section,))

    paragraph_texts = [call.args[0] for call in borb.Paragraph.call_args_list]

    assert f"Model: {long_model_path}" not in paragraph_texts
    assert "Model: isic2018_efficientNet4_fullmodule.pt" in paragraph_texts
    assert all(
        max((len(token) for token in str(text).split()), default=0) <= _MAX_PDF_TOKEN_LEN
        for text in paragraph_texts
    )


def test_prepare_raster_scales_large_png_to_bounds(tmp_path: Path) -> None:
    from PIL import Image as PILImage

    path = tmp_path / "large.png"
    PILImage.new("RGB", (4000, 3000), color="red").save(path)
    reporting = SimpleNamespace(
        formatting=SimpleNamespace(
            image_raster_multiplier=3.0,
            image_raster_max_edge_px=2400,
        ),
    )
    pil, (dw, dh) = _prepare_raster_for_pdf(
        path, max_width_pt=400, max_height_pt=300, reporting=reporting
    )
    assert (dw, dh) == (400, 300)
    assert pil.mode == "RGB"
    assert pil.size == (1200, 900)


def test_prepare_raster_does_not_upscale_small_png(tmp_path: Path) -> None:
    from PIL import Image as PILImage

    path = tmp_path / "small.png"
    PILImage.new("RGB", (80, 60), color="blue").save(path)
    reporting = SimpleNamespace(
        formatting=SimpleNamespace(
            image_raster_multiplier=3.0,
            image_raster_max_edge_px=2400,
        ),
    )
    pil, (dw, dh) = _prepare_raster_for_pdf(
        path, max_width_pt=400, max_height_pt=300, reporting=reporting
    )
    assert (dw, dh) == (80, 60)
    assert pil.size == (240, 180)
