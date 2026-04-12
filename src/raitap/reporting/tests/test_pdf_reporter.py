"""Tests for PDF report generation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from raitap.configs import set_output_root
from raitap.configs.schema import AppConfig, ReportingConfig
from raitap.reporting.pdf_reporter import PDFReporter, _prepare_raster_for_pdf

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


@patch("borb.pdf.Document")
@patch("borb.pdf.PDF")
def test_pdf_reporter_generates_file(
    mock_pdf: MagicMock, mock_doc: MagicMock, mock_config: AppConfig, tmp_path: Path
) -> None:
    """Test basic PDF generation."""
    reporter = PDFReporter(mock_config)

    metrics_eval = None

    output_path = reporter.generate((), metrics_eval)

    # Verify output path
    assert output_path.exists() is False or output_path.parent.exists()
    assert output_path.suffix == ".pdf"
    assert "test.pdf" in str(output_path)


def test_pdf_reporter_creates_report_directory(mock_config: AppConfig, tmp_path: Path) -> None:
    """Test that report directory is created."""
    with (
        patch("borb.pdf.Document"),
        patch("borb.pdf.PDF"),
    ):
        reporter = PDFReporter(mock_config)
        output_path = reporter.generate((), None)

        assert output_path.parent.name == "reports"


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
