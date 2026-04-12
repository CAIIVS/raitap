from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from raitap.configs import resolve_run_dir

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType

    from raitap.metrics.factory import MetricsEvaluation
    from raitap.transparency.results import ExplanationResult

from .base_reporter import BaseReporter

logger = logging.getLogger(__name__)

# A4 default in borb (points). SingleColumnLayout uses ~10% side margins.
_A4_WIDTH_PT = 595
_A4_HEIGHT_PT = 842
_MARGIN_FRAC = 0.1


def _borb_pdf_ns() -> SimpleNamespace:
    """Load borb only when generating a PDF (optional ``reporting`` extra)."""
    from borb.pdf import (
        PDF,
        Chart,
        Document,
        FixedColumnWidthTable,
        Image,
        Page,
        Paragraph,
        SingleColumnLayout,
    )

    return SimpleNamespace(
        Chart=Chart,
        Document=Document,
        FixedColumnWidthTable=FixedColumnWidthTable,
        Image=Image,
        PDF=PDF,
        Page=Page,
        Paragraph=Paragraph,
        SingleColumnLayout=SingleColumnLayout,
    )


def _column_content_bounds_pt() -> tuple[int, int]:
    """Approximate usable width / full inner height for default SingleColumnLayout margins."""
    w_page, h_page = _A4_WIDTH_PT, _A4_HEIGHT_PT
    m_x = int(w_page * _MARGIN_FRAC)
    m_y = int(h_page * _MARGIN_FRAC)
    w_avail = w_page - 2 * m_x
    h_inner = h_page - 2 * m_y
    return w_avail, h_inner


_DEFAULT_RASTER_MULT = 3.0
_DEFAULT_RASTER_MAX_EDGE_PX = 2400


def _raster_multiplier(reporting: Any) -> float:
    raw = getattr(reporting, "pdf_image_raster_multiplier", None)
    if raw is None:
        return _DEFAULT_RASTER_MULT
    return max(1.0, float(raw))


def _raster_max_edge_px(reporting: Any) -> int:
    raw = getattr(reporting, "pdf_image_raster_max_edge_px", None)
    if raw is None:
        return _DEFAULT_RASTER_MAX_EDGE_PX
    return max(400, int(raw))


def _prepare_raster_for_pdf(
    path: Path,
    max_width_pt: int,
    max_height_pt: int,
    *,
    reporting: Any,
) -> tuple[PILImageType, tuple[int, int]]:
    """
    Produce a sharp RGB bitmap and the **display size in PDF points** for borb.

    If pixels and points are equal, figures look ~72 DPI. borb's ``Image`` accepts a separate
    ``size=`` (layout points) while the bitmap may have more pixels, so viewers scale cleanly.
    """
    from PIL import Image as PILImage

    path = Path(path)
    mult = _raster_multiplier(reporting)
    max_edge = _raster_max_edge_px(reporting)

    with PILImage.open(path) as opened:
        im = opened.convert("RGBA")
        w, h = im.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image dimensions in {path}")

        scale_fit = min(max_width_pt / w, max_height_pt / h, 1.0)
        disp_w = max(1, int(w * scale_fit))
        disp_h = max(1, int(h * scale_fit))

        pix_w = max(1, int(disp_w * mult))
        pix_h = max(1, int(disp_h * mult))
        edge = max(pix_w, pix_h)
        if edge > max_edge:
            r = max_edge / edge
            pix_w = max(1, int(pix_w * r))
            pix_h = max(1, int(pix_h * r))

        im_resized = im.resize((pix_w, pix_h), PILImage.Resampling.LANCZOS)
        rgb = PILImage.new("RGB", im_resized.size, (255, 255, 255))
        rgb.paste(im_resized, mask=im_resized.split()[3])
        return rgb, (disp_w, disp_h)


def _image_limits_for_transparency(
    reporting: Any, *, num_explainers: int, num_pngs: int
) -> tuple[int, int]:
    """Derive max image width/height from reporting config and optional page budget."""
    base_w, h_inner = _column_content_bounds_pt()
    # Heading + margins: leave ~18% of column height for titles / spacing.
    base_h = int(h_inner * 0.82)

    w_lim = getattr(reporting, "pdf_max_image_width_pt", None)
    h_lim = getattr(reporting, "pdf_max_image_height_pt", None)
    max_w = int(w_lim) if w_lim is not None else base_w
    max_h = int(h_lim) if h_lim is not None else base_h

    max_pages = getattr(reporting, "pdf_transparency_max_pages", None)
    if max_pages is not None and max_pages > 0:
        blocks = num_explainers + max(num_pngs, 1)
        est_pages = max(1, math.ceil(blocks / 2.5))
        if est_pages > max_pages:
            factor = max_pages / est_pages
            max_w = max(200, int(max_w * math.sqrt(factor)))
            max_h = max(110, int(max_h * factor))

    return max_w, max_h


class PDFReporter(BaseReporter):
    """PDF report generator using borb library."""

    def generate(
        self,
        transparency_outputs: dict[str, ExplanationResult],
        metrics_evaluation: MetricsEvaluation | None,
    ) -> Path:
        """Generate PDF report."""
        b = _borb_pdf_ns()

        run_dir = resolve_run_dir(self.config, subdir="reports")
        run_dir.mkdir(parents=True, exist_ok=True)

        filename = getattr(self.config.reporting, "filename", "report.pdf")
        output_path = run_dir / filename

        doc = b.Document()

        self._add_cover_page(doc, b)

        if metrics_evaluation is not None:
            self._add_metrics_section(doc, metrics_evaluation, b)

        if transparency_outputs:
            self._add_transparency_section(doc, transparency_outputs, b)

        b.PDF.write(what=doc, where_to=output_path)

        logger.info("PDF report written to: %s", output_path)
        return output_path

    def _add_cover_page(self, doc: Any, b: SimpleNamespace) -> None:
        """Add cover page with experiment info."""
        page = b.Page()
        doc.append_page(page)
        layout = b.SingleColumnLayout(page)

        layout.append_layout_element(
            b.Paragraph(
                "RAITAP Transparency Report",
                font_size=24,
                font="Helvetica-Bold",
            )
        )

        layout.append_layout_element(b.Paragraph(f"Experiment: {self.config.experiment_name}"))
        layout.append_layout_element(
            b.Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        )
        layout.append_layout_element(
            b.Paragraph(f"Model: {getattr(self.config.model, 'source', 'N/A')}")
        )
        layout.append_layout_element(b.Paragraph(f"Dataset: {self.config.data.name}"))

    def _add_metrics_section(
        self, doc: Any, metrics_eval: MetricsEvaluation, b: SimpleNamespace
    ) -> None:
        """Add metrics section with charts."""
        page = b.Page()
        doc.append_page(page)
        layout = b.SingleColumnLayout(page)

        layout.append_layout_element(b.Paragraph("Metrics", font_size=20, font="Helvetica-Bold"))

        table_data = [["Metric", "Value"]]
        for name, value in metrics_eval.result.metrics.items():
            table_data.append([str(name), f"{float(value):.4f}"])

        table = b.FixedColumnWidthTable(
            number_of_rows=len(table_data),
            number_of_columns=2,
        )
        for row in table_data:
            for cell in row:
                table.append_layout_element(b.Paragraph(str(cell)))

        layout.append_layout_element(table)

        try:
            figures = metrics_eval.create_visualizations()
            max_w, max_h = _column_content_bounds_pt()
            chart_h = min(300, int(max_h * 0.55))
            chart_w = min(450, max_w)
            for chart_name, fig in figures.items():
                layout.append_layout_element(b.Paragraph(chart_name.replace("_", " ").title()))
                plt.figure(fig.number)
                layout.append_layout_element(b.Chart(plt, size=(chart_w, chart_h)))
                plt.close(fig)
        except Exception as e:
            logger.warning("Failed to generate metrics charts: %s", e)
            layout.append_layout_element(b.Paragraph(f"(Chart generation failed: {e})"))

    def _add_transparency_section(
        self,
        doc: Any,
        transparency_outputs: dict[str, ExplanationResult],
        b: SimpleNamespace,
    ) -> None:
        """Add transparency visualisations in one flowing layout (multi-page as needed)."""
        reporting = self.config.reporting
        num_explainers = len(transparency_outputs)
        num_pngs = sum(
            len(list(result.run_dir.glob("*.png"))) for result in transparency_outputs.values()
        )
        max_w, max_h = _image_limits_for_transparency(
            reporting,
            num_explainers=num_explainers,
            num_pngs=num_pngs,
        )

        page = b.Page()
        doc.append_page(page)
        layout = b.SingleColumnLayout(page)

        layout.append_layout_element(
            b.Paragraph("Transparency", font_size=20, font="Helvetica-Bold")
        )

        for explainer_name, result in transparency_outputs.items():
            layout.append_layout_element(
                b.Paragraph(
                    f"Explainer: {explainer_name}",
                    font_size=16,
                    font="Helvetica-Bold",
                )
            )

            viz_files = sorted(result.run_dir.glob("*.png"))
            if not viz_files:
                layout.append_layout_element(b.Paragraph("(No visualizations found)"))
                continue

            for png_file in viz_files:
                try:
                    pil_image, display_pt = _prepare_raster_for_pdf(
                        png_file, max_w, max_h, reporting=reporting
                    )
                    layout.append_layout_element(b.Image(pil_image, size=display_pt))
                except Exception as e:
                    logger.warning("Failed to add image %s: %s", png_file, e)
                    layout.append_layout_element(b.Paragraph(f"(Failed to load: {png_file.name})"))
