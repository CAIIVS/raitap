# Output

## Structure

Generated PDF reports contain the following sections:

### 1. Cover Page

The cover page displays:

- Experiment name
- Generation timestamp
- Model source
- Dataset name

### 2. Metrics Section

If metrics are enabled in the pipeline, this section includes:

- **Metrics Table** - Key-value table of all scalar metrics
- **Metrics Overview Chart** - Bar chart visualisation of metric values
- **Confusion Matrix** - If available in metric artifacts

### 3. Figure sections

The pipeline passes one or more **sections**; each has a title and ordered **groups**. Each group has a heading and raster figures matched under its output directory (by default `*.png`).

The default assessment run includes a **Transparency** section: for each configured explainer, a group lists the PNGs written to that explainer’s run directory. Additional modules (for example robustness) can supply further sections the same way.

Groups with no matching files are skipped (no placeholder text). If every group in a section is empty, the section title is not emitted either.
