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

### 3. Transparency Section

For each configured explainer:

- Explainer name and metadata
- All generated visualisations (PNG images)
