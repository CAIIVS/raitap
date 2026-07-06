# Text-classification (SST-2 sentiment) smoke config

End-to-end check for the text input modality (#340): loads a HuggingFace
sentiment model + tokenizer, tokenises a small text CSV into `input_ids` +
`attention_mask`, runs the forward pass + metrics, then `LayerIntegratedGradients`
token attribution over the embedding layer and renders per-token importances
with `CaptumTextVisualiser`.

This is the keystone that proves the text path works: load -> tokenise ->
predict -> token attribution -> HTML report with per-token panels.

## Run

Artifacts (`artifacts/reviews.csv`, `artifacts/labels.csv`) are gitignored
(`contributor-configs/*/artifacts/`), so generate them first:

```bash
uv run --extra text python contributor-configs/text-classification-sst2/build_data.py
```

Then run the assessment:

```bash
uv run raitap --config-dir contributor-configs/text-classification-sst2 --config-name assessment
```

`raitap-deps` infers the `text` (transformers), `captum`, `metrics`, and `html`
extras and installs them. The model
(`distilbert-base-uncased-finetuned-sst-2-english`, ~270 MB) downloads from the
HuggingFace hub on first run. The HTML report lands under
`outputs/<date>/<time>/reports/` and embeds one per-token attribution panel per
review.

## Notes

- `model.tokenizer` selects the text modality: the loader uses
  `AutoModelForSequenceClassification` + `AutoTokenizer` and the data layer
  tokenises the `text` column.
- `transparency.token_ig.constructor.layer_path: distilbert.embeddings` points
  `LayerIntegratedGradients` at the embedding module (dotted getattr path
  resolved on the loaded model). `call.target: 1` attributes the positive class.
- Attributions are the per-token scores after collapsing the embedding axis
  (canonical Captum text reduction, summed over the hidden dimension).
- Text has no per-sample image thumbnail, so the reporter prints a benign
  "skipping sample thumbnail" warning per pinned sample.
- Batch token rendering is one bar panel per review; richer inline token
  highlighting is follow-on polish (#99).

## Regenerate artifacts

`build_data.py` writes both CSVs (6 short hand-labelled reviews, 0 = negative,
1 = positive; labels row-aligned to reviews). Edit the `ROWS` list to change the
sample set, then re-run the command above.
