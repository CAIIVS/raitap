# Tree / XGBoost + SHAP TreeExplainer smoke config

Minimal end-to-end check for the tree/tabular model backend (#246): loads a
fitted XGBoost model (`.ubj`), runs `shap.TreeExplainer` over a tabular feature
matrix, and renders the attributions with the tabular `ShapBar` visualiser.

## Run

Artifacts (`artifacts/model.ubj`, `artifacts/features.csv`) are gitignored
(`contributor-configs/*/artifacts/`), so generate them first:

```bash
uv run --extra xgboost python contributor-configs/tree-xgboost-shap/build_model.py
```

Then run the assessment:

```bash
uv run raitap --config-dir contributor-configs/tree-xgboost-shap --config-name assessment
```

`raitap-deps` infers the `xgboost` (+ `shap`, `html`) extras and installs them.
The `xgboost` extra bundles scikit-learn (the backend loads via the sklearn-API
`XGBClassifier`) and CPU torch (raitap's tensor pipeline needs it, even though
XGBoost does the compute). The HTML report lands under
`outputs/<date>/<time>/reports/`.

## Notes

- Transparency-only (no metrics/labels) to keep it a focused backend smoke.
- `call.target: 1` picks the positive class — binary XGBoost SHAP returns a
  stacked `(B, F, 2)`, one slice per class.
- Tabular input has no per-sample image thumbnail, so the reporter prints a
  benign "skipping sample thumbnail" warning per pinned sample (tracked in #136).
