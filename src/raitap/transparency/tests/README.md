# Transparency Test Layout

## Lanes (markers)

| Marker | Lane | Selector |
| --- | --- | --- |
| _(none)_ | Quality Gate (every push, fast) | `uv run pytest -m "not e2e and not cuda"` |
| `e2e` | E2E / Transparency (PR-gated) | `uv run pytest -m e2e -v --tb=long --mpl` |
| `visual` | E2E / Transparency → Visual regression step | `uv run pytest -m "e2e and visual" --mpl` |
| `parity` | E2E / Transparency → Parity step | `uv run pytest -m "e2e and parity"` |
| `slow` | Quality Gate (opt-out) | `uv run pytest -m "not slow"` to skip |

Full marker list lives in `pyproject.toml` `[tool.pytest.ini_options]`.

## E2E Split

- Shared matrix rows live in `e2e_case_matrix.py`
- Broad automatic behavior coverage lives in `test_e2e_transparency_matrix.py`
- Visual-regression coverage lives in `test_e2e_mpl_baseline.py` (cases with an
  `mpl_baseline_filename` in the matrix)

## Where To Add Things

- New heavy combinations → rows in `e2e_case_matrix.py`
- Pixel baselines → `mpl_baseline/` (see Baseline Rule)
- Shared fakes/config/dep-gating → reuse `raitap.testing`
  (`make_tiny_classifier`, `make_app_config`, `requires(...)`), don't re-roll

## Baseline Rule

Baselines are pixel-compared with `remove_text=True` (heatmap drift is caught;
cosmetic title/label churn is ignored) and are **FreeType/Agg-sensitive** — they
are only valid on the CI image (`ubuntu-24.04`, pinned matplotlib). The suite
`skipif`s off non-Linux, so do not regenerate locally on Windows/macOS.

To add or refresh a baseline:

1. Give the matrix case an `mpl_baseline_filename` (and `mpl_use_deterministic_inputs=True`).
2. Run the **Regenerate Visual Baselines** workflow (`workflow_dispatch`,
   `.github/workflows/regen-baselines.yml`).
3. Download the `mpl-baselines` artifact and commit the PNGs into `mpl_baseline/`.

Do not auto-create or auto-accept baselines during normal test runs.
