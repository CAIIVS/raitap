# Transparency Test Layout

Use two lanes:

- Fast local and CI coverage: `uv run pytest -m "not e2e"`
- Heavy PR-only transparency coverage: `uv run pytest -m e2e -v --tb=long --mpl`

## E2E Split

- Shared matrix rows live in `e2e_case_matrix.py`
- Broad automatic behavior coverage lives in `test_e2e_transparency_matrix.py`
- Smaller automatic visual-regression coverage lives in `test_e2e_mpl_baseline.py`

## Where To Add Things

- Add new heavy transparency combinations as rows in `e2e_case_matrix.py`
- Add committed PNG baselines to `mpl_baseline/`

## Baseline Rule

Do not auto-create or auto-accept new baselines during normal test runs.

If `src/raitap/transparency/tests/mpl_baseline/captum_ig_image_heat_map.png` is missing, stop and generate or regenerate a candidate locally using the command below, or ask a maintainer for the approved baseline. The approved baseline should then be committed into `mpl_baseline/`.

Suggested candidate-generation command:

```bash
uv run pytest src/raitap/transparency/tests/test_e2e_mpl_baseline.py -m e2e --mpl-generate-path=src/raitap/transparency/tests/mpl_baseline_candidate -v
```
