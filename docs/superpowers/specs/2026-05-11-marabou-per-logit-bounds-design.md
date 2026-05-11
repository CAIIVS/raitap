# Per-logit output bounds for MarabouAssessor

**Issue:** [#131](https://github.com/CAIIVS/raitap/issues/131)
**Branch:** `131-implement-per-logit-output-bounds-for-marabouassessor`
**Date:** 2026-05-11

## Goal

Populate `VerificationOutcome.lower_bounds` / `upper_bounds` (per-class logit
ranges) for each verified sample in `MarabouAssessor`, so that
`RobustnessResult.output_bounds` is non-empty and reporting can render a
certified min/max logit table.

## Background

- `MarabouAssessor` (v1, PR #134) hard-codes `lower_bounds=None` /
  `upper_bounds=None` in `verify_sample`.
- The pipeline already stacks per-sample bounds into
  `RobustnessResult.output_bounds = {"lower": Tensor[N,K], "upper": Tensor[N,K]}`
  via `_stack_optional_bounds` in `base_assessor.py` (NaN-padded for samples
  that don't expose bounds).
- Marabou 2.0 / `maraboupy` exposes **no native min/max objective** on
  `MarabouNetwork`. Only `solve()` (SAT/UNSAT), `setLowerBound`,
  `setUpperBound`, `addInequality`, `addDisjunctionConstraint`.

## Approach: bisection-via-SAT

The issue's pseudocode ("minimize / maximize → solve → record") is not directly
expressible in maraboupy. The equivalent is **bisection on bounds via repeated
SAT queries**, which is the standard idiom for Marabou bound extraction:

For each output class `k` of a verified sample:

1. **Lower bound search.** Find smallest `c_lo` such that the box query
   "input ∈ [x−eps, x+eps] ∧ out[k] < c_lo" is UNSAT.
   - Fresh `Marabou.read_onnx(...)`, apply input box, set
     `network.setUpperBound(out[k], c_lo)`, `solve()`.
   - UNSAT → `out[k] ≥ c_lo` is certified; ratchet `c_lo` up.
   - SAT   → `out[k] < c_lo` is realisable; ratchet `c_lo` down.
   - Binary-search on `c_lo` in `[-search_range, +search_range]` until the
     window narrows below `tolerance`.

2. **Upper bound search.** Symmetric, using `network.setLowerBound(out[k], c_hi)`.

Returns `(lower_bounds[k], upper_bounds[k])` as a `torch.Tensor` of length `K`.

### Why bisection, not literal "minimize"

`MarabouNetwork.solve()` is decision-only (SAT/UNSAT). The DeepSoI cost
function is internal and not exposed via the Python API. Bisection is the
documented technique for extracting variable ranges from a SAT solver.

### Cost

Per **verified** sample: `K × 2 × ⌈log₂(search_range / tolerance)⌉` Marabou
solves. With `search_range=1e3`, `tolerance=1e-2`, `K=10`: ~340 solves per
sample. Plus K×2 fresh ONNX reads (per spec: "Build a fresh MarabouNetwork
from the same ONNX path"). Far heavier than the issue's "K× more" estimate;
documented as such.

This is acceptable because the feature is **opt-in** (`compute_output_bounds=False`
by default) and FALSIFIED / UNKNOWN / ERROR samples are skipped entirely
(only verified samples get bounds computed — bounds on a non-verified sample
are meaningless because the verdict already implies a counter-example exists).

## API changes

### MarabouAssessor constructor

```python
def __init__(
    self,
    *,
    algorithm: str = "linf-box",
    timeout_s: float = 300.0,
    epsilon: float = 0.05,
    norm: str = "Linf",
    compute_output_bounds: bool = False,
    bound_search_range: float = 1e3,
    bound_tolerance: float = 1e-2,
    **kwargs: Any,
) -> None: ...
```

### verify_sample flow

After the main verdict query:

```python
outcome_kwargs = {"lower_bounds": None, "upper_bounds": None, ...}
if self.compute_output_bounds and verdict == RobustnessVerdict.VERIFIED:
    lower, upper = self._compute_output_bounds(
        onnx_path=onnx_path,
        flat_sample=flat_sample,
        eps=eps,
        num_outputs=output_vars.size,
    )
    outcome_kwargs["lower_bounds"] = lower
    outcome_kwargs["upper_bounds"] = upper
```

Bounds returned as `torch.Tensor` of dtype `float32`, shape `(K,)`.

### New private helpers

- `_compute_output_bounds(onnx_path, flat_sample, eps, num_outputs) -> tuple[Tensor, Tensor]`
- `_bisect_output_bound(onnx_path, flat_sample, eps, output_index, mode) -> float`
  where `mode ∈ {"lower", "upper"}`.

Helpers live as module-level free functions to keep `MarabouAssessor` lean and
to match the existing pattern (`_export_torch_to_onnx`,
`_interpret_solver_result`, `_reconstruct_counter_example`).

## Reporting wiring

`_build_robustness_section` in `src/raitap/reporting/builder.py` already
appends metrics rows to `table_rows`. Add a new helper that, when
`result.output_bounds is not None`, appends per-class bound rows:

- For each class `k ∈ [0, K)`: rows `(f"logit_{k}_lower_mean", "<value>")`
  and `(f"logit_{k}_upper_mean", "<value>")`, averaged across rows that are
  not NaN. Skips classes where every entry is NaN.
- Plus one summary row `("output_bounds_samples", "<n_with_bounds>/<n_total>")`.

Why mean rather than per-sample table: PDF table_rows are flat `(key, value)`
tuples, so per-sample × per-class would be `N×K` rows — unreadable for typical
`N=100, K=10`. Mean preserves the certified-range signal and keeps the table
compact. Per-sample data remains accessible via the saved `RobustnessResult`
artefact for users who want it.

## Testing

New tests in `src/raitap/robustness/assessors/tests/test_marabou_assessor.py`:

1. `test_compute_output_bounds_disabled_by_default` — outcome has
   `lower_bounds is None`, `upper_bounds is None`.
2. `test_compute_output_bounds_populates_tensors_for_verified_sample` —
   with `compute_output_bounds=True`, mock Marabou to alternate UNSAT/SAT in
   a way that drives bisection to a known interval; assert shape `(K,)`,
   dtype `float32`, values within tolerance.
3. `test_compute_output_bounds_skipped_for_falsified_sample` — even with
   the flag on, `lower_bounds is None` when verdict is FALSIFIED.
4. `test_compute_output_bounds_extra_solves_count` — verify the expected
   number of Marabou `read_onnx + solve` invocations per sample
   (`2K × bisect_depth`) to lock in the cost contract.

New test in `src/raitap/reporting/tests/test_formal_section.py`:

5. `test_robustness_section_renders_logit_bounds_when_present` — populate
   `output_bounds` on the fixture result and assert `logit_0_lower_mean`,
   `logit_0_upper_mean`, etc., appear in `group.table_rows`.

All Marabou tests reuse the existing `_FakeNetwork` / `fake_maraboupy`
fixture pattern that monkeypatches `sys.modules`.

## Documentation

Update `docs/modules/robustness/output.md` (or the Marabou-specific section,
whichever exists) with:

- New constructor kwarg table entries (`compute_output_bounds`,
  `bound_search_range`, `bound_tolerance`).
- Cost note: "Enabling `compute_output_bounds` adds `2K × log₂(range/tol)`
  Marabou solves per **verified** sample. Default settings → ~340 extra
  solves/sample for K=10."

## Out of scope

- Caching `MarabouNetwork` instances across bisection steps — spec calls for
  fresh networks; revisit only if profiling justifies it.
- Bounds on FALSIFIED / UNKNOWN / ERROR samples (verdict already implies
  these are not certifiable).
- Visualisation (matplotlib bar chart of per-class bounds) — text table
  satisfies the acceptance criterion; image rendering can land in a
  follow-up issue.
- Migration of existing v1 `MarabouAssessor` test fixtures — additive
  changes only.

## Acceptance mapping

| Issue criterion | Where addressed |
|---|---|
| `compute_output_bounds=True` populates `output_bounds` | `_compute_output_bounds` + verify_sample wiring |
| Reporting renders min/max logit table for FORMAL_VERIFICATION | builder.py addition + new test |
| Unit tests cover bounds extraction with mocked Marabou min/max | tests 1–4 |
| Documentation notes K× runtime cost | `docs/modules/robustness/output.md` update |
