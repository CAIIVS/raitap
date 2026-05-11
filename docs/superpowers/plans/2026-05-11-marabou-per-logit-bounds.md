# Marabou per-logit output bounds — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate `VerificationOutcome.lower_bounds` / `upper_bounds` (per-class certified logit ranges) in `MarabouAssessor`, opt-in via a constructor flag, and surface them as text rows in the PDF report.

**Architecture:** For each verified sample × output class, run bisection-via-SAT against a fresh `MarabouNetwork` (Marabou exposes no native min/max). Two helpers — `_compute_output_bounds(...)` and `_bisect_output_bound(...)` — live as module-level functions next to the existing `_export_torch_to_onnx` / `_interpret_solver_result` helpers in `marabou_assessor.py`. Bisection probes `setUpperBound(out_var, mid)` (for `lower` mode) or `setLowerBound(out_var, mid)` (for `upper` mode) inside the same input box as the verdict query. The report builder appends `logit_{k}_lower_mean` / `logit_{k}_upper_mean` rows averaged across non-NaN samples.

**Tech Stack:** Python 3.11+, `maraboupy>=2.0` (mocked in tests), `torch`, `numpy`, `pytest`. Run all commands via `uv run …`.

**Spec:** `docs/superpowers/specs/2026-05-11-marabou-per-logit-bounds-design.md`

**Issue:** [#131](https://github.com/CAIIVS/raitap/issues/131) (follow-up visualiser tracked in #141).

**Branch:** `131-implement-per-logit-output-bounds-for-marabouassessor` (worktree `.worktrees/131-marabou-bounds`).

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/raitap/robustness/assessors/marabou_assessor.py` | Modify | Add 3 ctor kwargs; new helpers `_compute_output_bounds`, `_bisect_output_bound`; wire into `verify_sample`. |
| `src/raitap/robustness/assessors/tests/test_marabou_assessor.py` | Modify | Extend `_FakeNetwork` to script per-solve results; 6 new tests. |
| `src/raitap/reporting/builder.py` | Modify | Append per-class bound rows when `result.output_bounds is not None`. |
| `src/raitap/reporting/tests/test_formal_section.py` | Modify | One new test that exercises the bounds-row rendering. |
| `docs/modules/robustness/output.md` | Modify (or create section) | Document new kwargs + runtime cost. |

---

## Task 1 — Constructor kwargs (no-op behaviour, locks contract)

**Files:**
- Modify: `src/raitap/robustness/assessors/marabou_assessor.py` (signature + attribute storage only — no behaviour change yet)
- Test: `src/raitap/robustness/assessors/tests/test_marabou_assessor.py`

- [ ] **Step 1: Write failing test for default-off ctor.** Append to `test_marabou_assessor.py`:

```python
def test_compute_output_bounds_defaults_to_disabled() -> None:
    assessor = MarabouAssessor()
    assert assessor.compute_output_bounds is False
    assert assessor.bound_search_range == 1e3
    assert assessor.bound_tolerance == 1e-2


def test_compute_output_bounds_kwargs_round_trip() -> None:
    assessor = MarabouAssessor(
        compute_output_bounds=True,
        bound_search_range=50.0,
        bound_tolerance=0.05,
    )
    assert assessor.compute_output_bounds is True
    assert assessor.bound_search_range == 50.0
    assert assessor.bound_tolerance == 0.05
```

- [ ] **Step 2: Run, expect AttributeError.**

```
uv run pytest src/raitap/robustness/assessors/tests/test_marabou_assessor.py::test_compute_output_bounds_defaults_to_disabled -x
```

Expected: FAIL — `AttributeError: 'MarabouAssessor' object has no attribute 'compute_output_bounds'`.

- [ ] **Step 3: Add kwargs to `__init__`.** Edit the signature + body in `marabou_assessor.py` (replace the current `__init__`):

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
) -> None:
    del kwargs
    if algorithm not in type(self).algorithm_registry:
        valid = ", ".join(sorted(type(self).algorithm_registry))
        raise ValueError(f"MarabouAssessor: unknown algorithm {algorithm!r}. Known: {valid}.")
    self.algorithm = algorithm
    self.timeout_s = float(timeout_s)
    self.epsilon = float(epsilon)
    self._norm = str(norm)
    self.compute_output_bounds = bool(compute_output_bounds)
    self.bound_search_range = float(bound_search_range)
    self.bound_tolerance = float(bound_tolerance)
    if self.bound_tolerance <= 0:
        raise ValueError("MarabouAssessor: bound_tolerance must be > 0.")
    if self.bound_search_range <= 0:
        raise ValueError("MarabouAssessor: bound_search_range must be > 0.")
    self.init_kwargs: dict[str, Any] = {
        "epsilon": float(epsilon),
        "norm": str(norm),
        "timeout_s": float(timeout_s),
        "compute_output_bounds": bool(compute_output_bounds),
    }
    self._onnx_cache: dict[tuple[int, tuple[int, ...]], Path] = {}
    self._export_logged: bool = False
```

- [ ] **Step 4: Run tests, expect PASS.**

```
uv run pytest src/raitap/robustness/assessors/tests/test_marabou_assessor.py -x
```

Expected: PASS (all previously-passing tests still pass, two new tests green).

- [ ] **Step 5: Commit.**

```bash
git add src/raitap/robustness/assessors/marabou_assessor.py \
        src/raitap/robustness/assessors/tests/test_marabou_assessor.py
git commit -m "feat(robustness): add output-bounds kwargs to MarabouAssessor"
```

---

## Task 2 — Extend `_FakeNetwork` for scripted per-solve results

**Files:**
- Modify: `src/raitap/robustness/assessors/tests/test_marabou_assessor.py`

Bisection needs different `solve()` answers for different probes. Extend the fake so a test can queue answers or pass a callable.

- [ ] **Step 1: Edit `_FakeNetwork`** — replace `solve_result` attribute + `solve` method with:

```python
    self.solve_results: list[tuple[str, dict[int, float], object]] = []
    # Back-compat: tests that set `solve_result` keep working.
    self.solve_result: tuple[str, dict[int, float], object] = (
        "unsat", {}, _FakeStats(0.123),
    )
    self.solve_calls: list[dict[str, Any]] = []  # records (lower_bounds, upper_bounds) snapshots

def solve(self, options: object | None = None) -> tuple[str, dict[int, float], object]:
    del options
    self.solve_calls.append(
        {
            "lower_bounds": dict(self.lower_bounds),
            "upper_bounds": dict(self.upper_bounds),
        }
    )
    if self.solve_results:
        return self.solve_results.pop(0)
    return self.solve_result
```

- [ ] **Step 2: Run full test module, expect no regressions.**

```
uv run pytest src/raitap/robustness/assessors/tests/test_marabou_assessor.py -x
```

Expected: PASS — existing tests use `solve_result`, new behaviour is additive.

- [ ] **Step 3: Commit.**

```bash
git add src/raitap/robustness/assessors/tests/test_marabou_assessor.py
git commit -m "test(robustness): script per-solve results in MarabouAssessor fake"
```

---

## Task 3 — `_bisect_output_bound` helper

**Files:**
- Modify: `src/raitap/robustness/assessors/marabou_assessor.py`
- Test: `src/raitap/robustness/assessors/tests/test_marabou_assessor.py`

The helper returns the conservative certified bound (lower → lo side, upper → hi side) found by `O(log(range/tol))` SAT probes.

- [ ] **Step 1: Write failing test for `lower` mode** (top of `test_marabou_assessor.py`, near other imports):

```python
from raitap.robustness.assessors.marabou_assessor import _bisect_output_bound


def test_bisect_output_bound_lower_converges_to_true_minimum(
    fake_maraboupy: _FakeNetwork, tmp_path: Any
) -> None:
    """Mock returns UNSAT iff probe `c` is below the true min (0.3)."""
    from maraboupy import Marabou  # noqa: WPS433 — driven by fixture

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    true_min = 0.3
    flat_sample = np.zeros(5, dtype=np.float32)

    def scripted_solve(options: object | None = None) -> tuple[str, dict[int, float], object]:
        # The output var for index 0 has id num_inputs (=5). For lower-mode
        # probes we set setUpperBound(out_var, mid). UNSAT iff mid < true_min.
        del options
        mid = fake_maraboupy.upper_bounds[5]
        fake_maraboupy.solve_calls.append(
            {"upper_bounds": dict(fake_maraboupy.upper_bounds)}
        )
        if mid < true_min:
            return ("unsat", {}, _FakeStats(0.0))
        return ("sat", {}, _FakeStats(0.0))

    fake_maraboupy.solve = scripted_solve  # type: ignore[method-assign]

    bound = _bisect_output_bound(
        onnx_path=onnx_path,
        flat_sample=flat_sample,
        eps=0.05,
        output_index=0,
        mode="lower",
        search_range=1.0,
        tolerance=1e-3,
        timeout_s=1.0,
    )
    assert abs(bound - true_min) <= 1e-3
    # Each probe must re-read the ONNX (per spec).
    assert Marabou.read_onnx.call_count >= 1
```

Also add the symmetric upper-mode test:

```python
def test_bisect_output_bound_upper_converges_to_true_maximum(
    fake_maraboupy: _FakeNetwork, tmp_path: Any
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    true_max = -0.2
    flat_sample = np.zeros(5, dtype=np.float32)

    def scripted_solve(options: object | None = None) -> tuple[str, dict[int, float], object]:
        del options
        mid = fake_maraboupy.lower_bounds[5]
        if mid > true_max:
            return ("unsat", {}, _FakeStats(0.0))
        return ("sat", {}, _FakeStats(0.0))

    fake_maraboupy.solve = scripted_solve  # type: ignore[method-assign]

    bound = _bisect_output_bound(
        onnx_path=onnx_path,
        flat_sample=flat_sample,
        eps=0.05,
        output_index=0,
        mode="upper",
        search_range=1.0,
        tolerance=1e-3,
        timeout_s=1.0,
    )
    assert abs(bound - true_max) <= 1e-3
```

- [ ] **Step 2: Run, expect ImportError.**

```
uv run pytest src/raitap/robustness/assessors/tests/test_marabou_assessor.py::test_bisect_output_bound_lower_converges_to_true_minimum -x
```

Expected: FAIL with `ImportError: cannot import name '_bisect_output_bound'`.

- [ ] **Step 3: Implement helper at module level.** Append to `marabou_assessor.py` (after `_interpret_solver_result`):

```python
def _bisect_output_bound(
    *,
    onnx_path: Path,
    flat_sample: np.ndarray,
    eps: float,
    output_index: int,
    mode: str,
    search_range: float,
    tolerance: float,
    timeout_s: float,
) -> float:
    """Return certified per-logit bound via bisection-via-SAT.

    ``mode='lower'``: largest c such that out[k] ≥ c is provable.
    ``mode='upper'``: smallest c such that out[k] ≤ c is provable.

    Per spec, a fresh ``MarabouNetwork`` is built for every probe.
    """
    if mode not in {"lower", "upper"}:
        raise ValueError(f"_bisect_output_bound: mode must be 'lower'|'upper', got {mode!r}")

    from maraboupy import Marabou  # type: ignore[import-not-found]

    lo, hi = -float(search_range), float(search_range)
    options = Marabou.createOptions(timeoutInSeconds=int(timeout_s), verbosity=0)
    while (hi - lo) > tolerance:
        mid = (lo + hi) / 2.0
        network = Marabou.read_onnx(str(onnx_path))
        input_vars = np.asarray(network.inputVars[0]).reshape(-1)
        output_vars = np.asarray(network.outputVars[0]).reshape(-1)
        for var_id, value in zip(input_vars, flat_sample, strict=True):
            network.setLowerBound(int(var_id), float(value) - eps)
            network.setUpperBound(int(var_id), float(value) + eps)
        out_var = int(output_vars[output_index])
        if mode == "lower":
            network.setUpperBound(out_var, mid)
        else:
            network.setLowerBound(out_var, mid)
        exit_code, _, _ = network.solve(options=options)
        is_unsat = str(exit_code).strip().lower() in {"unsat", "valid"}
        if mode == "lower":
            # UNSAT: no x' with out[k] ≤ mid → min > mid.
            if is_unsat:
                lo = mid
            else:
                hi = mid
        else:
            # UNSAT: no x' with out[k] ≥ mid → max < mid.
            if is_unsat:
                hi = mid
            else:
                lo = mid
    return lo if mode == "lower" else hi
```

- [ ] **Step 4: Run both new tests, expect PASS.**

```
uv run pytest src/raitap/robustness/assessors/tests/test_marabou_assessor.py::test_bisect_output_bound_lower_converges_to_true_minimum src/raitap/robustness/assessors/tests/test_marabou_assessor.py::test_bisect_output_bound_upper_converges_to_true_maximum -x
```

Expected: PASS.

- [ ] **Step 5: Run quality gate.**

```
uv run ruff check src/raitap/robustness/assessors/marabou_assessor.py && uv run ruff format --check src/raitap/robustness/assessors/marabou_assessor.py
```

Expected: PASS. If format fails, run `uv run ruff format src/raitap/robustness/assessors/marabou_assessor.py` and re-stage.

- [ ] **Step 6: Commit.**

```bash
git add src/raitap/robustness/assessors/marabou_assessor.py \
        src/raitap/robustness/assessors/tests/test_marabou_assessor.py
git commit -m "feat(robustness): add _bisect_output_bound helper to MarabouAssessor"
```

---

## Task 4 — `_compute_output_bounds` helper + verify_sample wiring (verdict gating)

**Files:**
- Modify: `src/raitap/robustness/assessors/marabou_assessor.py`
- Test: `src/raitap/robustness/assessors/tests/test_marabou_assessor.py`

- [ ] **Step 1: Write failing tests** (append):

```python
def test_verify_sample_returns_none_bounds_when_flag_disabled(
    fake_maraboupy: _FakeNetwork, tmp_path: Any
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    fake_maraboupy.solve_result = ("unsat", {}, _FakeStats(0.0))

    class _Backend:
        def __init__(self, p: Any) -> None:
            self.onnx_path = p

    assessor = MarabouAssessor(compute_output_bounds=False)
    outcome = assessor.verify_sample(
        model=_IdentityModel(),
        sample=torch.zeros(1, 5),
        target=torch.tensor([0]),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        backend=_Backend(onnx_path),
    )
    assert outcome.verdict == RobustnessVerdict.VERIFIED
    assert outcome.lower_bounds is None
    assert outcome.upper_bounds is None


def test_verify_sample_populates_bounds_when_flag_enabled_and_verified(
    fake_maraboupy: _FakeNetwork, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    # Verdict query: UNSAT (verified). Bisection: stub helper instead of scripting solves.
    fake_maraboupy.solve_result = ("unsat", {}, _FakeStats(0.0))

    fake_bounds = [(-0.1 * k, 0.1 * k) for k in range(5)]
    calls: list[tuple[int, str]] = []

    def fake_bisect(**kwargs: Any) -> float:
        calls.append((int(kwargs["output_index"]), str(kwargs["mode"])))
        lo, hi = fake_bounds[int(kwargs["output_index"])]
        return lo if kwargs["mode"] == "lower" else hi

    monkeypatch.setattr(
        "raitap.robustness.assessors.marabou_assessor._bisect_output_bound",
        fake_bisect,
    )

    class _Backend:
        def __init__(self, p: Any) -> None:
            self.onnx_path = p

    assessor = MarabouAssessor(compute_output_bounds=True)
    outcome = assessor.verify_sample(
        model=_IdentityModel(),
        sample=torch.zeros(1, 5),
        target=torch.tensor([0]),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        backend=_Backend(onnx_path),
    )
    assert outcome.lower_bounds is not None and outcome.upper_bounds is not None
    assert outcome.lower_bounds.shape == (5,)
    assert outcome.upper_bounds.shape == (5,)
    assert outcome.lower_bounds.dtype == torch.float32
    assert torch.allclose(
        outcome.lower_bounds, torch.tensor([lo for lo, _ in fake_bounds])
    )
    assert torch.allclose(
        outcome.upper_bounds, torch.tensor([hi for _, hi in fake_bounds])
    )
    # 2K calls: one lower + one upper per class.
    assert sorted(calls) == sorted(
        [(k, "lower") for k in range(5)] + [(k, "upper") for k in range(5)]
    )


def test_verify_sample_skips_bounds_for_falsified_verdict(
    fake_maraboupy: _FakeNetwork, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    fake_maraboupy.solve_result = ("sat", {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, _FakeStats(0.0))

    def boom(**kwargs: Any) -> float:
        raise AssertionError("bisect must not be called on FALSIFIED")

    monkeypatch.setattr(
        "raitap.robustness.assessors.marabou_assessor._bisect_output_bound", boom
    )

    class _Backend:
        def __init__(self, p: Any) -> None:
            self.onnx_path = p

    assessor = MarabouAssessor(compute_output_bounds=True)
    outcome = assessor.verify_sample(
        model=_IdentityModel(),
        sample=torch.zeros(1, 5),
        target=torch.tensor([0]),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        backend=_Backend(onnx_path),
    )
    assert outcome.verdict == RobustnessVerdict.FALSIFIED
    assert outcome.lower_bounds is None
    assert outcome.upper_bounds is None
```

- [ ] **Step 2: Run, expect failures.**

```
uv run pytest src/raitap/robustness/assessors/tests/test_marabou_assessor.py -x -k "verify_sample_populates_bounds or verify_sample_returns_none_bounds or verify_sample_skips_bounds"
```

Expected: FAIL — the bounds remain `None` in `verify_sample`.

- [ ] **Step 3: Implement `_compute_output_bounds` + wire `verify_sample`.** In `marabou_assessor.py`, add a module-level helper (after `_bisect_output_bound`):

```python
def _compute_output_bounds(
    *,
    onnx_path: Path,
    flat_sample: np.ndarray,
    eps: float,
    num_outputs: int,
    search_range: float,
    tolerance: float,
    timeout_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(lower, upper)`` tensors of shape ``(num_outputs,)``."""
    lower = np.empty(num_outputs, dtype=np.float32)
    upper = np.empty(num_outputs, dtype=np.float32)
    for k in range(num_outputs):
        lower[k] = _bisect_output_bound(
            onnx_path=onnx_path,
            flat_sample=flat_sample,
            eps=eps,
            output_index=k,
            mode="lower",
            search_range=search_range,
            tolerance=tolerance,
            timeout_s=timeout_s,
        )
        upper[k] = _bisect_output_bound(
            onnx_path=onnx_path,
            flat_sample=flat_sample,
            eps=eps,
            output_index=k,
            mode="upper",
            search_range=search_range,
            tolerance=tolerance,
            timeout_s=timeout_s,
        )
    return (
        torch.from_numpy(np.ascontiguousarray(lower)),
        torch.from_numpy(np.ascontiguousarray(upper)),
    )
```

Then, in `verify_sample`, replace the final `return VerificationOutcome(...)` block with:

```python
lower_bounds: torch.Tensor | None = None
upper_bounds: torch.Tensor | None = None
if self.compute_output_bounds and verdict == RobustnessVerdict.VERIFIED:
    lower_bounds, upper_bounds = _compute_output_bounds(
        onnx_path=onnx_path,
        flat_sample=flat_sample,
        eps=eps,
        num_outputs=int(output_vars.size),
        search_range=self.bound_search_range,
        tolerance=self.bound_tolerance,
        timeout_s=self.timeout_s,
    )

return VerificationOutcome(
    verdict=verdict,
    counter_example=counter_example,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    runtime_seconds=float(runtime_seconds),
    diagnostics={"exit_code": str(exit_code)},
)
```

- [ ] **Step 4: Run new tests, expect PASS.**

```
uv run pytest src/raitap/robustness/assessors/tests/test_marabou_assessor.py -x
```

Expected: PASS.

- [ ] **Step 5: Quality gate.**

```
uv run ruff check src/raitap/robustness/assessors/marabou_assessor.py && uv run ruff format --check src/raitap/robustness/assessors/marabou_assessor.py
```

- [ ] **Step 6: Commit.**

```bash
git add src/raitap/robustness/assessors/marabou_assessor.py \
        src/raitap/robustness/assessors/tests/test_marabou_assessor.py
git commit -m "feat(robustness): populate per-logit output bounds in MarabouAssessor"
```

---

## Task 5 — End-to-end stack-through to `RobustnessResult.output_bounds`

**Files:**
- Test: `src/raitap/robustness/assessors/tests/test_marabou_assessor.py`

The pipeline already calls `_stack_optional_bounds` in `base_assessor.py:405`. Confirm with a black-box test that a full `MarabouAssessor.assess()` over 2 samples surfaces `output_bounds = {"lower": Tensor[2,K], "upper": Tensor[2,K]}` with the right NaN-padding when one sample is verified and one is not.

- [ ] **Step 1: Write failing test.** Append:

```python
def test_assess_propagates_output_bounds_to_result(
    fake_maraboupy: _FakeNetwork, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two samples: first VERIFIED with bounds, second FALSIFIED → NaN row."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    verdict_queue = [
        ("unsat", {}, _FakeStats(0.0)),  # sample 0 → VERIFIED
        ("sat", {i: 0.0 for i in range(5)}, _FakeStats(0.0)),  # sample 1 → FALSIFIED
    ]
    fake_maraboupy.solve_results = verdict_queue

    monkeypatch.setattr(
        "raitap.robustness.assessors.marabou_assessor._bisect_output_bound",
        lambda **kw: -1.0 if kw["mode"] == "lower" else 1.0,
    )

    class _Backend:
        def __init__(self, p: Any) -> None:
            self.onnx_path = p

    assessor = MarabouAssessor(compute_output_bounds=True)
    result = assessor.assess(
        model=_IdentityModel(),
        inputs=torch.zeros(2, 5),
        targets=torch.tensor([0, 0]),
        backend=_Backend(onnx_path),
    )
    assert result.output_bounds is not None
    lower = result.output_bounds["lower"]
    upper = result.output_bounds["upper"]
    assert lower.shape == (2, 5)
    assert torch.allclose(lower[0], torch.full((5,), -1.0))
    assert torch.allclose(upper[0], torch.full((5,), 1.0))
    assert torch.isnan(lower[1]).all()
    assert torch.isnan(upper[1]).all()
```

- [ ] **Step 2: Run, expect PASS.** (Logic already in place — this is a regression-pinning test.)

```
uv run pytest src/raitap/robustness/assessors/tests/test_marabou_assessor.py::test_assess_propagates_output_bounds_to_result -x
```

Expected: PASS. If FAIL, re-check Task 4 wiring; do **not** patch the test.

- [ ] **Step 3: Commit.**

```bash
git add src/raitap/robustness/assessors/tests/test_marabou_assessor.py
git commit -m "test(robustness): pin end-to-end output-bounds stacking for MarabouAssessor"
```

---

## Task 6 — Report builder: append per-class bound rows

**Files:**
- Modify: `src/raitap/reporting/builder.py`
- Test: `src/raitap/reporting/tests/test_formal_section.py`

- [ ] **Step 1: Write failing test.** Append to `test_formal_section.py`:

```python
def test_build_report_includes_per_class_bound_rows_for_formal_results(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="marabou_test")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    run_dir = tmp_path / "robustness" / "marabou_linf"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = _formal_result(run_dir)
    # Two verified samples + one NaN row → 3 samples, 5 classes.
    result.output_bounds = {
        "lower": torch.tensor(
            [
                [-1.0, -0.5, -0.3, -0.2, -0.1],
                [-2.0, -1.0, -0.6, -0.4, -0.2],
                [float("nan")] * 5,
            ]
        ),
        "upper": torch.tensor(
            [
                [1.0, 0.5, 0.3, 0.2, 0.1],
                [2.0, 1.0, 0.6, 0.4, 0.2],
                [float("nan")] * 5,
            ]
        ),
    }

    outputs = RunOutputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=torch.zeros(3, 5),
        sample_ids=None,
        robustness_results=[result],
        robustness_visualisations=[],
    )

    report = build_report(config, outputs)
    section = next(s for s in report.sections if s.title == "Robustness")
    rows = dict(section.groups[0].table_rows)
    assert rows["logit_0_lower_mean"] == f"{(-1.0 + -2.0) / 2:.4f}"
    assert rows["logit_0_upper_mean"] == f"{(1.0 + 2.0) / 2:.4f}"
    assert rows["logit_4_lower_mean"] == f"{(-0.1 + -0.2) / 2:.4f}"
    assert rows["output_bounds_samples"] == "2/3"
```

- [ ] **Step 2: Run, expect FAIL** (missing keys).

```
uv run pytest src/raitap/reporting/tests/test_formal_section.py::test_build_report_includes_per_class_bound_rows_for_formal_results -x
```

Expected: FAIL — `KeyError: 'logit_0_lower_mean'`.

- [ ] **Step 3: Implement in `builder.py`.** In `_build_robustness_section`, immediately after the existing `for metric_name, metric_value in result.metrics.as_dict()...` loop and before `staged_images: list[Path] = []`, insert:

```python
        if result.output_bounds is not None:
            lower = result.output_bounds.get("lower")
            upper = result.output_bounds.get("upper")
            if lower is not None and upper is not None and lower.ndim == 2:
                lower_np = lower.detach().cpu().to(torch.float32).numpy()
                upper_np = upper.detach().cpu().to(torch.float32).numpy()
                n_samples = lower_np.shape[0]
                per_sample_has_bounds = (~np.isnan(lower_np).all(axis=1)).sum()
                table_rows.append(
                    (
                        "output_bounds_samples",
                        f"{int(per_sample_has_bounds)}/{n_samples}",
                    )
                )
                for k in range(lower_np.shape[1]):
                    col_lo = lower_np[:, k]
                    col_hi = upper_np[:, k]
                    if np.isnan(col_lo).all() or np.isnan(col_hi).all():
                        continue
                    mean_lo = float(np.nanmean(col_lo))
                    mean_hi = float(np.nanmean(col_hi))
                    table_rows.append((f"logit_{k}_lower_mean", f"{mean_lo:.4f}"))
                    table_rows.append((f"logit_{k}_upper_mean", f"{mean_hi:.4f}"))
```

And at the top of `builder.py`, ensure `import numpy as np` and `import torch` are present (add only if missing — check current imports first).

- [ ] **Step 4: Run new test + previous, expect PASS.**

```
uv run pytest src/raitap/reporting/tests/test_formal_section.py -x
```

Expected: PASS (both the existing test and the new one).

- [ ] **Step 5: Quality gate.**

```
uv run ruff check src/raitap/reporting/builder.py && uv run ruff format --check src/raitap/reporting/builder.py
```

- [ ] **Step 6: Commit.**

```bash
git add src/raitap/reporting/builder.py \
        src/raitap/reporting/tests/test_formal_section.py
git commit -m "feat(reporting): render per-class certified logit bounds in robustness section"
```

---

## Task 7 — Documentation

**Files:**
- Modify: `docs/modules/robustness/output.md` (Marabou section)

- [ ] **Step 1: Read the file** and locate the Marabou subsection (or the assessor-options table).

```
uv run python -c "print(open('docs/modules/robustness/output.md').read())" | head -200
```

- [ ] **Step 2: Append a "Per-logit output bounds" subsection** (or add to the existing Marabou block):

```markdown
### Per-logit output bounds (opt-in)

`MarabouAssessor` can populate `RobustnessResult.output_bounds` with certified
per-class logit ranges for each VERIFIED sample. Enable via the constructor:

| Kwarg | Default | Meaning |
|---|---|---|
| `compute_output_bounds` | `False` | Master switch. When `True`, run bisection-via-SAT after each VERIFIED verdict. |
| `bound_search_range` | `1e3` | Initial probe window `[-range, +range]` per output variable. |
| `bound_tolerance` | `1e-2` | Stop bisection when the certified interval narrows below this. |

**Runtime cost.** Marabou exposes no native min/max objective; bounds are
extracted by binary search on `setUpperBound` / `setLowerBound` of each output
variable. Per verified sample, the assessor runs
`2 × K × ⌈log₂(bound_search_range / bound_tolerance)⌉` extra Marabou solves —
for example, `K=10` classes with default settings ≈ 340 additional solves per
sample. FALSIFIED / UNKNOWN / ERROR samples are skipped (their rows in the
stacked bounds tensor are NaN-padded).

The PDF report's Robustness section gains rows `logit_{k}_lower_mean`,
`logit_{k}_upper_mean` (averaged across samples that have bounds) and
`output_bounds_samples` (count of verified samples with bounds /
total samples). Visualisation of these bounds is tracked separately in
issue #141.
```

- [ ] **Step 3: Commit.**

```bash
git add docs/modules/robustness/output.md
git commit -m "docs(robustness): document MarabouAssessor output-bounds kwargs and cost"
```

---

## Task 8 — Full quality gate

**Files:** none (verification only)

- [ ] **Step 1: Full test suite for touched packages.**

```
uv run pytest src/raitap/robustness src/raitap/reporting --tb=short
```

Expected: all PASS (the pre-existing `test_explanation_visualise_sets_shap_image_default_title_from_algorithm` failure from baseline remains unrelated — shap not installed).

- [ ] **Step 2: Repo-wide ruff.**

```
uv run ruff check . && uv run ruff format --check .
```

Expected: clean. If format complains, run `uv run ruff format .` and amend the most recent commit (single fixup, not a force-push).

- [ ] **Step 3: Run `raitap` smoke** — not required (no CLI change) but verify import still works:

```
uv run python -c "from raitap.robustness.assessors import MarabouAssessor; a = MarabouAssessor(compute_output_bounds=True); print(a.bound_search_range, a.bound_tolerance)"
```

Expected: `1000.0 0.01`.

- [ ] **Step 4: Push branch** (only if user confirms — per user memory, commits are user-driven but this branch is the issue-131 branch and a push to remote is the natural integration point):

Stop here. User will push manually. Print summary of commits to be pushed:

```
git log --oneline origin/131-implement-per-logit-output-bounds-for-marabouassessor..HEAD
```

---

## Self-review

| Spec section | Tasks covering it |
|---|---|
| Bisection-via-SAT algorithm | 3 |
| Opt-in `compute_output_bounds` kwarg + `bound_search_range`, `bound_tolerance` | 1, 4 |
| Verified-only gating (skip FALSIFIED / UNKNOWN / ERROR) | 4 |
| `_compute_output_bounds` returns `(K,)` tensors | 4 |
| End-to-end `output_bounds = {"lower": Tensor[N,K], "upper": Tensor[N,K]}` | 5 |
| Reporting renders min/max logit table | 6 |
| Documentation of K× cost | 7 |
| Unit tests cover bounds extraction with mocked Marabou min/max | 3, 4, 5 |

No placeholders; types consistent (`_bisect_output_bound` signature kwargs match every call site; `_compute_output_bounds` returns `(Tensor, Tensor)` matching `verify_sample` consumption).
