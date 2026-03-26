---
name: Codebase Architecture Audit
overview: Comprehensive analysis of the RAITAP codebase identifying critical bugs, architectural issues, code quality problems, and Python best practices violations across all modules.
todos: []
isProject: false
---

# RAITAP Codebase Analysis & Refactoring Plan

## Executive Summary

Your codebase has **solid fundamentals** (good use of protocols, type hints, Hydra composition) but suffers from:

- **1 critical bug** that breaks functionality
- **Configuration/schema mismatches** between tests and runtime
- **Code duplication** and inconsistent patterns
- **Non-Pythonic practices** (coming from TypeScript/Java/C#)
- **Dead code and broken entry points**

Total issues found: **36** (1 critical, 11 medium, 24 low)

---

## Prioritized Implementation Backlog

Use this as the single source of truth. One issue per number, continuous numbering.

### Critical (High)

1. Apply metrics prefix in MLflow tracker (`log_metrics` ignores `prefix`).
  File: `[src/raitap/tracking/mlflow/mlflow_tracker.py](d:\Repos\ZHAW\BA\raitap\src\raitap\tracking\mlflow\mlflow_tracker.py)`

### Medium

1. Replace `print()` with logging in data module.
  File: `[src/raitap/data/data.py](d:\Repos\ZHAW\BA\raitap\src\raitap\data\data.py)`
2. Replace `print()` with logging in metrics factory.
  File: `[src/raitap/metrics/factory.py](d:\Repos\ZHAW\BA\raitap\src\raitap\metrics\factory.py)`
3. Replace `print()` with logging in MLflow tracker.
  File: `[src/raitap/tracking/mlflow/mlflow_tracker.py](d:\Repos\ZHAW\BA\raitap\src\raitap\tracking\mlflow\mlflow_tracker.py)`
4. Harden subprocess lifecycle for MLflow server startup (ownership, readiness, cleanup).
  File: `[src/raitap/tracking/mlflow/mlflow_tracker.py](d:\Repos\ZHAW\BA\raitap\src\raitap\tracking\mlflow\mlflow_tracker.py)`
5. Remove duplicated `__enter__`/`__exit__` from `MLFlowTracker`.
  File: `[src/raitap/tracking/mlflow/mlflow_tracker.py](d:\Repos\ZHAW\BA\raitap\src\raitap\tracking\mlflow\mlflow_tracker.py)`
6. Fix smoke test repo-root calculation (`parents[3]` currently points to `src`).
  File: `[src/raitap/tracking/tests/smoke_test_mlflow.py](d:\Repos\ZHAW\BA\raitap\src\raitap\tracking\tests\smoke_test_mlflow.py)`
7. Deduplicate repeated mlflow import error handling (`try: import mlflow` blocks).
  File: `[src/raitap/tracking/mlflow/mlflow_tracker.py](d:\Repos\ZHAW\BA\raitap\src\raitap\tracking\mlflow\mlflow_tracker.py)`
8. Deduplicate `TestTensorToPython` tests across metrics test files.
  Files: `[src/raitap/metrics/tests/test_classification_metrics.py](d:\Repos\ZHAW\BA\raitap\src\raitap\metrics\tests\test_classification_metrics.py)`, `[src/raitap/metrics/tests/test_detection_metrics.py](d:\Repos\ZHAW\BA\raitap\src\raitap\metrics\tests\test_detection_metrics.py)`
9. Rename/fix misleading data test name (`test_describe_omits_sample_shape_for_1d_tensors`).
  File: `[src/raitap/data/tests/test_data_class.py](d:\Repos\ZHAW\BA\raitap\src\raitap\data\tests\test_data_class.py)`
10. Rename/fix misleading detection backend test name (`test_backend_pycocotools`).
  File: `[src/raitap/metrics/tests/test_detection_metrics.py](d:\Repos\ZHAW\BA\raitap\src\raitap\metrics\tests\test_detection_metrics.py)`
11. Rename/fix misleading model loading test name (`test_model_loads_from_local_pth_file`).
  File: `[src/raitap/models/tests/test_model_class.py](d:\Repos\ZHAW\BA\raitap\src\raitap\models\tests\test_model_class.py)`

### Low

1. Normalize remaining non-Pythonic naming remnants.
2. Tighten `**kwargs: Any` typing where practical.
3. Add missing return type hints on remaining helper/error-handling functions.
4. Fix minor docstring/type annotation mismatches.
5. Validate and reject whitespace-only `source` values.
6. Add Parquet loading test coverage.
7. Add end-to-end URL-source tests for `Data`.
8. Add tests for download/network failure paths.
9. Reduce overlap between `test_data.py` and `test_data_class.py`.
10. Improve `_load_pretrained` unhappy-path coverage/clarity.
11. Replace broad `except Exception` with narrower exceptions where known.
12. Align `BaseVisualiser.visualise` docstring with accepted input types.
13. Add isolated tests for `create_explainer()`.
14. Add isolated tests for `check_explainer_visualiser_compat()`.
15. Add isolated tests for `resolve_default_run_dir()`.
16. Add isolated tests for `_serialisable()` edge cases.
17. Add isolated tests for `ExplanationResult.log()` edge cases.
18. Fix README test command path (`pytest tests/transparency/` is outdated).
19. Fix transparency README reference to nonexistent `test_integration.py`.
20. Clarify/align `AppConfig.transparency` typing with runtime shape.
21. Consolidate overlapping model test files (`test_model_class.py` vs `test_models.py`).
22. Remove duplicated model test scenarios (e.g., `test_accepts_string_path`).
23. Remove broken Sphinx references in docs.
24. Audit and remove stale test helper configs with wrong field names.

