---
name: ci topology cleanup
overview: Split the CI evolution into two phases: first refactor workflow topology and test taxonomy, then expand to broader real hardware suites only for GitHub-hosted platforms that actually exist.
todos:
  - id: phase-1-topology-refactor
    content: Refactor the current workflows into a clear Linux quality gate, a clear Apple runtime validation job, and a shared composite setup action.
    status: pending
  - id: phase-1-runtime-markers
    content: Introduce the runtime marker and retarget accelerator-order tests so Phase 2 can scale by semantic test selection rather than file targeting.
    status: pending
  - id: phase-1-document-boundaries
    content: Encode the boundary that Phase 1 is topology cleanup only and that real hardware expansion is deferred to Phase 2.
    status: pending
  - id: phase-2-cuda-suite
    content: Add a real CUDA GitHub-hosted suite with as much parity as practical with the Linux CPU quality-gate test surface.
    status: pending
  - id: phase-2-mps-suite
    content: Expand Apple Silicon from narrow runtime validation to a broader real MPS-backed test suite with as much parity as practical with the Linux CPU quality-gate test surface.
    status: pending
  - id: phase-2-explicit-xpu-limitation
    content: Preserve XPU coverage as mocked resolver/runtime tests only and document that no real XPU CI is possible without non-hosted infrastructure.
    status: pending
isProject: false
---

# CI And Test Topology Refined Plan

## Decision
Split the work into two phases.

Phase 1 is a topology and taxonomy refactor.
Phase 2 is real hardware CI expansion only for hardware classes that GitHub-hosted runners can actually execute.

Do not build a fake all-accelerator matrix in Phase 1.
Do not promise real XPU CI in Phase 2.

## Hosted-Runner Constraint
This plan assumes no self-hosted infrastructure.

Consequences:
- Linux CPU CI remains the baseline broad suite.
- CUDA can only be expanded in Phase 2 if the repository is allowed to use GitHub-hosted GPU runners.
- Apple MPS can be expanded in Phase 2 using GitHub-hosted Apple Silicon runners.
- Intel XPU cannot gain a real hardware CI suite under this constraint and therefore remains mocked-only coverage.

## Naming Convention
Use names that describe purpose, platform, and scope directly.

Use these exact workflow and job names:
- Workflow name in [`.github/workflows/code-quality.yaml`](.github/workflows/code-quality.yaml): `Quality Gate`
- Linux job name: `Quality Gate (Linux)`
- macOS job name: `Runtime Validation (macOS Apple Silicon)`
- Workflow name in [`.github/workflows/test-e2e.yml`](.github/workflows/test-e2e.yml): `SHAP E2E`
- E2E job name: `SHAP E2E (Linux)`

Do not use vague names like `checks`, `apple-silicon-runtime`, or `shap-e2e` once this refactor is implemented.

## Phase 1: Topology Refactor
Phase 1 prepares Phase 2 by making platform-specific expansion semantic and predictable:
- marker-based runtime selection instead of file targeting
- clear job naming
- shared setup action
- explicit comments about what is mocked versus real

### Phase 1 Target Topology
The end of Phase 1 should leave the repository in this structure:
- one main workflow in [`.github/workflows/code-quality.yaml`](.github/workflows/code-quality.yaml)
- one separate slow workflow in [`.github/workflows/test-e2e.yml`](.github/workflows/test-e2e.yml)
- one Linux job that installs once and then runs all fast quality checks on that same runner
- one macOS job that installs once and then runs only runtime-validation checks on that same runner

This is the conventional compromise for a repo like this:
- Linux remains the broad, fast merge gate.
- Apple Silicon becomes an explicit platform-validation layer instead of an accidental extra job.
- Expensive SHAP E2E remains separate.
- Setup is not duplicated inside each runner job.
- Shared bootstrap logic lives in one local composite action instead of being copied across workflows.

### Phase 1 Main Workflow
Update [`.github/workflows/code-quality.yaml`](.github/workflows/code-quality.yaml) so it contains exactly two jobs:
- `quality-gate-linux`
- `runtime-validation-macos-apple-silicon`

Responsibilities:
- `quality-gate-linux` runs Ruff check, Ruff format check, Pyright, and `pytest -m "not e2e and not runtime"` with coverage.
- `runtime-validation-macos-apple-silicon` runs `pytest -m runtime` only.

### Phase 1 E2E Workflow
Keep [`.github/workflows/test-e2e.yml`](.github/workflows/test-e2e.yml) as a separate workflow with one job only:
- job id: `shap-e2e-linux`
- job name: `SHAP E2E (Linux)`

It continues to run only `pytest -m e2e`.

### Phase 1 Granular Implementation Steps
#### 1. Refactor the main workflow names and job boundaries
Edit [`.github/workflows/code-quality.yaml`](.github/workflows/code-quality.yaml).

Make these exact structural changes:
- Rename the workflow from `Code Quality` to `Quality Gate`.
- Rename job id `checks` to `quality-gate-linux`.
- Rename that job’s display name to `Quality Gate (Linux)`.
- Rename job id `apple-silicon-runtime` to `runtime-validation-macos-apple-silicon`.
- Rename that job’s display name to `Runtime Validation (macOS Apple Silicon)`.
- Keep the existing workflow triggers unchanged.

Keep one install sequence per job:
- checkout
- setup Python 3.13
- install uv
- run one `uv sync ...`
- run all job-specific checks on that already prepared runner

Make these exact simplification changes inside the Linux job:
- Remove the custom `actions/github-script` check-run creation and update steps.
- Remove the shell step that manually re-fails the workflow based on previous step outcomes.
- Remove `continue-on-error: true` from Ruff, format, Pyright, and pytest steps.
- Use normal step failure semantics so the job name `Quality Gate (Linux)` is the primary status surface in GitHub Actions.
- Keep explicit step names for `Ruff`, `Ruff Format`, `Pyright`, and `Pytest`.

Do not split Linux lint and Linux tests into separate jobs in Phase 1.

#### 2. Replace macOS file-targeting with semantic runtime selection
Edit [`.github/workflows/code-quality.yaml`](.github/workflows/code-quality.yaml), [`pyproject.toml`](pyproject.toml), and [`src/raitap/models/tests/test_models.py`](src/raitap/models/tests/test_models.py).

Make these exact test-selection changes:
- Register a new pytest marker named `runtime` in [`pyproject.toml`](pyproject.toml).
- Keep the existing `e2e` marker unchanged.
- Mark only these tests in [`src/raitap/models/tests/test_models.py`](src/raitap/models/tests/test_models.py) with `@pytest.mark.runtime`:
  - `test_torch_gpu_falls_back_to_cpu_with_warning`
  - `test_torch_gpu_selects_cuda_when_available`
  - `test_torch_gpu_selects_mps_when_cuda_unavailable`
  - `test_torch_gpu_prefers_cuda_over_mps`
  - `test_torch_gpu_prefers_mps_over_xpu_when_cuda_unavailable`
  - `test_torch_gpu_selects_xpu_when_cuda_unavailable`
  - `test_onnx_backend_cpu_mode_exposes_cpu_provider`
  - `test_onnx_provider_resolution_prefers_cuda_when_available`
  - `test_onnx_provider_resolution_selects_coreml_when_cuda_unavailable`
  - `test_onnx_provider_resolution_prefers_cuda_over_coreml`
  - `test_onnx_provider_resolution_prefers_coreml_over_openvino`
  - `test_onnx_provider_resolution_falls_back_to_cpu_with_warning`
  - `test_onnx_provider_resolution_uses_cpu_in_cpu_mode`
  - `test_onnx_provider_resolution_selects_openvino_when_cuda_unavailable`
  - `test_torch_backend_exposes_intel_xpu_hardware_label`
  - `test_torch_backend_exposes_apple_mps_hardware_label`
  - `test_onnx_backend_exposes_openvino_hardware_label`
  - `test_onnx_backend_exposes_apple_coreml_hardware_label`
- Do not mark any other test in that file as `runtime`.

Change CI commands to use marker selection instead of filename selection:
- Linux job test command becomes: `uv run pytest -m "not e2e and not runtime" -v --cov=src/raitap --cov-report=term-missing --cov-report=html`
- macOS job test command becomes: `uv run pytest -m runtime -v`
- E2E workflow remains: `uv run pytest -m e2e -x -v --tb=long`

#### 3. Normalize bootstrap conventions across workflows
Edit [`.github/workflows/code-quality.yaml`](.github/workflows/code-quality.yaml), [`.github/workflows/test-e2e.yml`](.github/workflows/test-e2e.yml), and add one local composite action under [`.github/actions/`](.github/actions/).

Make these exact normalization changes:
- Create one local composite action dedicated to Python and uv environment setup.
- Place it exactly at [`.github/actions/setup-raitap-env/action.yml`](.github/actions/setup-raitap-env/action.yml).
- The composite action must perform exactly these steps:
  - `actions/setup-python@v6` with an input-driven Python version
  - `astral-sh/setup-uv@v7` with cache enabled
  - one shell step that executes the exact `uv sync` command passed in as an input string
- The composite action must accept exactly these inputs:
  - `python-version`
  - `sync-command`
- Both [`.github/workflows/code-quality.yaml`](.github/workflows/code-quality.yaml) jobs and [`.github/workflows/test-e2e.yml`](.github/workflows/test-e2e.yml) must call that same composite action instead of inlining setup steps.
- Both workflows must use `actions/checkout@v6`.
- Do not use `uv python install 3.13` anywhere after the refactor.

Keep dependency profiles explicit and different by purpose:
- Linux quality gate install command stays the full fast-check environment already used in the main workflow.
- macOS runtime-validation install command stays `uv sync --group dev --extra torch-cpu --extra mlflow --extra shap --extra captum --extra metrics --extra onnx-cpu`.
- SHAP E2E install command stays the lighter `uv sync --group dev --extra torch-cpu --extra shap`.

Clarification:
- The composite action deduplicates workflow code only.
- It does not make Linux and macOS share a prepared environment.
- Each job still runs setup once on its own runner, which is required because GitHub Actions jobs are isolated per runner machine.

#### 4. Encode the Phase 1 boundary explicitly
Document and preserve these semantics with workflow-level comments where the current files already contain explanatory comments:
- `Quality Gate` means fast merge-gating checks.
- `Runtime Validation (macOS Apple Silicon)` means resolver and hardware-label validation for Apple paths only.
- `SHAP E2E` means slow, expensive transparency end-to-end validation and is not part of the fast gate.

Add one short comment in the macOS workflow job stating:
- real MPS/CoreML smoke tests are deferred to Phase 2
- current macOS validation remains mocked resolver-order and backend-label coverage plus install/runtime import validation
- Intel XPU real-platform validation is out of scope because no hosted runner is assumed available

### Phase 1 Expected End State
After Phase 1:
- anyone reading GitHub Actions sees one fast gate, one Apple runtime validation job, and one separate SHAP E2E workflow
- the Apple job no longer looks bolted on because its name and marker target explain its role
- setup still runs only once inside each job on each runner machine, but the YAML for that setup lives in one composite action
- test selection is semantic (`runtime`, `e2e`) instead of path-based for accelerator checks
- the codebase is ready for broader hosted-runner expansion in Phase 2 without another taxonomy refactor

## Phase 2: Real Hardware Expansion
Phase 2 is a separate implementation step after Phase 1 is merged and stable.

Phase 2 goal:
- move from narrow runtime validation to broader real hardware suites on GitHub-hosted platforms that exist
- pursue practical parity with the Linux CPU quality-gate suite, not blind parity at any cost

### Phase 2A: CUDA Suite
Target:
- one GPU-backed Linux job using a GitHub-hosted GPU runner, if available to the repository

Scope:
- install the CUDA-capable dependency profile
- run Ruff, Ruff format, Pyright, and a broader non-E2E pytest slice on the CUDA runner
- use the same composite setup action shape introduced in Phase 1

Phase 2A parity rule:
- aim for the same fast-check surface as `Quality Gate (Linux)` unless a test is CPU-only by design or unstable on CUDA
- any excluded test class must be documented explicitly in the workflow comments

### Phase 2B: Apple MPS Suite
Target:
- one Apple Silicon job that expands beyond `pytest -m runtime`

Scope:
- switch the Apple job from narrow runtime-validation only to a broader fast-check suite on real Apple Silicon
- keep SHAP E2E out unless explicitly promoted later
- use an explicit Apple Silicon runner label rather than relying on `macos-latest` alias behavior

Phase 2B parity rule:
- aim for the same fast-check surface as `Quality Gate (Linux)` unless a test is known to be platform-specific or unstable on MPS
- any excluded test class must be documented explicitly in the workflow comments

### Phase 2C: XPU Limitation
Do not add a real XPU suite in Phase 2.

Exact reason:
- no self-hosted infrastructure is available
- no GitHub-hosted Intel XPU runner is assumed available
- therefore real XPU execution cannot be made part of CI

What remains for XPU:
- mocked resolver-order tests
- backend hardware-label tests
- local manual validation on contributor hardware when available

### Phase 2 Preconditions
Do not start Phase 2 until all of these are true:
- Phase 1 is merged
- the marker split is stable
- the composite setup action is in use in both workflows
- the repository has access to the relevant GitHub-hosted runner classes for CUDA and Apple Silicon

## Files To Change
### Phase 1
- [`.github/workflows/code-quality.yaml`](.github/workflows/code-quality.yaml)
- [`.github/workflows/test-e2e.yml`](.github/workflows/test-e2e.yml)
- [`.github/actions/setup-raitap-env/action.yml`](.github/actions/setup-raitap-env/action.yml)
- [`pyproject.toml`](pyproject.toml)
- [`src/raitap/models/tests/test_models.py`](src/raitap/models/tests/test_models.py)

### Phase 2
- [`.github/workflows/code-quality.yaml`](.github/workflows/code-quality.yaml)
- possibly one new hardware-specific workflow file if the CUDA suite is kept separate from the main workflow
- possibly targeted test files or marker annotations if broader hardware exclusions need to be expressed precisely

## Non-Goals
### Phase 1
- no single OS x accelerator matrix
- no split Linux lint job and Linux test job
- no full macOS duplication of the Linux fast suite
- no new Windows runner job
- no real-device MPS/CoreML smoke test

### Phase 2
- no real XPU suite
- no Windows-only CUDA job unless the hosted GPU runner strategy explicitly requires Windows instead of Linux
