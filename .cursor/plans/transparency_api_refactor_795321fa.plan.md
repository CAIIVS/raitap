---
name: transparency api refactor
overview: >
  Execute in order: (A) payload vocabulary + explainer/visualiser contracts + CustomExplainer hook;
  (B) Alibi pilot adapter + BSL licensing warnings in CLI/docs; (C) ONNX numpy groundwork;
  (D) remove torch from onnx extras only when preconditions are met. Maintainer owns all git commits.
todos:
  - id: phase-a-contracts
    content: "Steps 1–15: contracts.py, ExplanationResult, factory checks, CustomExplainer, tests, docs."
    status: pending
  - id: phase-a-commit
    content: "Step 16: maintainer review + commit Phase A only (implementer does not commit)."
    status: pending
  - id: phase-b-alibi
    content: "Steps 17–23: alibi extra, CLI/docs BSL warnings, AlibiExplainer, YAML, exports, tests."
    status: pending
  - id: phase-c-onnx-numpy
    content: "Steps 24–27: OnnxBackend.forward_numpy, load_numpy_from_source, tests."
    status: pending
  - id: phase-d-onnx-extra
    content: "Steps 28–30: drop torch from onnx-* extras + docs (only after preconditions)."
    status: pending
isProject: false
---

# Transparency API Refactor

## Locked rules (no branching during implementation)

- **Additive API:** extend types and behaviour; do not rename or remove public fields on `ExplanationResult`, `Explanation`, or `BaseExplainer.explain`’s parameters in this epic.
- **Default payload:** every existing explainer and visualiser remains **attribution / tensor heat-map** compatible (`ATTRIBUTIONS`). The enum also defines `STRUCTURED` for tests and future non–heat-map payloads; production paths do not use it until implemented.
- **Wildcard visualisers:** `supported_payload_kinds == frozenset()` means the visualiser accepts **all** payload kinds. A non-empty frozenset means the explainer’s `output_payload_kind` must be an element of that set.
- **Explainer registration:** Hydra `_target_` may instantiate either `BaseExplainer` subclasses **or** `CustomExplainer` subclasses; both must expose instance method `explain` with the **same parameter list and semantics** as `BaseExplainer.explain` today (see `src/raitap/transparency/explainers/base_explainer.py`).
- **Phase B pilot library:** **Alibi** only (Seldon Alibi Explain). Do not add an OmniXAI adapter in this epic.
- **Git and commit history:** The implementer **must not** run `git commit`, `git merge`, rebases, or any other command that creates or rewrites commits. Produce file changes only. **Phase B (Alibi) must not start until** the maintainer has **reviewed and committed** Phase A (steps 1–15) so the history shows a **clean separation** between the core contract refactor and the Alibi integration.

---

## Phase A — Contracts, results, factory, custom explainer hook

1. **Add** `src/raitap/transparency/contracts.py`. Define `class ExplanationPayloadKind(StrEnum)` with members: `ATTRIBUTIONS = "attributions"` and `STRUCTURED = "structured"`. **Production** explainers and visualisers use only `ATTRIBUTIONS` until a feature implements `STRUCTURED` end-to-end. `STRUCTURED` exists so tests (and future adapters) can assert payload mismatch without hacks. Add a module docstring stating that further kinds are **only** added here when first implemented end-to-end.

2. **Add** `def explainer_output_kind(explainer: object) -> ExplanationPayloadKind` in `contracts.py`: return `type(explainer).output_payload_kind` if that attribute exists and is an `ExplanationPayloadKind`; otherwise return `ExplanationPayloadKind.ATTRIBUTIONS`.

3. **Add** `@typing.runtime_checkable` `class ExplainerAdapter(typing.Protocol)` in `contracts.py` requiring: instance method `explain` with the same signature as `BaseExplainer.explain` (mirror parameters from `src/raitap/transparency/explainers/base_explainer.py` `def explain` through `**kwargs`); class attribute `output_payload_kind: ClassVar[ExplanationPayloadKind]`. (Protocols cannot enforce `ClassVar`; `explainer_output_kind()` is the runtime source of truth.)

4. **Edit** `src/raitap/transparency/results.py`. Add field `payload_kind: ExplanationPayloadKind` to `ExplanationResult`, default `ExplanationPayloadKind.ATTRIBUTIONS`. Extend `_metadata()` to include `"payload_kind": self.payload_kind.value`.

5. **Edit** `ExplanationResult.write_artifacts`: if `payload_kind == ExplanationPayloadKind.ATTRIBUTIONS`, keep current behaviour (`torch.save` attributions). If `payload_kind == ExplanationPayloadKind.STRUCTURED`, **raise** `NotImplementedError` with a clear message until persistence for structured payloads is implemented. Any future enum member defaults to the same guard until implemented.

6. **Edit** `src/raitap/transparency/explainers/base_explainer.py`. Add `output_payload_kind: ClassVar[ExplanationPayloadKind] = ExplanationPayloadKind.ATTRIBUTIONS`. Pass `payload_kind=self.output_payload_kind` into `ExplanationResult(...)` inside `explain`.

7. **Edit** `src/raitap/transparency/explainers/captum_explainer.py` and `src/raitap/transparency/explainers/shap_explainer.py`: add explicit `output_payload_kind: ClassVar[ExplanationPayloadKind] = ExplanationPayloadKind.ATTRIBUTIONS` on each class (documents the pattern for future adapters).

8. **Edit** `src/raitap/transparency/exceptions.py`. Add `PayloadVisualiserIncompatibilityError` with constructor args: `explainer_target: str`, `visualiser: str`, `output_payload_kind: str`, `supported_payload_kinds: list[str]`. Message must state that the visualiser’s `supported_payload_kinds` does not include the explainer’s output kind.

9. **Edit** `src/raitap/transparency/visualisers/base_visualiser.py`. Add `supported_payload_kinds: ClassVar[frozenset[ExplanationPayloadKind]] = frozenset({ExplanationPayloadKind.ATTRIBUTIONS})`. Document the empty-frozenset wildcard rule in the class docstring. Do **not** edit concrete visualiser subclasses unless one needs the wildcard later.

10. **Edit** `src/raitap/transparency/factory.py`. Add `check_explainer_visualiser_payload_compat(explainer: object, visualisers: list[ConfiguredVisualiser]) -> None` after `check_explainer_visualiser_compat` succeeds: for each visualiser, read `supported_payload_kinds`; if non-empty and `explainer_output_kind(explainer) not in supported_payload_kinds`, raise `PayloadVisualiserIncompatibilityError`. Call this new function from `Explanation.__new__` immediately after `check_explainer_visualiser_compat`.

11. **Edit** `src/raitap/transparency/factory.py`: change `create_explainer` return annotation to `tuple[ExplainerAdapter, str]` (import `ExplainerAdapter` from `contracts`). After `instantiate`, assert `callable(getattr(explainer, "explain", None))` or raise `ValueError` with the same tone as the existing instantiation error.

12. **Add** `src/raitap/transparency/explainers/custom_explainer.py`. Define `class CustomExplainer(ABC)` with: `check_backend_compat(self, backend: object) -> None` default `return None`; **abstract** `def explain(self, model: torch.nn.Module, inputs: torch.Tensor, *, backend: object | None = None, run_dir: str | Path | None = None, output_root: str | Path = ".", experiment_name: str | None = None, explainer_target: str | None = None, explainer_name: str | None = None, visualisers: list[ConfiguredVisualiser] | None = None, **kwargs: Any) -> ExplanationResult` (copy parameter list from `BaseExplainer.explain` exactly). Subclasses set `output_payload_kind` when not `ATTRIBUTIONS`. **Do not** inherit `BaseExplainer`; this avoids `compute_attributions` for third-party adapters such as Alibi.

13. **Edit** `src/raitap/transparency/explainers/__init__.py`: export `CustomExplainer`.

14. **Edit** `src/raitap/transparency/__init__.py`: export `ExplanationPayloadKind`, `ExplainerAdapter`, and `CustomExplainer`.

15. **Tests and docs (mandatory end of Phase A):**
    - Add or extend tests in `src/raitap/transparency/tests/test_factory.py` for payload compatibility: (a) explainer `ATTRIBUTIONS` + visualiser with `supported_payload_kinds=frozenset()` passes (wildcard); (b) explainer `ATTRIBUTIONS` + visualiser whose `supported_payload_kinds` is `frozenset({ExplanationPayloadKind.STRUCTURED})` raises `PayloadVisualiserIncompatibilityError`.
    - Run `uv run pytest src/raitap/transparency -q` and fix all failures.
    - Update `docs/contributor/transparency.md`: document `ExplanationPayloadKind`, `supported_payload_kinds` wildcard, `CustomExplainer` vs `BaseExplainer`, and `ExplainerAdapter`.
    - Update `src/raitap/configs/schema.py` comment on `_target_` to say it targets an `ExplainerAdapter` implementation (not only `BaseExplainer`).

16. **Phase A complete — stop for maintainer (mandatory gate before any Alibi work):**
    - The implementer **stops** after step 15. **Do not** implement steps 17–23 until the maintainer says to continue.
    - The implementer **must not** run `git commit` (or any git history mutation). The maintainer **reviews** Phase A changes and **creates the commit(s)** for Phase A so the history separates “contract refactor” from “Alibi pilot.”
    - Only after that commit exists on the branch the implementer uses may work begin on step 17.

---

## Phase B — Alibi pilot adapter (after Phase A is committed by maintainer)

17. **Add** `[project.optional-dependencies]` extra **`alibi`** in `pyproject.toml` with a pinned lower-bound appropriate for Python 3.13. **Do not** add Alibi to the composite **`transparency`** extra until the adapter is stable (same rule as before: `transparency` stays `shap` + `captum` only until explicitly changed).

18. **Add Alibi licensing warnings (BSL 1.1):**
    - **Docs:** In `docs/using-raitap/installation.md` (and `docs/contributor/setup.md` if Alibi install is mentioned), add a short **callout** that **Alibi Explain** is licensed under **Seldon’s Business Source License 1.1**, not GPLv3; **non-production** use is permitted on Seldon’s terms; **production** (and many commercial uses) may require a **commercial license** from Seldon — users must read Seldon’s license and FAQ. Link to `https://github.com/SeldonIO/alibi/blob/master/LICENSE` and Seldon’s licensing FAQ or pricing page. State that **raitap** (GPLv3) does not relicense Alibi.
    - **CLI:** Define `ALIBI_BSL_LICENSE_WARNING: ClassVar[bool] = True` on `AlibiExplainer` only. In `Explanation.__new__` in `factory.py`, immediately after `create_explainer` returns, if `getattr(type(explainer), "ALIBI_BSL_LICENSE_WARNING", False)` is true, emit a **one-time per process** `logging.warning` with the same substance as the docs callout (BSL, verify Seldon terms for production/commercial use). Use a **module-level** flag in `factory.py` (e.g. `_ALIBI_BSL_WARNING_EMITTED`) so the warning prints at most once even if multiple explainers run. Document this mechanism in `docs/contributor/transparency.md`.

19. **Add** `src/raitap/transparency/explainers/alibi_explainer.py`: subclass `CustomExplainer`, integrate **Alibi `IntegratedGradients` first** (numpy-friendly inputs, `.attributions` → map to `torch.Tensor` for `ExplanationResult.attributions`). Set `output_payload_kind` to `ExplanationPayloadKind.ATTRIBUTIONS`. If a later change adds a non–heat-map Alibi method, extend `contracts.py`, `write_artifacts`, and visualisers per the existing plan rules.

20. **Add** Hydra YAML under `src/raitap/configs/transparency/` pointing `_target_` at `AlibiExplainer`, following existing `captum` / `shap` config style.

21. **Export** `AlibiExplainer` from `src/raitap/transparency/explainers/__init__.py` and `src/raitap/transparency/__init__.py` `__all__`.

22. **Add** focused tests under `src/raitap/transparency/explainers/tests/` with optional-dependency markers (e.g. `needs_alibi`) mirroring Captum/SHAP. **Done when:** `uv run pytest …` passes for the new tests and the adapter runs the chosen algorithm on a minimal fixture.

23. **Extend** `docs/contributor/transparency.md` with an **Alibi** subsection: dependency extra `alibi`, BSL reminder, pointer to installation doc, and how CLI warning is triggered.

**Phase B git rule:** The implementer **must not** commit Phase B. The maintainer **reviews and commits** Phase B separately after step 23 so history stays split from Phase A.

---

## Phase C — ONNX numpy groundwork (does not remove torch yet)

24. **Edit** `src/raitap/models/backend.py`. Add `def forward_numpy(self, batch: np.ndarray) -> np.ndarray` on `OnnxBackend` that runs `session.run` and returns the same primary output selection logic as today’s multi-output handling, **without** requiring `torch.Tensor` for the forward path. Refactor `OnnxBackend.__call__(self, inputs: torch.Tensor)` to: convert tensor → numpy → `forward_numpy` → `torch.from_numpy` on the result (preserve existing return type for callers).

25. **Add** tests in `src/raitap/models/tests/` covering `forward_numpy` vs `__call__` numerical agreement on a tiny ONNX model fixture (reuse existing ONNX fixtures if present).

26. **Edit** `src/raitap/data/data.py`. Add `def load_numpy_from_source(source: str, n_samples: int | None = None) -> np.ndarray` by extracting shared loading logic from `load_tensor_from_source` into private helpers so CSV/image/tabular paths match, returning numpy arrays with the same shapes/dtypes as `.numpy()` on the current tensors.

27. **Add** tests in `src/raitap/data/tests/test_data.py` for `load_numpy_from_source` parity with `load_tensor_from_source` on at least one tabular and one image sample.

---

## Phase D — Remove torch from `onnx-*` extras (blocked until preconditions)

**Do not start Phase D until all of the following are true:**

- At least one `CustomExplainer` path can run with **numpy** inputs through `Explanation` **or** the pipeline documents that ONNX black-box explainers use `forward_numpy` only inside adapters without importing `torch` in the explainer module.
- `Data` / pipeline no longer **require** `torch.Tensor` for the assessment entrypoint used in that configuration **or** torch is an optional import confined to torch backends only.

28. **Edit** `pyproject.toml`: remove `torch` and `torchvision` from `onnx-cpu`, `onnx-cuda`, and `onnx-intel` extras; remove or rewrite the TODO comment accordingly.

29. **Regenerate** lockfile with `uv lock` (or project-standard command).

30. **Edit** `docs/using-raitap/installation.md` and `docs/contributor/setup.md` so `onnx-*` examples no longer imply torch is pulled in; add explicit note that torch-backed explainers still need `torch-cpu` / `torch-cuda` / `torch-intel` or `transparency` as today.

---

## Pilot choice (locked)

**Alibi** is the Phase B pilot. OmniXAI is out of scope for this epic unless the maintainer opens a follow-up plan.
