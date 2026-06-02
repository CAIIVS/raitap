"""Pipeline phase infrastructure — cross-cutting, task-agnostic pieces shared by
every run: the ``AssessmentPhase`` base + ``PhaseContext`` (``base.py``), the
``run_adapters`` loop helper, the phase ``registry``, and the forward pass /
prediction summaries / input-metadata steps. Module-specific phase work
(``assess_transparency``, ``assess_robustness``, ``evaluate_metrics``,
``explain_detection``) lives in each owning module's ``phase.py``."""
