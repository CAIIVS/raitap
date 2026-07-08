"""Grade explanations as a post-step of the transparency phase (#341).

``grade_explanations`` is the seam between the transparency phase loop and the
Quantus-backed evaluators. It instantiates the configured evaluator from the
per-adapter ``EvaluationConfig`` (via ``hydra.utils.instantiate``), pulls the raw
model + device off the ``PreparedExplainer``'s backend, puts the model in eval
mode (Quantus's MODEL-requiring metrics raise ``AttributeError`` otherwise), and
grades each produced explanation. Returns ``[]`` when no evaluation is configured
or nothing was explained, so the phase stays a no-op unless a ``evaluation``
block is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra.utils import instantiate

from raitap.configs.registry_resolve import stamp_target_from_use
from raitap.configs.utils import cfg_to_dict
from raitap.transparency.evaluation.semantics import EvaluationContext

if TYPE_CHECKING:
    from raitap.configs.schema import EvaluationConfig
    from raitap.transparency.evaluation.contracts import EvaluationResult
    from raitap.transparency.results import ExplanationResult


def grade_explanations(
    evaluation_cfg: EvaluationConfig | None,
    explanations: list[ExplanationResult],
    prepared: object,
    *,
    softmax_default: bool = False,
) -> list[EvaluationResult]:
    """Grade each explanation with the configured evaluator; ``[]`` when disabled.

    ``prepared`` is a :class:`~raitap.transparency.phase.PreparedExplainer` (typed
    ``object`` to avoid a phase<->evaluation import cycle). ``instantiate`` is
    imported at module level so tests can monkeypatch ``step.instantiate``.
    """
    if evaluation_cfg is None or not explanations:
        return []
    cfg = cfg_to_dict(evaluation_cfg)
    stamp_target_from_use(cfg, group="_unscoped")
    evaluator = instantiate(cfg)
    backend = prepared.backend  # type: ignore[attr-defined]
    model = backend.autograd_module()
    if hasattr(model, "eval"):
        model.eval()
    device = backend.device
    explainer = getattr(prepared, "explainer", None)
    softmax = bool(getattr(evaluator, "softmax", softmax_default))

    out: list[EvaluationResult] = []
    for result in explanations:
        # ``result.baseline`` is a ``BaselineRecord`` (metadata) or ``None``; it is
        # used only for gating in ``EvaluationContext.available_requirements`` (the
        # record's presence flips ``EvalRequirement.BASELINE``), never forwarded to
        # Quantus by ``gather``.
        ctx = EvaluationContext(
            result=result,
            model=model,
            device=device,
            explainer=explainer,
            masks=None,
            baseline=getattr(result, "baseline", None),
            softmax=softmax,
        )
        evaluation = evaluator.evaluate(ctx, run_dir=result.run_dir)
        evaluation.write_artifacts()
        out.append(evaluation)
    return out
