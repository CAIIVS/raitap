"""
Factory / orchestration layer for the RAITAP transparency module.

Public surface
--------------
explain(config, model, inputs, **kwargs)
    One-call entry point.  Uses Hydra ``instantiate()`` to build the
    explainer and visualisers from ``_target_`` keys in the config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from ..configs.factory_utils import cfg_to_dict, resolve_target
from .methods_registry import VisualiserIncompatibilityError

_TRANSPARENCY_PREFIX = "raitap.transparency."

if TYPE_CHECKING:
    import torch.nn as nn
    from matplotlib.figure import Figure

    from ..configs.schema import AppConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _serialisable(v) -> object:
    """Best-effort conversion to a JSON-serialisable scalar."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    return repr(v)


# ---------------------------------------------------------------------------
# Primary API
# ---------------------------------------------------------------------------


def explain(
    config: AppConfig,
    model: nn.Module,
    inputs: torch.Tensor,
    **kwargs,
) -> dict:
    """
    Compute attributions and produce visualisations in one call.

    Parameters
    ----------
    config:
        ``AppConfig`` whose ``transparency`` section drives the run:

        - ``_target_``   - fully-qualified ``BaseExplainer`` subclass
        - ``algorithm``  - algorithm name forwarded to the explainer
        - ``visualisers``- list of dicts each carrying a ``_target_`` key
          pointing to a ``BaseVisualiser`` subclass

    model:
        PyTorch model to explain.
    inputs:
        Input tensor passed to both the explainer and visualisers.
    **kwargs:
        Framework-specific keyword arguments forwarded to
        ``compute_attributions`` (e.g. ``target``, ``baselines``,
        ``background_data``).

    Returns
    -------
    dict with keys:

    ``"attributions"``
        Attribution tensor saved to the run directory.

    ``"visualisations"``
        Dict mapping each visualiser class name to its Matplotlib Figure.
        Figures are also saved to the run directory.

    ``"run_dir"``
        :class:`~pathlib.Path` of the run directory.

    Raises
    ------
    VisualiserIncompatibilityError
        If any requested visualiser is not compatible with the chosen algorithm.
    ValueError
        If a ``_target_`` cannot be resolved or instantiated.
    """
    tc = config.transparency
    raw_tc = cfg_to_dict(tc)

    target_path: str = raw_tc.get("_target_", "")
    algorithm: str = raw_tc.get("algorithm", "")
    vis_cfgs: list[dict] = raw_tc.get("visualisers", [])

    # ------------------------------------------------------------------
    # 1. Instantiate explainer.
    #    Pass everything EXCEPT ``visualisers`` (which is used only here
    #    in the orchestrator, not by the explainer constructor).
    # ------------------------------------------------------------------
    explainer_cfg = {k: v for k, v in raw_tc.items() if k != "visualisers"}
    target = explainer_cfg.get("_target_", "")
    explainer_cfg["_target_"] = resolve_target(target, _TRANSPARENCY_PREFIX)
    try:
        explainer = instantiate(explainer_cfg)
    except Exception as e:
        raise ValueError(
            f"Could not instantiate explainer {target_path!r}.\n"
            f"Check that _target_ points to a valid BaseExplainer subclass."
        ) from e

    # ------------------------------------------------------------------
    # 2. Validate visualiser compatibility BEFORE any computation.
    # ------------------------------------------------------------------
    for vis_cfg in vis_cfgs:
        vis_target = vis_cfg.get("_target_", "") if isinstance(vis_cfg, dict) else ""
        if isinstance(vis_cfg, dict) and "_target_" in vis_cfg:
            vis_cfg = {
                **vis_cfg,
                "_target_": resolve_target(vis_cfg["_target_"], _TRANSPARENCY_PREFIX),
            }
        try:
            visualiser = instantiate(vis_cfg)
        except Exception as e:
            raise ValueError(f"Could not instantiate visualiser {vis_target!r}.") from e
        if visualiser.compatible_algorithms and algorithm not in visualiser.compatible_algorithms:
            raise VisualiserIncompatibilityError(
                framework=target_path,
                visualiser=type(visualiser).__name__,
                algorithm=algorithm,
                compatible_algorithms=sorted(visualiser.compatible_algorithms),
            )

    # ------------------------------------------------------------------
    # 3. Compute attributions.
    # ------------------------------------------------------------------
    attributions = explainer.compute_attributions(model, inputs, **kwargs)

    # ------------------------------------------------------------------
    # 4. Resolve run directory (Hydra first, fallback to config.fallback_output_dir).
    # ------------------------------------------------------------------
    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        run_dir = Path(config.fallback_output_dir)

    # ------------------------------------------------------------------
    # 5. Persist attributions.
    # ------------------------------------------------------------------
    torch.save(attributions, run_dir / "attributions.pt")

    # ------------------------------------------------------------------
    # 6. Generate and persist each visualisation.
    # ------------------------------------------------------------------
    visualisations: dict[str, Figure] = {}
    for vis_cfg in vis_cfgs:
        if isinstance(vis_cfg, dict) and "_target_" in vis_cfg:
            vis_cfg = {
                **vis_cfg,
                "_target_": resolve_target(vis_cfg["_target_"], _TRANSPARENCY_PREFIX),
            }
        visualiser = instantiate(vis_cfg)
        name = type(visualiser).__name__
        fig = visualiser.visualise(attributions, inputs=inputs)
        visualiser.save(attributions, run_dir / f"{name}.png", inputs=inputs)
        visualisations[name] = fig

    # ------------------------------------------------------------------
    # 7. Save metadata snapshot.
    # ------------------------------------------------------------------
    metadata = {
        "experiment_name": config.experiment_name,
        "target": resolve_target(target_path, _TRANSPARENCY_PREFIX),
        "algorithm": algorithm,
        "visualisers": [resolve_target(v["_target_"], _TRANSPARENCY_PREFIX) for v in vis_cfgs],
        "kwargs": {k: _serialisable(v) for k, v in kwargs.items()},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"✓ Attributions shape: {attributions.shape}")
    for name in visualisations:
        print(f"✓ Visualisation saved: {run_dir}/{name}.png")
    print(f"✓ Metadata saved:      {run_dir}/metadata.json")

    return {
        "attributions": attributions,
        "visualisations": visualisations,
        "run_dir": run_dir,
    }
