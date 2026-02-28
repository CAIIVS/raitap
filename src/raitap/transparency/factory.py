"""
Factory functions for the RAITAP transparency module.

Public surface
--------------
explain(config, model, inputs, **kwargs)
    One-call entry point (new simplified API).

create_explainer(method, **init_kwargs)
    Lower-level factory: builds an explainer from a registry method.

method_from_config(config)
    Translates a config object to an ExplainerMethod from the registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from hydra.core.hydra_config import HydraConfig

from .explainers import BaseExplainer, CaptumExplainer, ShapExplainer
from .methods_registry import (
    SHAP,
    Captum,
    ExplainerMethod,
    validate_visualiser_compatibility,
)
from .visualisers import (
    CaptumImageVisualiser,
    CaptumTextVisualiser,
    CaptumTimeSeriesVisualiser,
    ShapBarVisualiser,
    ShapBeeswarmVisualiser,
    ShapForceVisualiser,
    ShapImageVisualiser,
    ShapWaterfallVisualiser,
)
from .visualisers.base import BaseVisualiser

if TYPE_CHECKING:
    import torch.nn as nn
    from matplotlib.figure import Figure

    from raitap.configs.schema import AppConfig


# ---------------------------------------------------------------------------
# Visualiser lookup table
# ---------------------------------------------------------------------------

_VISUALISER_MAP: dict[str, dict[str, type[BaseVisualiser]]] = {
    "captum": {
        "image": CaptumImageVisualiser,
        "time_series": CaptumTimeSeriesVisualiser,
        "text": CaptumTextVisualiser,
    },
    "shap": {
        "bar": ShapBarVisualiser,
        "beeswarm": ShapBeeswarmVisualiser,
        "waterfall": ShapWaterfallVisualiser,
        "force": ShapForceVisualiser,
        "image": ShapImageVisualiser,
    },
}


def _create_visualiser(framework: str, name: str) -> BaseVisualiser:
    """Instantiate a visualiser from the lookup table."""
    fw = _VISUALISER_MAP.get(framework)
    if fw is None:
        raise ValueError(f"Unknown framework {framework!r}.")
    cls = fw.get(name)
    if cls is None:
        available = list(fw)
        raise ValueError(
            f"Unknown visualiser {name!r} for framework {framework!r}.  Available: {available}"
        )
    return cls()


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

        - ``framework``   - ``"captum"`` or ``"shap"``
        - ``algorithm``   - algorithm name from the registry
        - ``visualisers`` - list of visualiser names for the chosen framework

    The output directory is read from ``config.output_dir``.

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
        Attribution tensor saved to Hydra's run directory.

    ``"visualisations"``
        Dict mapping each visualiser name to its Matplotlib Figure.
        Figures are also saved to Hydra's run directory.

    ``"run_dir"``
        :class:`~pathlib.Path` of the Hydra run directory (``Path.cwd()``).

    Raises
    ------
    VisualiserIncompatibilityError
        If any requested visualiser is not compatible with the chosen algorithm.
    ValueError
        If an unknown framework, algorithm, or visualiser name is requested.

    Examples
    --------
    >>> from raitap.transparency import explain
    >>> result = explain(config, model, images, target=0)
    >>> result["attributions"].shape
    torch.Size([4, 3, 224, 224])
    >>> list(result["visualisations"])
    ['image']
    """
    tc = config.transparency
    framework: str = tc.framework
    algorithm: str = tc.algorithm
    visualiser_names: list[str] = tc.visualisers

    # 1. Validate ALL requested visualisers before doing any computation.
    for vis_name in visualiser_names:
        validate_visualiser_compatibility(framework, vis_name, algorithm)

    # 2. Build explainer and compute attributions.
    method = method_from_config(tc)
    explainer = create_explainer(method)
    attributions = explainer.compute_attributions(model, inputs, **kwargs)

    # 3. Resolve run directory (Hydra when available, else config.output_dir).
    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        run_dir = Path(config.output_dir)

    # 4. Persist attributions.
    torch.save(attributions, run_dir / "attributions.pt")

    # 5. Generate and persist each requested visualisation.
    visualisations: dict[str, Figure] = {}
    for vis_name in visualiser_names:
        visualiser = _create_visualiser(framework, vis_name)
        fig = visualiser.visualise(attributions, inputs=inputs)
        visualiser.save(attributions, run_dir / f"{vis_name}.png", inputs=inputs)
        visualisations[vis_name] = fig

    # 6. Save metadata snapshot.
    metadata = {
        "experiment_name": config.experiment_name,
        "framework": framework,
        "algorithm": algorithm,
        "visualisers": list(visualiser_names),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return {
        "attributions": attributions,
        "visualisations": visualisations,
        "run_dir": run_dir,
    }


# ---------------------------------------------------------------------------
# Lower-level helpers (kept for power users)
# ---------------------------------------------------------------------------


def create_explainer(method: ExplainerMethod, **init_kwargs) -> BaseExplainer:
    """
    Instantiate an explainer adapter from a registry method.

    Parameters
    ----------
    method:
        ``ExplainerMethod`` from the registry
        (e.g. ``Captum.IntegratedGradients``).
    **init_kwargs:
        Constructor arguments forwarded to the explainer
        (e.g. ``layer=model.layer4`` for GradCAM).

    Returns
    -------
    A :class:`BaseExplainer` subclass ready for ``compute_attributions``.

    Examples
    --------
    >>> from raitap.transparency import create_explainer, Captum, SHAP
    >>> explainer = create_explainer(Captum.IntegratedGradients)
    >>> explainer = create_explainer(Captum.LayerGradCam, layer=model.layer4)
    >>> explainer = create_explainer(SHAP.GradientExplainer)
    """
    assert method.algorithm is not None, "Algorithm must be set on the ExplainerMethod."

    if method.framework == "captum":
        return CaptumExplainer(method.algorithm, **init_kwargs)
    elif method.framework == "shap":
        return ShapExplainer(method.algorithm, **init_kwargs)
    else:
        raise ValueError(f"Unknown framework: {method.framework!r}")


def method_from_config(config) -> ExplainerMethod:
    """
    Translate a config section to an :class:`ExplainerMethod` from the registry.

    Parameters
    ----------
    config:
        Config object with ``.framework`` and ``.algorithm`` attributes
        (typically ``cfg.transparency``).

    Returns
    -------
    The matching :class:`ExplainerMethod`.

    Raises
    ------
    ValueError
        If the algorithm is not found in the framework registry.
    AttributeError
        If the config is missing ``.framework`` or ``.algorithm``.

    Examples
    --------
    >>> method = method_from_config(cfg.transparency)
    >>> explainer = create_explainer(method)
    """
    framework_name = config.framework
    algorithm_name = config.algorithm

    if framework_name == "captum":
        try:
            return getattr(Captum, algorithm_name)
        except AttributeError:
            available = [n for n in dir(Captum) if not n.startswith("_")]
            raise ValueError(
                f"Captum has no method {algorithm_name!r} in RAITAP registry.\n"
                f"Available: {', '.join(available)}\n"
                f"Add new methods to transparency/methods_registry.py after testing."
            ) from None
    elif framework_name == "shap":
        try:
            return getattr(SHAP, algorithm_name)
        except AttributeError:
            available = [n for n in dir(SHAP) if not n.startswith("_")]
            raise ValueError(
                f"SHAP has no explainer {algorithm_name!r} in RAITAP registry.\n"
                f"Available: {', '.join(available)}\n"
                f"Add new explainers to transparency/methods_registry.py after testing."
            ) from None
    else:
        raise ValueError(f"Unknown framework: {framework_name!r}.  Supported: 'captum', 'shap'")
