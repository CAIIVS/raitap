"""
Alibi Explain adapter.

Alibi Explain is licensed under Seldon's BSL 1.1 (not GPLv3). See installation and contributor docs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

from raitap.configs import resolve_run_dir

from ..contracts import ExplanationPayloadKind
from ..results import ConfiguredVisualiser, ExplanationResult
from .custom_explainer import CustomExplainer

_VISUALISATION_ONLY_KWARGS = frozenset({"sample_names", "show_sample_names"})


class AlibiExplainer(CustomExplainer):
    """
    Wraps selected Alibi Explain algorithms.

    ``KernelShap`` works with PyTorch ``nn.Module`` predictions (black-box). ``IntegratedGradients``
    follows Alibi's TensorFlow/Keras API: pass ``keras_model`` in the Hydra ``constructor`` block.
    """

    ALIBI_BSL_LICENSE_WARNING: ClassVar[bool] = True

    output_payload_kind: ClassVar[ExplanationPayloadKind] = ExplanationPayloadKind.ATTRIBUTIONS

    def __init__(self, algorithm: str = "KernelShap", **init_kwargs: Any) -> None:
        self.algorithm = algorithm
        self.init_kwargs = dict(init_kwargs)

    def explain(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        *,
        backend: object | None = None,
        run_dir: str | Path | None = None,
        output_root: str | Path = ".",
        experiment_name: str | None = None,
        explainer_target: str | None = None,
        explainer_name: str | None = None,
        visualisers: list[ConfiguredVisualiser] | None = None,
        **kwargs: Any,
    ) -> ExplanationResult:
        if importlib.util.find_spec("alibi") is None:
            raise ImportError(
                "Alibi explainer requires the `alibi` package. Install it with pip/uv into your "
                "Install it via `uv add raitap[alibi]` after adding the required dependency "
                "overrides to your pyproject.toml. "
                "See docs/using-raitap/installation.md for the exact steps."
            )

        del backend
        visualisers_list: list[ConfiguredVisualiser] = [] if visualisers is None else visualisers
        metadata_kwargs = dict(kwargs)
        call_kwargs = {
            key: value for key, value in kwargs.items() if key not in _VISUALISATION_ONLY_KWARGS
        }

        if self.algorithm == "KernelShap":
            attributions = self._kernel_shap_attributions(model, inputs, **call_kwargs)
        elif self.algorithm == "IntegratedGradients":
            attributions = self._integrated_gradients_attributions(inputs, **call_kwargs)
        else:
            raise ValueError(
                f"Unsupported Alibi algorithm {self.algorithm!r}. "
                "Supported: 'KernelShap', 'IntegratedGradients'."
            )

        explanation = ExplanationResult(
            attributions=attributions,
            inputs=inputs,
            run_dir=(
                Path(run_dir)
                if run_dir is not None
                else resolve_run_dir(output_root=output_root, subdir="transparency")
            ),
            experiment_name=experiment_name,
            explainer_target=(explainer_target or f"{type(self).__module__}.{type(self).__name__}"),
            algorithm=self.algorithm,
            explainer_name=explainer_name,
            kwargs=metadata_kwargs,
            visualisers=visualisers_list,
            payload_kind=self.output_payload_kind,
        )
        explanation.write_artifacts()
        return explanation

    def _kernel_shap_attributions(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        from alibi.explainers import KernelShap

        x = inputs.detach().cpu().numpy().astype(np.float32, copy=False)
        background = kwargs.get("background_data")
        if background is None:
            background_np = x
        elif isinstance(background, torch.Tensor):
            background_np = background.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            background_np = np.asarray(background, dtype=np.float32)

        device = next(model.parameters()).device

        def predict_fn(arr: np.ndarray) -> np.ndarray:
            tensor = torch.as_tensor(arr, dtype=torch.float32, device=device)
            model.eval()
            with torch.no_grad():
                out = model(tensor)
            return out.detach().cpu().numpy()

        task = str(kwargs.get("task", "classification"))
        ks = KernelShap(predict_fn, task=task)
        ks.fit(background_np)

        nsamples = kwargs.get("nsamples", 50)
        if not isinstance(nsamples, int):
            raise TypeError(f"nsamples must be an int, got {type(nsamples).__name__}.")
        explain_extra: dict[str, Any] = {}
        if "target" in kwargs:
            explain_extra["target"] = kwargs["target"]
        explanation = ks.explain(x, nsamples=nsamples, **explain_extra)
        raw = explanation.shap_values
        return torch.as_tensor(
            self._normalise_shap_array(raw, inputs.shape, kwargs.get("target")),
            dtype=inputs.dtype,
            device=inputs.device,
        )

    def _integrated_gradients_attributions(
        self,
        inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        from alibi.explainers import IntegratedGradients

        keras_model = self.init_kwargs.get("keras_model")
        if keras_model is None:
            raise ValueError(
                "Alibi IntegratedGradients requires `keras_model` in the explainer `constructor` "
                "(a tf.keras.Model). For PyTorch use Captum or set algorithm: KernelShap."
            )

        ig_kw = {key: value for key, value in self.init_kwargs.items() if key != "keras_model"}
        ig = IntegratedGradients(keras_model, **ig_kw)

        x = inputs.detach().cpu().numpy().astype(np.float32, copy=False)
        baselines = kwargs.get("baselines")
        baselines_np = None
        if baselines is not None:
            if isinstance(baselines, torch.Tensor):
                baselines_np = baselines.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                baselines_np = np.asarray(baselines, dtype=np.float32)

        explain_kw = {
            key: value
            for key, value in kwargs.items()
            if key not in {"baselines", "background_data", "task", "nsamples"}
        }
        explanation = ig.explain(x, baselines=baselines_np, **explain_kw)
        attr = np.asarray(explanation.attributions, dtype=np.float32)
        return torch.as_tensor(attr, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def _normalise_shap_array(
        raw: np.ndarray | list[np.ndarray],
        expected_shape: torch.Size,
        target: Any,
    ) -> np.ndarray:
        if isinstance(raw, list):
            if not raw:
                raise ValueError("Alibi KernelShap returned empty shap_values list.")
            if target is None:
                arr = np.stack(raw, axis=-1)
                arr = np.sum(arr, axis=-1)
            else:
                idx = int(target) if isinstance(target, (int, np.integer)) else int(target[0])
                arr = np.asarray(raw[idx], dtype=np.float32)
        else:
            arr = np.asarray(raw, dtype=np.float32)

        expected_tuple = tuple(int(dim) for dim in expected_shape)
        if tuple(arr.shape) != expected_tuple:
            expected_numel = int(np.prod(expected_tuple))
            if arr.size == expected_numel:
                arr = arr.reshape(expected_tuple)
            else:
                raise ValueError(
                    f"Alibi shap_values shape {arr.shape} cannot be aligned "
                    f"with inputs {expected_tuple}."
                )
        return arr.astype(np.float32, copy=False)
