"""Base class for attribution computation (framework-agnostic interface)"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from raitap.configs import resolve_run_dir

from ..results import ConfiguredVisualiser, ExplanationResult

_VISUALISATION_ONLY_KWARGS = frozenset({"sample_names", "show_sample_names"})
_BATCH_SIZE_KWARGS = frozenset({"batch_size", "max_batch_size"})
_PROGRESS_TOGGLE_KWARG = "show_progress"
_PROGRESS_DESC_KWARG = "progress_desc"


class BaseExplainer(ABC):
    """
    Abstract base class for all explainer adapters.
    """

    def __init__(self):
        self.attributions: torch.Tensor | None = None

    def check_backend_compat(self, backend: object) -> None:
        del backend
        return None

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
        visualisers_list: list[ConfiguredVisualiser] = [] if visualisers is None else visualisers
        metadata_kwargs = dict(kwargs)
        attribution_kwargs = {
            key: value for key, value in kwargs.items() if key not in _VISUALISATION_ONLY_KWARGS
        }
        attributions = self._compute_with_optional_batches(
            model,
            inputs,
            attribution_kwargs,
            backend,
        )
        self.attributions = attributions

        explanation = ExplanationResult(
            attributions=attributions,
            inputs=inputs,
            run_dir=(
                Path(run_dir)
                if run_dir is not None
                else resolve_run_dir(
                    output_root=output_root,
                    subdir="transparency",
                )
            ),
            experiment_name=experiment_name,
            explainer_target=(explainer_target or f"{type(self).__module__}.{type(self).__name__}"),
            algorithm=getattr(self, "algorithm", ""),
            explainer_name=explainer_name,
            kwargs=metadata_kwargs,
            visualisers=visualisers_list,
        )
        explanation.write_artifacts()
        return explanation

    def _compute_with_optional_batches(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        attribution_kwargs: dict[str, Any],
        backend: object | None,
    ) -> torch.Tensor:
        batch_size = self._pop_batch_size(attribution_kwargs)
        show_progress, progress_desc = self._pop_progress_settings(attribution_kwargs)
        if batch_size is None or inputs.shape[0] <= batch_size:
            return self.compute_attributions(model, inputs, backend=backend, **attribution_kwargs)

        chunks: list[torch.Tensor] = []
        total_batch = int(inputs.shape[0])
        starts = range(0, total_batch, batch_size)
        if show_progress:
            starts = self._wrap_with_progress(
                starts,
                total_batches=len(starts),
                progress_desc=progress_desc,
            )

        for start in starts:
            end = min(start + batch_size, total_batch)
            batch_inputs = inputs[start:end]
            batch_kwargs = self._slice_kwargs_for_batch(attribution_kwargs, start, end, total_batch)
            chunk = self.compute_attributions(
                model,
                batch_inputs,
                backend=backend,
                **batch_kwargs,
            )
            chunks.append(chunk.detach().cpu())
            del chunk, batch_inputs, batch_kwargs
            # Clean up memory after each batch to avoid OOM errors in long runs
            # (e.g. with SHAP partition explainer).
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(chunks, dim=0)

    def _pop_batch_size(self, attribution_kwargs: dict[str, Any]) -> int | None:
        batch_size: int | None = None
        for key in _BATCH_SIZE_KWARGS:
            if key in attribution_kwargs and attribution_kwargs[key] is not None:
                candidate = attribution_kwargs.pop(key)
                if not isinstance(candidate, int):
                    raise TypeError(f"{key} must be an int, got {type(candidate).__name__}.")
                if candidate <= 0:
                    raise ValueError(f"{key} must be > 0, got {candidate}.")
                batch_size = candidate
        return batch_size

    def _pop_progress_settings(self, attribution_kwargs: dict[str, Any]) -> tuple[bool, str | None]:
        show_progress_raw = attribution_kwargs.pop(_PROGRESS_TOGGLE_KWARG, True)
        if not isinstance(show_progress_raw, bool):
            raise TypeError(
                f"{_PROGRESS_TOGGLE_KWARG} must be a bool, got {type(show_progress_raw).__name__}."
            )

        progress_desc = attribution_kwargs.pop(_PROGRESS_DESC_KWARG, None)
        if progress_desc is not None and not isinstance(progress_desc, str):
            raise TypeError(
                f"{_PROGRESS_DESC_KWARG} must be a str, got {type(progress_desc).__name__}."
            )
        return show_progress_raw, progress_desc

    def _wrap_with_progress(
        self,
        starts: range,
        *,
        total_batches: int,
        progress_desc: str | None,
    ) -> Any:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return starts

        desc = progress_desc or f"{getattr(self, 'algorithm', type(self).__name__)} batches"
        return tqdm(starts, total=total_batches, desc=desc)

    def _slice_kwargs_for_batch(
        self,
        attribution_kwargs: dict[str, Any],
        start: int,
        end: int,
        total_batch: int,
    ) -> dict[str, Any]:
        return {
            key: self._slice_batch_value(key, value, start, end, total_batch)
            for key, value in attribution_kwargs.items()
        }

    @staticmethod
    def _slice_batch_value(
        key: str,
        value: Any,
        start: int,
        end: int,
        total_batch: int,
    ) -> Any:
        if key == "background_data":
            return value

        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == total_batch:
            return value[start:end]

        if isinstance(value, list) and len(value) == total_batch:
            return value[start:end]

        if isinstance(value, tuple) and len(value) == total_batch:
            return value[start:end]

        shape = getattr(value, "shape", None)
        if (
            shape is not None
            and len(shape) > 0
            and shape[0] == total_batch
            and hasattr(value, "__getitem__")
        ):
            return value[start:end]

        return value

    @abstractmethod
    def compute_attributions(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute attributions for the given inputs.

        Args:
            model:   PyTorch model to explain.
            inputs:  Input tensor (shape depends on modality).
            **kwargs: Framework-specific keyword arguments
                      (e.g. ``target``, ``baselines``, ``background_data``).

        Returns:
            Attribution tensor matching the input shape.
        """
