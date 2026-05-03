"""Explainer base classes for the RAITAP transparency module."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, cast

import torch

from raitap.configs import resolve_run_dir

from ..contracts import (
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    ExplanationSemantics,
    ExplanationTarget,
    InputSpec,
    SampleSelection,
    ScopeDefinitionStep,
)
from ..results import ConfiguredVisualiser, ExplanationResult
from ..semantics import infer_input_spec, infer_output_space, method_families_for_explainer

_NON_BATCHABLE_KWARGS = frozenset({"background_data"})


class AbstractExplainer:
    """
    Root base class for all explainer adapters.

    Owns the shared interface: ``output_payload_kind`` class variable (default
    ``ATTRIBUTIONS``) and the ``check_backend_compat`` no-op default.

    Extend via ``AttributionOnlyExplainer`` when the framework should manage the
    full ``explain`` pipeline and you only need to implement ``compute_attributions``,
    or via ``FullExplainer`` when you own the entire ``explain`` pipeline yourself.
    """

    output_payload_kind: ClassVar[ExplanationPayloadKind] = ExplanationPayloadKind.ATTRIBUTIONS

    def check_backend_compat(self, backend: object) -> None:
        del backend
        return None


class AttributionOnlyExplainer(AbstractExplainer, ABC):
    """
    Explainer where you implement one step and the framework handles the rest.

    Subclasses implement :meth:`compute_attributions` only; batching, normalisation,
    result construction, and artifact persistence are provided by this class via
    :meth:`explain`.
    """

    def __init__(self) -> None:
        self.attributions: torch.Tensor | None = None

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
        raitap_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ExplanationResult:
        """
        Compute attributions (via :meth:`compute_attributions`), build an
        ``ExplanationResult``, write artifacts, and return it.
        """
        visualisers_list: list[ConfiguredVisualiser] = [] if visualisers is None else visualisers
        rk = {} if raitap_kwargs is None else dict(raitap_kwargs)
        batch_size = self._batch_size_from_raitap_kwargs(rk)
        show_progress, progress_desc = self._progress_settings_from_raitap_kwargs(rk)
        metadata_kwargs = {
            "sample_names": rk.get("sample_names"),
            "show_sample_names": bool(rk.get("show_sample_names", False)),
        }
        sample_ids = _normalise_optional_str_list(rk.get("sample_ids"))
        sample_display_names = _normalise_optional_str_list(rk.get("sample_names"))
        call_kwargs = dict(kwargs)
        # Validate the explainer registry before running potentially expensive attribution code.
        method_families = method_families_for_explainer(self)
        attributions = self._compute_with_optional_batches(
            model,
            inputs,
            call_kwargs,
            backend,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
        self.attributions = attributions
        input_spec = infer_input_spec(inputs, input_metadata=rk.get("input_metadata"))
        output_space = infer_output_space(
            input_spec=input_spec,
            attributions=attributions,
            explainer=self,
            method_families=method_families,
            layer_path=_layer_path_for_explainer(self),
        )
        _validate_output_space_shape(input_spec=input_spec, output_space=output_space)
        semantics = ExplanationSemantics(
            scope=ExplanationScope.LOCAL,
            scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
            payload_kind=self.output_payload_kind,
            method_families=method_families,
            target=ExplanationTarget(target=_normalise_target(call_kwargs.get("target"))),
            sample_selection=SampleSelection(
                sample_ids=sample_ids,
                sample_display_names=sample_display_names,
            ),
            input_spec=input_spec,
            output_space=output_space,
        )

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
            call_kwargs=call_kwargs,
            visualisers=visualisers_list,
            payload_kind=self.output_payload_kind,
            semantics=semantics,
        )
        explanation.write_artifacts()
        return explanation

    def _compute_with_optional_batches(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        attribution_kwargs: dict[str, Any],
        backend: object | None,
        *,
        batch_size: int | None = None,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ) -> torch.Tensor:
        if batch_size is None or inputs.shape[0] <= batch_size:
            prepared_inputs = self._prepare_inputs_for_backend(inputs, backend)
            attributions = self.compute_attributions(
                model,
                prepared_inputs,
                backend=backend,
                **attribution_kwargs,
            )
            self._reject_structured_attributions(attributions)
            normalised = self._normalise_attributions(attributions)
            del attributions, prepared_inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return normalised

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
            prepared_batch_inputs = self._prepare_inputs_for_backend(batch_inputs, backend)
            batch_kwargs = self._slice_kwargs_for_batch(attribution_kwargs, start, end, total_batch)
            chunk = self.compute_attributions(
                model,
                prepared_batch_inputs,
                backend=backend,
                **batch_kwargs,
            )
            self._reject_structured_attributions(chunk)
            # Normalise all attribution outputs to detached CPU tensors so batched and
            # unbatched runs return the same device semantics and persist consistently.
            chunks.append(self._normalise_attributions(chunk))
            del chunk, batch_inputs, prepared_batch_inputs, batch_kwargs
            # Clean up memory after each batch to avoid OOM errors in long runs
            # (e.g. with SHAP partition explainer).
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _prepare_inputs_for_backend(inputs: torch.Tensor, backend: object | None) -> torch.Tensor:
        if backend is None:
            return inputs
        prepare_inputs = getattr(backend, "_prepare_inputs", None)
        if not callable(prepare_inputs):
            return inputs
        return cast("torch.Tensor", prepare_inputs(inputs))

    @staticmethod
    def _normalise_attributions(attributions: torch.Tensor) -> torch.Tensor:
        return attributions.detach().cpu()

    @staticmethod
    def _reject_structured_attributions(attributions: object) -> None:
        if isinstance(attributions, (tuple, list)):
            raise TypeError(
                "tuple/list attribution outputs are not supported; convergence deltas are "
                "not first-class payloads yet."
            )

    @staticmethod
    def _batch_size_from_raitap_kwargs(raitap_kwargs: dict[str, Any]) -> int | None:
        candidate = raitap_kwargs.get("batch_size")
        if candidate is None:
            return None
        if not isinstance(candidate, int):
            raise TypeError(f"batch_size must be an int, got {type(candidate).__name__}.")
        if candidate <= 0:
            raise ValueError(f"batch_size must be > 0, got {candidate}.")
        return candidate

    @staticmethod
    def _progress_settings_from_raitap_kwargs(
        raitap_kwargs: dict[str, Any],
    ) -> tuple[bool, str | None]:
        show_progress_raw = raitap_kwargs.get("show_progress", True)
        if not isinstance(show_progress_raw, bool):
            raise TypeError(
                f"show_progress must be a bool, got {type(show_progress_raw).__name__}."
            )

        progress_desc = raitap_kwargs.get("progress_desc")
        if progress_desc is not None and not isinstance(progress_desc, str):
            raise TypeError(f"progress_desc must be a str, got {type(progress_desc).__name__}.")
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
        if key in _NON_BATCHABLE_KWARGS:
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
            model: PyTorch model to explain.
            inputs: Input tensor (shape depends on modality).
            **kwargs: Framework-specific keyword arguments (e.g. ``target``,
                ``baselines``, ``background_data``).

        Returns:
            Attribution tensor matching the input shape.
        """


def _normalise_optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _normalise_target(value: Any) -> int | str | list[int] | None:
    if value is None:
        return None
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.ndim == 0:
            return int(detached.item())
        return [int(item) for item in detached.flatten().tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return str(value)


def _layer_path_for_explainer(explainer: object) -> str | None:
    init_kwargs = getattr(explainer, "init_kwargs", None)
    if isinstance(init_kwargs, dict) and init_kwargs.get("layer_path") is not None:
        return str(init_kwargs["layer_path"])
    return None


def _validate_output_space_shape(*, input_spec: InputSpec, output_space: object) -> None:
    if getattr(output_space, "space", None) is not ExplanationOutputSpace.INPUT_FEATURES:
        return
    if input_spec.shape is None:
        return
    output_shape = getattr(output_space, "shape", None)
    if output_shape is None:
        return
    if tuple(output_shape) != tuple(input_spec.shape):
        raise ValueError(
            "Input-feature attribution output shape must match input metadata shape; "
            f"got attribution shape {tuple(output_shape)} and input shape {input_spec.shape}."
        )
