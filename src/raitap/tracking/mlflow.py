"""
MLFlow tracking implementation for RAITAP
"""

import importlib
from pathlib import Path
from typing import Any

from ..configs.factory_utils import cfg_to_dict
from .base import AssessmentContext, Tracker


class MLFlowTracker(Tracker):
    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        experiment_name: str = "raitap-assessment",
        log_model: bool = False,
        registry_enabled: bool = False,
        registered_model_name: str | None = None,
    ):
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.experiment_name = experiment_name
        self.log_model_enabled = log_model
        self.registry_enabled = registry_enabled
        self.registered_model_name = registered_model_name

        self._mlflow: Any | None = None
        self._active_run = False  # Flag if there is an active run initiated in start_assessment

    def _require_mlflow(self) -> Any:
        """
        Lazy import mlflow
        """
        if self._mlflow is None:
            try:
                importlib.import_module("mlflow")
            except ImportError as e:
                raise ImportError(
                    "MLFlow tracking is enabled but mlflow is not installed. "
                    "Install it with `uv sync --extra mlflow`."
                ) from e
        return self._mlflow

    def start_assessment(self, context: AssessmentContext) -> None:
        mlflow = self._require_mlflow()

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)

        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=context.assessment_name)

        mlflow.set_tags(
            {
                "assessment.name": context.assessment_name,
                "model.source": str(context.model_source or ""),
                "data.name": context.data_name,
                "data.source": str(context.data_source or ""),
                "run.output_dir": str(context.output_dir) if context.output_dir else "",
            }
        )

        self._active_run = True

    def finalize(self, status: str = "FINISHED") -> None:
        if not self._active_run:  # No active run, nothing to do
            return
        self._require_mlflow().end_run(status=status)
        self._active_run = False

    def log_config(self, config: Any) -> None:
        mlflow = self._require_mlflow()
        # TODO: Improve this very basic logging
        mlflow.log_params(cfg_to_dict(config))

    def log_transparency(self, results: dict[str, Any]) -> None:
        run_dir = results.get("run_dir")

        if isinstance(run_dir, str):
            run_dir = Path(run_dir)

        if isinstance(run_dir, Path) and run_dir.exists():
            mlflow = self._require_mlflow()
            mlflow.log_artifacts(
                str(run_dir),
                artifact_path="transparency",
            )

    def log_dataset(self, dataset_info: dict[str, Any], artifact_path: str = "dataset") -> None:
        mlflow = self._require_mlflow()
        mlflow.log_dict(dataset_info, f"{artifact_path}/dataset.json")

    def log_metrics(self, result: dict[str, Any], prefix: str = "performance") -> None:
        """
        Logs performance metrics and artifacts from a given result dictionary
        to an MLflow tracking server.

        The method extracts metrics and logs them as scalar values
        with an optional prefix. It also logs artifacts from
        a specified directory. This method is designed for logging
        model performance or evaluation results.

        :param result: A dictionary containing the result data to log. The supported keys are:
            - "result": An object containing a "metrics" attribute of type dict[str, Any].
            - "run_dir": A string or Path object representing the directory containing
                        artifacts for logging.
        :param prefix: A string prefix to prepend to the metric keys when logged in MLFlow.
                        Defaults to "performance".
        :return: This method does not return a value.
        """
        mlflow = self._require_mlflow()
        metric_result = result.get("result")
        metrics = getattr(metric_result, "metrics", {})

        if isinstance(metrics, dict):
            scalar_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, int | float | bool):
                    scalar_metrics[f"{prefix}.{k}"] = float(v)
            if scalar_metrics:
                mlflow.log_metrics(scalar_metrics)

        run_dir = result.get("run_dir")
        if isinstance(run_dir, str):
            run_dir = Path(run_dir)

        if isinstance(run_dir, Path) and run_dir.exists():
            mlflow.log_artifacts(str(run_dir), artifact_path="metrics")

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        if not self.log_model_enabled:
            return

        mlflow_pytorch = importlib.import_module("mlflow.pytorch")
        mlflow_pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            registered_model_name=(self.registered_model_name if self.registry_enabled else None),
        )
