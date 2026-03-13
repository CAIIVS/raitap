"""
MLFlow tracking implementation for RAITAP
"""

import importlib
from pathlib import Path
from typing import Any

from raitap.configs.factory_utils import cfg_to_dict

from .base import Tracker


class MLFlowTracker(Tracker):
    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        log_model: bool = False,
        registry_enabled: bool = False,
        registered_model_name: str | None = None,
    ):
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.log_model_enabled = log_model
        self.registry_enabled = registry_enabled
        self.registered_model_name = registered_model_name

        self._mlflow: Any | None = None
        self._active_run = False  # Flag if there is an active run initiated in start_assessment

    @staticmethod
    def _config_params(config_dict: dict[str, Any]) -> dict[str, str]:
        params = {
            "assessment.name": str(config_dict.get("experiment_name", "")),
            "model.source": str(config_dict.get("model", {}).get("source", "")),
            "data.name": str(config_dict.get("data", {}).get("name", "")),
            "data.source": str(config_dict.get("data", {}).get("source", "")),
            "transparency.algorithm": str(config_dict.get("transparency", {}).get("algorithm", "")),
        }
        return {key: value for key, value in params.items() if value}

    def _require_mlflow(self) -> Any:
        """
        Lazy import mlflow
        """
        if self._mlflow is None:
            try:
                self._mlflow = importlib.import_module("mlflow")
            except ImportError as e:
                raise ImportError(
                    "MLFlow tracking is enabled but mlflow is not installed. "
                    "Install it with `uv sync --extra mlflow`."
                ) from e
        return self._mlflow

    def start_assessment(self, assessment_name: str) -> None:
        mlflow = self._require_mlflow()

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)

        mlflow.set_experiment(assessment_name)
        mlflow.start_run(run_name=assessment_name)
        self._active_run = True

    def finalize(self, status: str = "FINISHED") -> None:
        if not self._active_run:  # No active run, nothing to do
            return
        self._require_mlflow().end_run(status=status)
        self._active_run = False

    def log_config(self, config: Any) -> None:
        mlflow = self._require_mlflow()
        config_dict = cfg_to_dict(config)
        mlflow.log_dict(config_dict, "config/config.json")

        params = self._config_params(config_dict)
        if params:
            mlflow.log_params(params)

    def log_dataset(self, dataset_info: dict[str, Any], artifact_path: str = "dataset") -> None:
        mlflow = self._require_mlflow()
        mlflow.log_dict(dataset_info, f"{artifact_path}/dataset.json")

    def log_artifacts(self, local_dir: str | Path, artifact_path: str) -> None:
        if isinstance(local_dir, str):
            local_dir = Path(local_dir)

        if isinstance(local_dir, Path) and local_dir.exists():
            self._require_mlflow().log_artifacts(str(local_dir), artifact_path=artifact_path)

    def log_metrics(
        self,
        metrics: dict[str, float | int | bool],
        prefix: str = "performance",
    ) -> None:
        scalar_metrics = {
            f"{prefix}.{key}": float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float, bool))
        }
        if scalar_metrics:
            self._require_mlflow().log_metrics(scalar_metrics)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        if not self.log_model_enabled:
            return

        mlflow_pytorch = importlib.import_module("mlflow.pytorch")
        mlflow_pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            registered_model_name=(self.registered_model_name if self.registry_enabled else None),
        )
