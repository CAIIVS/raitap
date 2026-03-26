"""
MLFlow tracking implementation for RAITAP
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlflow.entities import RunStatus

from raitap.configs.factory_utils import cfg_to_dict, resolve_run_dir

from ..base_tracker import BaseTracker

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig


def _nested_dict(obj: Any) -> dict[str, Any] | None:
    return obj if isinstance(obj, dict) else None


def _param_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _mlflow_summary_params(config_dict: dict[str, Any]) -> dict[str, str]:
    """
    Build a flat string map of high-signal run parameters for MLflow search and comparison.

    Walks the normalised config from :func:`~raitap.configs.factory_utils.cfg_to_dict`
    instead of hard-coding brittle ``.get`` chains. Transparency is a mapping of
    explainer name → explainer config; each explainer contributes
    ``transparency.<name>.algorithm`` and ``transparency.<name>._target_``.
    """
    out: dict[str, str] = {}

    def put(key: str, value: Any) -> None:
        s = _param_str(value)
        if s is not None:
            out[key] = s

    put("assessment.name", config_dict.get("experiment_name"))

    model = _nested_dict(config_dict.get("model"))
    if model is not None:
        put("model.source", model.get("source"))

    data = _nested_dict(config_dict.get("data"))
    if data is not None:
        put("data.name", data.get("name"))
        put("data.source", data.get("source"))

    transparency = _nested_dict(config_dict.get("transparency"))
    if transparency:
        put("transparency.explainers", ",".join(sorted(transparency)))
        for name in sorted(transparency):
            explainer = _nested_dict(transparency[name])
            if explainer is None:
                continue
            prefix = f"transparency.{name}"
            put(f"{prefix}.algorithm", explainer.get("algorithm"))
            put(f"{prefix}._target_", explainer.get("_target_"))

    return out


class MLFlowTracker(BaseTracker):
    def __init__(self, config: AppConfig):
        self.output_dir = resolve_run_dir(config)
        self.config = config

        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "MLFlow tracking is enabled but mlflow is not installed. "
                "Install it with `uv sync --extra mlflow`."
            ) from e

        self.tracking_uri: str = config.tracking.output_forwarding_url or "./mlruns"

        self._ensure_server_running()

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        mlflow.start_run(run_name=config.experiment_name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self.terminate(successfully=exc_type is None)
        return False

    def terminate(self, successfully: bool = True) -> None:
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "MLFlow tracking is enabled but mlflow is not installed. "
                "Install it with `uv sync --extra mlflow`."
            ) from e

        status = RunStatus.FINISHED if successfully else RunStatus.FAILED
        mlflow.end_run(status=RunStatus.to_string(status))

        if self.config.tracking.open_when_done:
            self._open_mlflow_ui()

    def log_config(self) -> None:
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "MLFlow tracking is enabled but mlflow is not installed. "
                "Install it with `uv sync --extra mlflow`."
            ) from e

        config_dict = cfg_to_dict(self.config)
        mlflow.log_dict(config_dict, "config/config.json")

        params = _mlflow_summary_params(config_dict)
        if params:
            mlflow.log_params(params)

    def log_dataset(self, description: dict[str, Any]) -> None:
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "MLFlow tracking is enabled but mlflow is not installed. "
                "Install it with `uv sync --extra mlflow`."
            ) from e

        mlflow.log_dict(description, "dataset.json")

    def log_artifacts(
        self, source_directory: str | Path | None, target_subdirectory: str | None = None
    ) -> None:
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "MLFlow tracking is enabled but mlflow is not installed. "
                "Install it with `uv sync --extra mlflow`."
            ) from e

        mlflow.log_artifacts(str(source_directory), artifact_path=target_subdirectory)

    def log_metrics(
        self,
        metrics: dict[str, float],
        prefix: str = "performance",
    ) -> None:
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "MLFlow tracking is enabled but mlflow is not installed. "
                "Install it with `uv sync --extra mlflow`."
            ) from e

        mlflow.log_metrics(metrics)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        try:
            import mlflow.pytorch  # type: ignore[attr-defined]
        except ImportError as e:
            raise ImportError(
                "MLFlow tracking is enabled but mlflow is not installed. "
                "Install it with `uv sync --extra mlflow`."
            ) from e

        mlflow.pytorch.log_model(  # type: ignore[attr-defined]
            pytorch_model=model,
            artifact_path=artifact_path,
            # registered_model_name=(self.config.tracking.registered_model_name if self.config.tracking.registry_enabled else None),  # noqa: E501
        )

    def _ensure_server_running(self) -> None:
        import subprocess
        import time
        from urllib.parse import urlparse

        if not self.tracking_uri.startswith("http"):
            return

        parsed = urlparse(self.tracking_uri)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 5000

        if self._is_localhost(host) and not self._is_port_open(host, port):
            print(f"MLflow server not running. Starting server at {self.tracking_uri}...")
            subprocess.Popen(
                ["mlflow", "server", "--host", host, "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            print("Waiting for server to start...")
            time.sleep(3)

    def _open_mlflow_ui(self) -> None:
        import subprocess
        import time
        import webbrowser

        if self.tracking_uri.startswith("http"):
            print(f"\nOpening MLflow UI at {self.tracking_uri}")
            webbrowser.open(self.tracking_uri)
        else:
            port = 5000
            url = f"http://127.0.0.1:{port}"

            if not self._is_port_open("127.0.0.1", port):
                print(f"\nStarting MLflow UI at {url}")
                print(f"Backend store: {self.tracking_uri}")

                subprocess.Popen(
                    ["mlflow", "ui", "--backend-store-uri", self.tracking_uri, "--port", str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

                print("Waiting for UI to start...")
                time.sleep(3)

            webbrowser.open(url)

    @staticmethod
    def _is_localhost(host: str) -> bool:
        return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")

    @staticmethod
    def _is_port_open(host: str, port: int) -> bool:
        import socket

        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (TimeoutError, ConnectionRefusedError, OSError):
            return False
