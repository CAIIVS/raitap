"""
MLFlow tracking implementation for RAITAP
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from raitap.configs import cfg_to_dict, resolve_run_dir
from raitap.models.backend import OnnxBackend, TorchBackend

from .base_tracker import BaseTracker

logger = logging.getLogger(__name__)

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


def _tracking_dict(config: Any) -> dict[str, Any]:
    """
    Plain dict for tracking options (Hydra DictConfig, dataclass, or SimpleNamespace in tests).
    """
    raw = cfg_to_dict(config).get("tracking")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "__dict__"):
        return dict(vars(raw))
    return dict(raw)


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

            self._mlflow = mlflow
        except ImportError as e:
            raise ImportError(
                "MLFlow tracking is enabled but mlflow is not installed. "
                "Install it with `uv sync --extra mlflow`."
            ) from e

        tracking_conf = _tracking_dict(config)
        self.tracking_uri: str = tracking_conf.get("output_forwarding_url") or "./mlruns"

        # Track spawned subprocesses for cleanup
        self._server_process: Any = None
        self._ui_process: Any = None

        self._ensure_server_running()

        self._mlflow.set_tracking_uri(self.tracking_uri)
        self._mlflow.set_experiment(config.experiment_name)
        self._mlflow.start_run(run_name=config.experiment_name)

    def terminate(self, successfully: bool = True) -> None:
        from mlflow.entities import RunStatus

        status = RunStatus.FINISHED if successfully else RunStatus.FAILED
        self._mlflow.end_run(status=RunStatus.to_string(status))

        if _tracking_dict(self.config).get("open_when_done", True):
            self._open_mlflow_ui()

        self._cleanup_subprocesses()

    def log_config(self) -> None:
        config_dict = cfg_to_dict(self.config)
        self._mlflow.log_dict(config_dict, "config/config.json")

        params = _mlflow_summary_params(config_dict)
        if params:
            self._mlflow.log_params(params)

    def log_dataset(self, description: dict[str, Any]) -> None:
        self._mlflow.log_dict(description, "dataset.json")

    def log_artifacts(
        self, source_directory: str | Path | None, target_subdirectory: str | None = None
    ) -> None:
        self._mlflow.log_artifacts(str(source_directory), artifact_path=target_subdirectory)

    def log_metrics(
        self,
        metrics: dict[str, float],
        prefix: str = "performance",
    ) -> None:
        prefixed_metrics = {f"{prefix}.{key}": value for key, value in metrics.items()}
        self._mlflow.log_metrics(prefixed_metrics)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        if isinstance(model, OnnxBackend):
            self._log_onnx_model(model, artifact_path)
            return

        if isinstance(model, TorchBackend):
            pytorch_model = model.as_model_for_explanation()
        else:
            pytorch_model = model

        self._mlflow.pytorch.log_model(  # type: ignore[attr-defined]
            pytorch_model=pytorch_model,
            artifact_path=artifact_path,
        )

    def _log_onnx_model(self, model: OnnxBackend, artifact_path: str) -> None:
        if model.model_path is None:
            raise ValueError("OnnxBackend cannot be logged because its source path is unknown.")

        try:
            import onnx
        except ImportError as error:
            raise ImportError(
                "MLflow ONNX logging requires the `onnx` package. "
                "Install it with `uv sync --extra onnx-cpu`, `uv sync --extra onnx-cuda`, "
                "or `uv sync --extra onnx-intel`."
            ) from error

        self._mlflow.onnx.log_model(  # type: ignore[attr-defined]
            onnx_model=onnx.load(model.model_path),
            artifact_path=artifact_path,
        )

    def _ensure_server_running(self) -> None:
        import subprocess
        from urllib.parse import urlparse

        if not self.tracking_uri.startswith("http"):
            return

        parsed = urlparse(self.tracking_uri)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 5000

        if self._is_localhost(host) and not self._is_port_open(host, port):
            logger.info("MLflow server not running. Starting server at %s...", self.tracking_uri)
            self._server_process = subprocess.Popen(
                ["mlflow", "server", "--host", host, "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            if not self._wait_for_port_ready(host, port, timeout=10):
                logger.warning(
                    "MLflow server may not be ready at %s:%d after 10 seconds", host, port
                )
            else:
                logger.info("MLflow server is ready at %s:%d", host, port)

    def _open_mlflow_ui(self) -> None:
        import subprocess
        import webbrowser

        if self.tracking_uri.startswith("http"):
            logger.info("Opening MLflow UI at %s", self.tracking_uri)
            webbrowser.open(self.tracking_uri)
        else:
            port = 5000
            url = f"http://127.0.0.1:{port}"

            if not self._is_port_open("127.0.0.1", port):
                logger.info("Starting MLflow UI at %s", url)
                logger.info("Backend store: %s", self.tracking_uri)

                self._ui_process = subprocess.Popen(
                    ["mlflow", "ui", "--backend-store-uri", self.tracking_uri, "--port", str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

                if not self._wait_for_port_ready("127.0.0.1", port, timeout=10):
                    logger.warning("MLflow UI may not be ready at %s after 10 seconds", url)
                else:
                    logger.info("MLflow UI is ready at %s", url)

            webbrowser.open(url)

    def _wait_for_port_ready(self, host: str, port: int, timeout: float = 10) -> bool:
        """
        Wait for a port to become ready, with retries.

        Args:
            host: Hostname to check
            port: Port number to check
            timeout: Maximum time to wait in seconds

        Returns:
            True if port became ready, False if timeout reached
        """
        import time

        start_time = time.time()
        retry_interval = 0.5

        while time.time() - start_time < timeout:
            if self._is_port_open(host, port):
                return True
            time.sleep(retry_interval)

        return False

    def _cleanup_subprocesses(self) -> None:
        """Clean up any subprocesses spawned by this tracker."""
        for process_name, process in [
            ("MLflow server", self._server_process),
            ("MLflow UI", self._ui_process),
        ]:
            if process is not None:
                try:
                    # Check if process is still running
                    if process.poll() is None:
                        logger.debug("Terminating %s (PID: %d)", process_name, process.pid)
                        process.terminate()

                        # Give it a moment to terminate gracefully
                        try:
                            process.wait(timeout=2)
                        except Exception:
                            # Force kill if it doesn't terminate
                            logger.debug("Force killing %s (PID: %d)", process_name, process.pid)
                            process.kill()
                            process.wait()
                except Exception as e:
                    logger.debug("Error cleaning up %s: %s", process_name, e)

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
