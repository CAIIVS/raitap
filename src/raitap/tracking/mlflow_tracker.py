"""
MLFlow tracking implementation for RAITAP
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from raitap import raitap_log
from raitap.configs import cfg_to_dict, resolve_run_dir
from raitap.models.backend import OnnxBackend, TorchBackend
from raitap.utils.diagnostics import Module

from .base_tracker import BaseTracker
from .process_registry import (
    announce_stop_hint,
    pop_entries_for_tracker,
    register_background_process,
    reinsert_entries,
    watch_port,
)
from .registration import register_tracker

DEFAULT_MLFLOW_BACKEND_STORE_URI = "sqlite:///mlflow/mlflow.db"
DEFAULT_MLFLOW_ARTIFACT_ROOT = "./mlflow/artifacts"
DEFAULT_MLFLOW_UI_HOST = "127.0.0.1"
DEFAULT_MLFLOW_UI_PORT = 5000
MLFLOW_SERVER_READY_TIMEOUT = 30.0
_NOISY_HTTP_LOGGERS = ("urllib3.connectionpool", "urllib3.retry")


def _route_mlflow_logs_to_root() -> None:
    """Drop mlflow's own handlers/formatter so logs render via the raitap rich handler."""
    import logging as _logging

    for name in (
        "mlflow",
        "mlflow.tracking",
        "mlflow.tracking.fluent",
        "mlflow.utils",
        "mlflow.store",
    ):
        lg = _logging.getLogger(name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        lg.propagate = True


@contextmanager
def _silence_http_retry_noise() -> Iterator[None]:
    """Suppress urllib3's per-retry WARNING log spam during expected startup races."""
    import logging as _logging

    saved: list[tuple[_logging.Logger, int]] = []
    for name in _NOISY_HTTP_LOGGERS:
        lg = _logging.getLogger(name)
        saved.append((lg, lg.level))
        lg.setLevel(_logging.ERROR)
    try:
        yield
    finally:
        for lg, level in saved:
            lg.setLevel(level)


if TYPE_CHECKING:
    from collections.abc import Iterator

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


@register_tracker(registry_name="mlflow", extra="mlflow")
class MLFlowTracker(BaseTracker):
    @classmethod
    def stop_detached(cls, timeout: float = 5.0) -> tuple[int, int]:
        """Kill the local MLflow server/UI tree this user has spawned or reused.

        MLflow has no remote shutdown API (no ``mlflow stop`` CLI, no HTTP
        ``/shutdown``), so the helper PIDs are torn down directly. Forked
        ``multiprocessing`` workers re-parent after the root dies on Windows,
        so we walk the process tree before SIGTERM.
        """
        from raitap.utils.process import (
            is_pid_alive,
            is_port_listening,
            pids_listening_on_port,
            terminate_pid_tree,
        )

        entries, ports = pop_entries_for_tracker(cls.tracker_name())
        if not entries and not ports:
            return (0, 0)

        killed = 0
        skipped = 0
        failed_entries: list[Any] = []
        failed_ports: list[Any] = []

        from urllib.parse import urlparse

        for entry in entries:
            target_pids: list[int] = []
            if is_pid_alive(entry.pid):
                target_pids.append(entry.pid)

            # MLflow's CLI parent often exits while multiprocessing workers stay
            # bound to the port. Fall back to URL-derived port lookup so the
            # actual listeners get killed instead of the long-gone parent PID.
            parsed = urlparse(entry.url) if entry.url else None
            if (
                parsed
                and parsed.hostname
                and parsed.port
                and is_port_listening(parsed.hostname, parsed.port)
            ):
                for pid in pids_listening_on_port(parsed.port):
                    if pid not in target_pids:
                        target_pids.append(pid)

            if not target_pids:
                skipped += 1
                continue

            entry_failed = False
            for pid in target_pids:
                if terminate_pid_tree(pid, timeout=timeout):
                    killed += 1
                    raitap_log.info("Stopped %s (PID %d)", entry.label, pid)
                else:
                    entry_failed = True
            # Port-based rescue: on Windows, ``taskkill`` returns before the
            # process table reflects the death. If the helper's port is no
            # longer listening, the service is gone even if liveness lingered.
            port_alive = bool(
                parsed
                and parsed.hostname
                and parsed.port
                and is_port_listening(parsed.hostname, parsed.port)
            )
            if entry_failed and not port_alive:
                entry_failed = False
            if entry_failed:
                raitap_log.warn(
                    "%s on %s still alive after stop attempt",
                    entry.label,
                    entry.url or f"PID {entry.pid}",
                    module=Module.tracking,
                )
                failed_entries.append(entry)

        for port in ports:
            if not is_port_listening(port.host, port.port):
                skipped += 1
                continue
            pids = pids_listening_on_port(port.port)
            if not pids:
                raitap_log.warn(
                    "Could not resolve PID listening on %s:%d (%s)",
                    port.host,
                    port.port,
                    port.label,
                    module=Module.tracking,
                )
                failed_ports.append(port)
                continue
            port_failed = False
            for pid in pids:
                if terminate_pid_tree(pid, timeout=timeout):
                    killed += 1
                    raitap_log.info("Stopped %s on port %d (PID %d)", port.label, port.port, pid)
                else:
                    port_failed = True
            # Re-check the port: even if a stale PID lingers in the table,
            # the helper is gone once it stops listening.
            if port_failed and not is_port_listening(port.host, port.port):
                port_failed = False
            if port_failed:
                raitap_log.warn(
                    "%s on %s:%d still listening after stop attempt",
                    port.label,
                    port.host,
                    port.port,
                    module=Module.tracking,
                )
                failed_ports.append(port)

        if failed_entries or failed_ports:
            reinsert_entries(failed_entries, failed_ports)
        return killed, skipped

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

        _route_mlflow_logs_to_root()

        tracking_conf = _tracking_dict(config)
        configured_backend_store_uri = _param_str(tracking_conf.get("backend_store_uri"))
        configured_default_artifact_root = _param_str(tracking_conf.get("default_artifact_root"))
        configured_output_forwarding_url = _param_str(tracking_conf.get("output_forwarding_url"))

        self._backend_store_uri_configured = configured_backend_store_uri is not None
        self._default_artifact_root_configured = configured_default_artifact_root is not None
        self._output_forwarding_url_configured = configured_output_forwarding_url is not None

        self.backend_store_uri: str = (
            configured_backend_store_uri or DEFAULT_MLFLOW_BACKEND_STORE_URI
        )
        self.default_artifact_root: str = (
            configured_default_artifact_root or DEFAULT_MLFLOW_ARTIFACT_ROOT
        )
        self.tracking_uri: str = configured_output_forwarding_url or self.backend_store_uri

        # Track spawned subprocesses for cleanup
        self._server_process: Any = None
        self._ui_process: Any = None
        self._run_ui_path: str | None = None

        if not self.tracking_uri.startswith("http"):
            if self._output_forwarding_url_configured:
                if self._sqlite_file_path(self.tracking_uri) is not None:
                    self._prepare_local_mlflow_paths(
                        backend_store_uri=self.tracking_uri,
                        artifact_root=(
                            self.default_artifact_root
                            if self._default_artifact_root_configured
                            else None
                        ),
                    )
                else:
                    self._prepare_local_mlflow_paths(
                        file_store_uri=self.tracking_uri,
                        artifact_root=(
                            self.default_artifact_root
                            if self._default_artifact_root_configured
                            else None
                        ),
                    )
            else:
                self._prepare_local_mlflow_paths(
                    backend_store_uri=self.backend_store_uri,
                    artifact_root=self.default_artifact_root,
                )
        with _silence_http_retry_noise():
            self._ensure_server_running()

            self._mlflow.set_tracking_uri(self.tracking_uri)
            self._ensure_direct_store_experiment()
            self._mlflow.set_experiment(config.experiment_name)
            run = self._mlflow.start_run(run_name=config.experiment_name)
        self._run_ui_path = self._build_run_ui_path(run)
        announce_stop_hint()

    def terminate(self, successfully: bool = True) -> None:
        from mlflow.entities import RunStatus

        status = RunStatus.FINISHED if successfully else RunStatus.FAILED
        self._mlflow.end_run(status=RunStatus.to_string(status))

        if _tracking_dict(self.config).get("open_when_done", True):
            try:
                self._open_mlflow_ui()
            except Exception:
                self._cleanup_subprocesses()
                raise
            return

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

        if self._is_localhost(host) and self._is_port_open(host, port):
            # Reusing a server we did not spawn this run — record the port so
            # ``raitap tracking stop`` can still find and terminate it.
            watch_port(
                port,
                host=host,
                tracker="MLFlowTracker",
                label="MLflow server",
                url=self.tracking_uri,
            )
            return

        if self._is_localhost(host) and not self._is_port_open(host, port):
            self._prepare_local_mlflow_paths(
                backend_store_uri=self.backend_store_uri,
                artifact_root=self.default_artifact_root,
            )
            raitap_log.info(
                "MLflow server not running. Starting server at %s...", self.tracking_uri
            )
            self._server_process = subprocess.Popen(
                [
                    "mlflow",
                    "server",
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--backend-store-uri",
                    self.backend_store_uri,
                    "--default-artifact-root",
                    self.default_artifact_root,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            register_background_process(
                self._server_process.pid,
                tracker="MLFlowTracker",
                label="MLflow server",
                url=self.tracking_uri,
            )

            if not self._wait_for_http_ready(
                self.tracking_uri, timeout=MLFLOW_SERVER_READY_TIMEOUT
            ):
                raitap_log.warn(
                    "MLflow server may not be ready at %s:%d after %.0f seconds",
                    host,
                    port,
                    MLFLOW_SERVER_READY_TIMEOUT,
                    module=Module.tracking,
                )
            else:
                raitap_log.info("MLflow server is ready at %s:%d", host, port)

    def _open_mlflow_ui(self) -> None:
        import subprocess
        import webbrowser

        if self.tracking_uri.startswith("http"):
            url = self._mlflow_ui_url(self.tracking_uri)
            raitap_log.info("Opening MLflow UI at %s", url)
            webbrowser.open(url)
        else:
            host = DEFAULT_MLFLOW_UI_HOST
            port = DEFAULT_MLFLOW_UI_PORT
            url = f"http://{host}:{port}"

            if not self._is_port_open(host, port):
                raitap_log.info("Starting MLflow UI at %s", url)
                raitap_log.info("Backend store: %s", self.tracking_uri)

                self._ui_process = subprocess.Popen(
                    ["mlflow", "ui", "--backend-store-uri", self.tracking_uri, "--port", str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                register_background_process(
                    self._ui_process.pid,
                    tracker="MLFlowTracker",
                    label="MLflow UI",
                    url=url,
                )

                if not self._wait_for_http_ready(url, timeout=MLFLOW_SERVER_READY_TIMEOUT):
                    raitap_log.warn(
                        "MLflow UI may not be ready at %s after %.0f seconds",
                        url,
                        MLFLOW_SERVER_READY_TIMEOUT,
                        module=Module.tracking,
                    )
                else:
                    raitap_log.info("MLflow UI is ready at %s", url)
            else:
                raitap_log.warn(
                    "Reusing existing MLflow UI at %s; intended backend store is %s, "
                    "but the existing UI may use a different backend store.",
                    url,
                    self.tracking_uri,
                    module=Module.tracking,
                )

            webbrowser.open(self._mlflow_ui_url(url))

    @staticmethod
    def _build_run_ui_path(run: Any) -> str | None:
        run_info = getattr(run, "info", None)
        experiment_id = getattr(run_info, "experiment_id", None)
        run_id = getattr(run_info, "run_id", None)
        if not isinstance(experiment_id, str) or not isinstance(run_id, str):
            return None
        if not experiment_id or not run_id:
            return None
        return f"#/experiments/{experiment_id}/runs/{run_id}"

    def _mlflow_ui_url(self, base_url: str) -> str:
        if self._run_ui_path is None:
            return base_url
        return f"{base_url.rstrip('/')}/{self._run_ui_path}"

    def _prepare_local_mlflow_paths(
        self,
        *,
        backend_store_uri: str | None = None,
        artifact_root: str | None = None,
        file_store_uri: str | None = None,
    ) -> None:
        if backend_store_uri is not None:
            sqlite_path = self._sqlite_file_path(backend_store_uri)
            if sqlite_path is not None:
                sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        for uri in (artifact_root, file_store_uri):
            if uri is None:
                continue
            path = self._local_artifact_path(uri)
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)

    def _ensure_direct_store_experiment(self) -> None:
        if self.tracking_uri.startswith("http"):
            return

        experiment = self._mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            if not self._output_forwarding_url_configured or self._default_artifact_root_configured:
                self._mlflow.create_experiment(
                    self.config.experiment_name,
                    artifact_location=self.default_artifact_root,
                )
            else:
                self._mlflow.create_experiment(self.config.experiment_name)

    def _wait_for_http_ready(self, base_url: str, timeout: float) -> bool:
        """Poll mlflow's ``/health`` endpoint until it returns 2xx or timeout elapses.

        TCP-port-open is not enough: the Flask app keeps refusing connections briefly
        after ``bind`` succeeds, which is exactly the window urllib3 retries log on.
        """
        import time
        from urllib.error import HTTPError, URLError
        from urllib.request import urlopen

        url = f"{base_url.rstrip('/')}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urlopen(url, timeout=1) as response:
                    if 200 <= response.status < 300:
                        return True
            except HTTPError as e:
                if 200 <= e.code < 500:
                    return True
            except (URLError, ConnectionError, TimeoutError, OSError):
                # Expected during startup: server has bound the port but the
                # Flask app isn't accepting connections yet. Retry until the
                # outer timeout expires.
                pass
            time.sleep(0.25)
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
                        raitap_log.debug("Terminating %s (PID: %d)", process_name, process.pid)
                        process.terminate()

                        # Give it a moment to terminate gracefully
                        try:
                            process.wait(timeout=2)
                        except Exception:
                            # Force kill if it doesn't terminate
                            raitap_log.debug(
                                "Force killing %s (PID: %d)", process_name, process.pid
                            )
                            process.kill()
                            process.wait()
                except Exception as e:
                    raitap_log.debug("Error cleaning up %s: %s", process_name, e)

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

    @staticmethod
    def _sqlite_file_path(uri: str) -> Path | None:
        if not uri.startswith("sqlite:///"):
            return None
        raw_path = uri.removeprefix("sqlite:///")
        if raw_path in {"", ":memory:"}:
            return None
        return Path(raw_path)

    @staticmethod
    def _local_artifact_path(uri: str) -> Path | None:
        if "://" in uri and not uri.startswith("file://"):
            return None
        if uri.startswith("file://"):
            uri = uri.removeprefix("file://")
        return Path(uri)
