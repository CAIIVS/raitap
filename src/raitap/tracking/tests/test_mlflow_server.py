from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from raitap.tracking.mlflow_server import (
    LocalMLflowServerConfig,
    build_server_command,
    ensure_local_mlflow_dirs,
    load_local_mlflow_server_config,
    local_mlflow_paths,
)


def test_local_mlflow_paths_are_under_project_root(tmp_path: Path) -> None:
    paths = local_mlflow_paths(tmp_path)

    assert paths.root_dir == tmp_path / "mlflow"
    assert paths.database_path == tmp_path / "mlflow" / "mlflow.db"
    assert paths.artifact_dir == tmp_path / "mlflow" / "artifacts"
    assert paths.backend_store_uri == f"sqlite:///{paths.database_path.resolve().as_posix()}"


def test_ensure_local_mlflow_dirs_creates_required_directories(tmp_path: Path) -> None:
    paths = local_mlflow_paths(tmp_path)

    ensure_local_mlflow_dirs(paths)

    assert paths.root_dir.is_dir()
    assert paths.artifact_dir.is_dir()


def test_build_server_command_uses_python_module_and_local_paths(tmp_path: Path) -> None:
    config = LocalMLflowServerConfig()
    command = build_server_command(config=config, project_root=tmp_path)
    paths = local_mlflow_paths(tmp_path, config)

    assert command[:4] == [sys.executable, "-m", "mlflow", "server"]
    assert "--host" in command and config.host in command
    assert "--port" in command and str(config.port) in command
    assert "--backend-store-uri" in command and paths.backend_store_uri in command
    assert "--default-artifact-root" in command
    assert str(paths.artifact_dir.resolve()) in command


def test_load_local_mlflow_server_config_reads_yaml_defaults() -> None:
    config = load_local_mlflow_server_config()

    assert config == LocalMLflowServerConfig()
