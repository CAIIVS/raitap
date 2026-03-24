from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class LocalMLflowServerConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    root_dir: str = "mlflow"
    database_name: str = "mlflow.db"
    artifact_dir: str = "artifacts"


@dataclass(frozen=True)
class LocalMLflowPaths:
    root_dir: Path
    database_path: Path
    artifact_dir: Path

    @property
    def backend_store_uri(self) -> str:
        return f"sqlite:///{self.database_path.resolve().as_posix()}"


def _load_simple_yaml_mapping(path: Path) -> dict[str, str | int]:
    data: dict[str, str | int] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid config line: {raw_line!r}")
        parsed = value.strip()
        if parsed.isdigit():
            data[key.strip()] = int(parsed)
        elif parsed.startswith(("'", '"')) and parsed.endswith(("'", '"')):
            data[key.strip()] = parsed[1:-1]
        else:
            data[key.strip()] = parsed
    return data


def load_local_mlflow_server_config() -> LocalMLflowServerConfig:
    resource = files("raitap.configs.tracking").joinpath("mlflow_server.yaml")
    with as_file(resource) as config_path:
        data = _load_simple_yaml_mapping(config_path)
    return LocalMLflowServerConfig(
        host=str(data["host"]),
        port=int(data["port"]),
        root_dir=str(data["root_dir"]),
        database_name=str(data["database_name"]),
        artifact_dir=str(data["artifact_dir"]),
    )


def local_mlflow_paths(
    project_root: Path | None = None,
    config: LocalMLflowServerConfig | None = None,
) -> LocalMLflowPaths:
    root = (project_root or Path.cwd()).resolve()
    server_config = config or load_local_mlflow_server_config()
    mlflow_dir = root / server_config.root_dir
    return LocalMLflowPaths(
        root_dir=mlflow_dir,
        database_path=mlflow_dir / server_config.database_name,
        artifact_dir=mlflow_dir / server_config.artifact_dir,
    )


def ensure_local_mlflow_dirs(paths: LocalMLflowPaths) -> None:
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)


def build_server_command(
    *,
    config: LocalMLflowServerConfig,
    project_root: Path | None = None,
) -> list[str]:
    paths = local_mlflow_paths(project_root, config)
    return [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--backend-store-uri",
        paths.backend_store_uri,
        "--default-artifact-root",
        str(paths.artifact_dir.resolve()),
    ]


def build_parser() -> argparse.ArgumentParser:
    defaults = load_local_mlflow_server_config()
    parser = argparse.ArgumentParser(
        description="Start a local MLflow server backed by ./mlflow/mlflow.db.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"Override server host (default from YAML: {defaults.host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Override server port (default from YAML: {defaults.port})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    project_root = Path.cwd().resolve()
    config = load_local_mlflow_server_config()
    if args.host is not None:
        config = LocalMLflowServerConfig(
            host=args.host,
            port=config.port,
            root_dir=config.root_dir,
            database_name=config.database_name,
            artifact_dir=config.artifact_dir,
        )
    if args.port is not None:
        config = LocalMLflowServerConfig(
            host=config.host,
            port=args.port,
            root_dir=config.root_dir,
            database_name=config.database_name,
            artifact_dir=config.artifact_dir,
        )
    paths = local_mlflow_paths(project_root, config)
    ensure_local_mlflow_dirs(paths)
    command = build_server_command(
        config=config,
        project_root=project_root,
    )
    try:
        return subprocess.run(command, check=False, cwd=project_root).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
