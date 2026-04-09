from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Sphinx live-reload preview server for the local docs site."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind the preview server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the preview server to.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the preview in a browser when the server starts.",
    )
    return parser


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root = project_root()
    docs_dir = root / "docs"
    build_dir = docs_dir / "_build" / "html"

    command = [
        sys.executable,
        "-m",
        "sphinx_autobuild",
        str(docs_dir),
        str(build_dir),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    if args.open_browser:
        command.append("--open-browser")

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
