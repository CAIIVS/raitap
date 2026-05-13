"""``raitap-deps`` CLI: infer uv extras from a Hydra config and run uv.

Examples::

    raitap-deps                                   # default config, exec
    raitap-deps --dry-run                         # print only
    raitap-deps --config-dir examples/lwise-ham10000 \\
                --config-name assessment
    raitap-deps --hardware xpu --mode add data=mnist_samples model=mlp_mnist
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from raitap.configs.extras.command import render_command, select_mode
from raitap.configs.extras.conflicts import ExtrasConflictError, validate_conflicts
from raitap.configs.extras.inference import infer_extras
from raitap.configs.extras.probe import detect_hardware
from raitap.utils.diagnostics import is_dev_install

if TYPE_CHECKING:
    from collections.abc import Mapping

_PYPROJECT = Path(__file__).resolve().parents[4] / "pyproject.toml"
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raitap-deps",
        description="Infer uv extras from a raitap Hydra config and run uv.",
    )
    parser.add_argument("--config-dir", type=Path, default=_DEFAULT_CONFIG_DIR)
    parser.add_argument("--config-name", type=str, default="config")
    parser.add_argument("--hardware", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    parser.add_argument("--mode", choices=("auto", "sync", "add"), default="auto")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command but do not run uv."
    )
    parser.add_argument(
        "overrides", nargs="*", help="Hydra overrides, forwarded to compose() verbatim."
    )
    return parser


def _compose_config(
    *, config_dir: Path, config_name: str, overrides: list[str]
) -> Mapping[str, Any]:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        composed = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
        return OmegaConf.to_container(composed, resolve=False)  # type: ignore[return-value]


def _print_frame(
    *,
    config_dir: Path,
    config_name: str,
    hardware_value: str,
    hardware_origin: str,
    mode: str,
    mode_origin: str,
    extras: list[str],
    pretty_command: str,
) -> None:
    width = 78
    bar = "─" * (width - 2)
    lines = [
        f"┌─ raitap-deps {bar[len('raitap-deps ') :]}",
        f"│ Config        {config_dir}/{config_name}.yaml",
        f"│ Hardware      {hardware_value} ({hardware_origin})",
        f"│ Install mode  {mode} ({mode_origin})",
        f"│ Extras        {', '.join(extras) if extras else '(none)'}",
        f"│ Command       {pretty_command}",
        f"└{bar}",
    ]
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.hardware == "auto":
        hardware = detect_hardware()
        hardware_origin = "probed"
    else:
        hardware = args.hardware
        hardware_origin = "forced"

    try:
        composed = _compose_config(
            config_dir=args.config_dir,
            config_name=args.config_name,
            overrides=list(args.overrides),
        )
    except Exception as exc:
        print(f"raitap-deps: failed to compose config: {exc}", file=sys.stderr)
        return 2

    try:
        extras, origins = infer_extras(composed, hardware=hardware)
    except Exception as exc:
        print(f"raitap-deps: inference failed: {exc}", file=sys.stderr)
        return 2

    try:
        validate_conflicts(extras, _PYPROJECT, origins=origins)
    except ExtrasConflictError as exc:
        print(f"raitap-deps: {exc}", file=sys.stderr)
        return 2

    mode = select_mode(args.mode)
    mode_origin = (
        "dev checkout"
        if (args.mode == "auto" and is_dev_install())
        else ("installed" if args.mode == "auto" else "forced")
    )

    argv_cmd, pretty = render_command(mode=mode, extras=extras)
    _print_frame(
        config_dir=args.config_dir,
        config_name=args.config_name,
        hardware_value=hardware,
        hardware_origin=hardware_origin,
        mode=mode,
        mode_origin=mode_origin,
        extras=sorted(extras),
        pretty_command=pretty,
    )

    if args.dry_run:
        print("[dry-run] not executing.")
        return 0

    print("[exec] running uv...")
    completed = subprocess.run(argv_cmd, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
