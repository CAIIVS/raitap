"""Auto-install raitap extras inferred from the active Hydra config.

Invoked by :mod:`raitap.run.__main__` before Hydra takes over. Walks the
composed config, picks the right backend + adapter extras for the current
host, then re-launches ``raitap`` through ``uv run`` so the freshly synced
venv (and possibly a different interpreter, see
:mod:`raitap.configs.extras.python_version`) is the one that actually runs
the pipeline.

Flags consumed (and removed from ``sys.argv`` before Hydra sees it):

- ``--dry-run`` — print the decision, do not sync, do not run.
- ``--sync-only`` — sync the venv with the inferred extras, then exit
  without running the pipeline.
- ``--custom-deps`` — skip the whole flow; assume the user manages extras
  manually.

A sentinel env var prevents recursion when the re-exec'd process re-enters
this module.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from raitap.configs.extras.availability import (
    ExtraUnavailableError,
    check_platform_availability,
)
from raitap.configs.extras.command import render_command
from raitap.configs.extras.conflicts import ExtrasConflictError, validate_conflicts
from raitap.configs.extras.frame import print_deps_error_frame, print_deps_frame
from raitap.configs.extras.inference import infer_extras
from raitap.configs.extras.probe import detect_hardware
from raitap.configs.extras.python_version import pick_python_version
from raitap.utils.diagnostics import is_dev_install

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

_SENTINEL = "_RAITAP_DEPS_BOOTSTRAPPED"
_PYPROJECT = Path(__file__).resolve().parents[4] / "pyproject.toml"
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_DEFAULT_CONFIG_NAME = "config"

_CONFIG_DIR_FLAGS = ("--config-dir", "-cd", "--config-path", "-cp")
_CONFIG_NAME_FLAGS = ("--config-name", "-cn")


def _flag_value(argv: list[str], flags: tuple[str, ...]) -> str | None:
    for i, a in enumerate(argv):
        for flag in flags:
            if a == flag and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(flag + "="):
                return a[len(flag) + 1 :]
    return None


def _split_error_message(msg: str) -> list[str]:
    """Split a multi-line exception message into ``[header, *bullets]``.

    The conflicts/availability errors emit ``"<sentence>\\n  - <bullet>\\n  - ..."``
    so we strip the leading marker and return each part as plain text for the
    error frame to render.
    """
    lines = msg.splitlines()
    header = lines[0].rstrip()
    bullets = [line.lstrip(" -•").rstrip() for line in lines[1:] if line.strip()]
    return [header, *bullets]


def _strip_deps_flags(argv: list[str]) -> tuple[list[str], bool, bool, bool]:
    """Return ``(cleaned_argv, dry_run, sync_only, custom_deps)``."""
    keep: list[str] = []
    dry_run = False
    sync_only = False
    custom = False
    for a in argv:
        if a == "--dry-run":
            dry_run = True
        elif a == "--sync-only":
            sync_only = True
        elif a == "--custom-deps":
            custom = True
        else:
            keep.append(a)
    return keep, dry_run, sync_only, custom


def _hydra_overrides(argv: list[str]) -> list[str]:
    """Return positional Hydra-style overrides (e.g. ``data=mnist_samples``).

    Skips the program name and any ``--flag`` / ``--flag=value`` pairs.
    """
    overrides: list[str] = []
    i = 0
    # argv[0] is the program name when called as ``main(sys.argv)``.
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") or a.startswith("-"):
            two_token = (
                "=" not in a
                and i + 1 < len(argv)
                and not argv[i + 1].startswith("-")
                and any(a == f for f in (*_CONFIG_DIR_FLAGS, *_CONFIG_NAME_FLAGS))
            )
            i += 2 if two_token else 1
            continue
        overrides.append(a)
        i += 1
    return overrides


def _compose(config_dir: Path, config_name: str, overrides: list[str]) -> Mapping[str, Any]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        composed = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
        return OmegaConf.to_container(composed, resolve=False)  # type: ignore[return-value]


def _ensure_utf8_stdout() -> None:
    import contextlib

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8")


def maybe_bootstrap(argv: list[str]) -> list[str]:
    """Run the deps-bootstrap flow if appropriate; return cleaned argv.

    The cleaned argv has ``--dry-run`` / ``--custom-deps`` removed and is the
    value the caller should feed back into ``sys.argv`` before invoking the
    Hydra entry point.
    """
    cleaned, dry_run, sync_only, custom = _strip_deps_flags(argv)

    if os.environ.get(_SENTINEL) == "1" or custom:
        return cleaned

    # Bootstrap only manages deps in a uv-driven developer checkout. Wheel /
    # pip installs (no local pyproject, no uv) keep the user's chosen extras
    # untouched.
    import shutil

    if not is_dev_install() or shutil.which("uv") is None or not _PYPROJECT.exists():
        return cleaned

    _ensure_utf8_stdout()

    # Locate config dir/name from cleaned argv; otherwise fall back to defaults.
    config_dir = Path(_flag_value(cleaned, _CONFIG_DIR_FLAGS) or _DEFAULT_CONFIG_DIR)
    config_name = _flag_value(cleaned, _CONFIG_NAME_FLAGS) or _DEFAULT_CONFIG_NAME
    overrides = _hydra_overrides(cleaned[1:])

    try:
        composed = _compose(config_dir, config_name, overrides)
    except Exception as exc:
        print_deps_error_frame(
            label="Config error",
            message="Could not compose Hydra config.",
            details=[str(exc)],
        )
        sys.exit(2)

    hardware = detect_hardware()
    try:
        extras, origins = infer_extras(composed, hardware=hardware)
        validate_conflicts(extras, _PYPROJECT, origins=origins)
        check_platform_availability(_PYPROJECT, extras)
    except (ExtrasConflictError, ExtraUnavailableError) as exc:
        header, *bullets = _split_error_message(str(exc))
        label = (
            "Platform mismatch" if isinstance(exc, ExtraUnavailableError) else "Conflicting extras"
        )
        print_deps_error_frame(label=label, message=header, details=bullets)
        sys.exit(2)
    except Exception as exc:
        print_deps_error_frame(
            label="Inference error",
            message="Deps inference failed.",
            details=[str(exc)],
        )
        sys.exit(2)

    python_version = pick_python_version(_PYPROJECT, extras)
    sync_argv, sync_pretty = render_command(
        mode="sync", extras=extras, python_version=python_version
    )

    if dry_run:
        action = "Dry-run preview"
    elif sync_only:
        action = "Sync only"
    else:
        action = "Sync then run"

    print_deps_frame(
        hardware=hardware,
        hardware_origin="probed",
        python_version=python_version,
        extras=sorted(extras),
        pretty_command=sync_pretty,
        action=action,
    )

    if dry_run:
        sys.exit(0)

    if sync_only:
        # Sync the venv with the inferred extras and exit; do not run the
        # pipeline. Useful for prepping a venv ahead of time.
        try:
            completed = subprocess.run(sync_argv, check=False)
        except OSError as exc:
            print(f"raitap: failed to launch uv ({exc}). Is uv on PATH?", file=sys.stderr)
            sys.exit(2)
        sys.exit(int(completed.returncode))

    # Re-launch through ``uv run`` so the right interpreter / extras execute.
    relaunch = ["uv", "run"]
    if python_version is not None:
        relaunch += ["-p", python_version]
    for x in sorted(extras):
        relaunch += ["--extra", x]
    relaunch += ["raitap", *cleaned[1:]]
    env = {**os.environ, _SENTINEL: "1"}
    try:
        completed = subprocess.run(relaunch, check=False, env=env)
    except OSError as exc:
        print(f"raitap: failed to relaunch via uv ({exc}). Is uv on PATH?", file=sys.stderr)
        sys.exit(2)
    sys.exit(int(completed.returncode))
