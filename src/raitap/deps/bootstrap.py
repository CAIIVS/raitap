"""Auto-install raitap extras inferred from the active Hydra config.

Invoked by :mod:`raitap.cli` before Hydra (or any torch-heavy code)
takes over. Walks the composed config, picks the right backend + adapter
extras for the current host, then dispatches per install context:

============  ==========  =======================================
``is_dev``    ``uv``      Action
============  ==========  =======================================
True          present     **Case A** — ``uv sync`` + ``uv run`` relaunch
True          missing     **Case B** — abort with "install uv" error
False         present     **Case C** — print ``uv add`` plan; only exec it when
                          ``--allow-project-edit`` was passed (otherwise abort)
False         missing     **Case D** — print ``pip install`` plan; auto-exec
                          when running inside a venv, require ``--exec-global``
                          when targeting the base interpreter
============  ==========  =======================================

Flags consumed (and removed from ``sys.argv`` before Hydra sees it):

- ``--dry-run`` — print the decision, do nothing.
- ``--sync-only`` — install, but skip the run step.
- ``--custom-deps`` — skip the whole flow.
- ``--allow-project-edit`` — Case C: consent to ``uv add`` modifying the
  caller's ``pyproject.toml``.
- ``--exec-global`` — Case D: consent to ``pip install`` against the base
  interpreter (not in a venv).

A sentinel env var prevents recursion when the re-exec'd process re-enters
this module.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from raitap.deps.availability import (
    ExtraUnavailableError,
    check_platform_availability,
)
from raitap.deps.command import render_command
from raitap.deps.conflicts import ExtrasConflictError, validate_conflicts
from raitap.deps.frame import print_deps_error_frame, print_deps_frame
from raitap.deps.inference import infer_extras
from raitap.deps.probe import detect_hardware
from raitap.deps.python_version import pick_python_version
from raitap.utils.diagnostics import is_dev_install

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

_SENTINEL = "_RAITAP_DEPS_BOOTSTRAPPED"
_PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_DEFAULT_CONFIG_NAME = "demo"

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
    """Split a multi-line exception message into ``[header, *bullets]``."""
    lines = msg.splitlines()
    header = lines[0].rstrip()
    bullets = [line.lstrip(" -•").rstrip() for line in lines[1:] if line.strip()]
    return [header, *bullets]


class _DepFlags:
    __slots__ = ("allow_project_edit", "custom", "dry_run", "exec_global", "sync_only")

    def __init__(self) -> None:
        self.dry_run = False
        self.sync_only = False
        self.custom = False
        self.allow_project_edit = False
        self.exec_global = False


def _strip_deps_flags(argv: list[str]) -> tuple[list[str], _DepFlags]:
    flags = _DepFlags()
    keep: list[str] = []
    for a in argv:
        if a == "--dry-run":
            flags.dry_run = True
        elif a == "--sync-only":
            flags.sync_only = True
        elif a == "--custom-deps":
            flags.custom = True
        elif a == "--allow-project-edit":
            flags.allow_project_edit = True
        elif a == "--exec-global":
            flags.exec_global = True
        else:
            keep.append(a)
    return keep, flags


def _hydra_overrides(argv: list[str]) -> list[str]:
    overrides: list[str] = []
    i = 0
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

    from raitap.configs import register_configs

    register_configs()
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


def _uv_available() -> bool:
    import shutil

    return shutil.which("uv") is not None


def _in_venv() -> bool:
    return sys.prefix != sys.base_prefix


def _is_dev_install() -> bool:
    return is_dev_install()


def _host_python() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _check_pin_matches_host(python_version: str | None) -> None:
    """Abort when the inferred Python pin cannot be satisfied by the host.

    Only relevant for non-uv paths — uv handles ``-p X.Y`` itself.
    """
    if python_version is None or python_version == _host_python():
        return
    print_deps_error_frame(
        label="Python mismatch",
        message=(
            f"This config needs Python {python_version}, but the running interpreter "
            f"is {_host_python()}."
        ),
        details=[
            "Switch to a Python " + python_version + " interpreter and rerun, or use a "
            "uv-managed dev checkout so the bootstrap can pin the version automatically."
        ],
    )
    sys.exit(2)


def _exec(argv: list[str], *, set_sentinel: bool = False) -> int:
    env = {**os.environ, _SENTINEL: "1"} if set_sentinel else None
    try:
        completed = subprocess.run(argv, check=False, env=env)
    except OSError as exc:
        print_deps_error_frame(
            label="Exec error",
            message=f"Failed to launch: {argv[0]!r}.",
            details=[str(exc)],
        )
        return 2
    return int(completed.returncode)


def _case_b_no_uv() -> None:
    print_deps_error_frame(
        label="Missing uv",
        message="Developer checkout requires uv but uv was not found on PATH.",
        details=[
            "Install uv from https://docs.astral.sh/uv/getting-started/installation/ "
            "and rerun. Or pass --custom-deps to bypass the auto-installer."
        ],
    )
    sys.exit(2)


def _case_c_uv_add(extras: set[str], python_version: str | None, flags: _DepFlags) -> int:
    _check_pin_matches_host(python_version)
    add_argv, _ = render_command(mode="add", extras=extras, python_version=None)
    return _exec(add_argv, set_sentinel=True)


def _case_d_pip(extras: set[str], python_version: str | None) -> int:
    _check_pin_matches_host(python_version)
    pip_argv, _ = render_command(mode="pip", extras=extras, python_version=None)
    return _exec(pip_argv, set_sentinel=True)


def _hint_invocation(cleaned: list[str], extra_flag: str) -> str:
    """Reconstruct the user's invocation with an extra flag appended."""
    tail = " ".join(cleaned[1:])
    base = "uv run raitap" if _uv_available() else "raitap"
    return f"{base} {tail} {extra_flag}".replace("  ", " ").strip()


def _refusal_note_blocks(case: str, extras: set[str], cleaned: list[str]) -> list:
    """Build the yellow/white note blocks for case C/D refusal."""
    from rich.style import Style
    from rich.text import Text

    from raitap.utils.colour import Status, colour

    warn = colour(Status.WARNING).base
    white = Style(color="white")

    if case == "C":
        cmd = f"uv add raitap[{','.join(sorted(extras))}]" if extras else "uv add raitap"
        hint = _hint_invocation(cleaned, "--allow-project-edit")
        return [
            Text("Running the uv add command would modify your project file. Either:", style=warn),
            Text.assemble(
                ("- Run it yourself: ", warn),
                (cmd, white),
            ),
            Text.assemble(
                ("- Add the --allow-project-edit flag: ", warn),
                (hint, white),
            ),
        ]
    # Case D
    pip_cmd = (
        f"{sys.executable} -m pip install raitap[{','.join(sorted(extras))}]"
        if extras
        else f"{sys.executable} -m pip install raitap"
    )
    hint = _hint_invocation(cleaned, "--exec-global")
    return [
        Text(
            "Running pip install would target the base interpreter (no venv detected). Either:",
            style=warn,
        ),
        Text.assemble(
            ("- Activate a venv and rerun.", warn),
        ),
        Text.assemble(
            ("- Run it yourself: ", warn),
            (pip_cmd, white),
        ),
        Text.assemble(
            ("- Add the --exec-global flag: ", warn),
            (hint, white),
        ),
    ]


def _is_consumer_project_cwd() -> bool:
    """Return ``True`` when cwd's pyproject.toml is for a non-raitap project.

    Bootstrap's auto-sync runs ``uv sync --extra ...`` against the *current*
    pyproject — that only makes sense when the current project IS raitap (dev
    checkout). For consumer projects that declare raitap as a dependency, the
    extras are namespaced under ``raitap[...]`` in their dependency string and
    the consumer's pyproject won't define those bare extra names, so the sync
    would fail. Detect this and skip the bootstrap flow.
    """
    cwd_pyproject = Path.cwd() / "pyproject.toml"
    if not cwd_pyproject.exists():
        return False
    try:
        import tomllib

        with cwd_pyproject.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        return False
    name = data.get("project", {}).get("name")
    return isinstance(name, str) and name != "raitap"


def maybe_bootstrap(argv: list[str]) -> list[str]:
    """Run the deps-bootstrap flow if appropriate; return cleaned argv."""
    cleaned, flags = _strip_deps_flags(argv)

    if os.environ.get(_SENTINEL) == "1" or flags.custom:
        return cleaned

    if _is_consumer_project_cwd():
        # Consumer projects declare raitap (and its extras) via their own
        # pyproject — auto-sync would target the wrong project. Treat as if
        # ``--custom-deps`` was passed.
        return cleaned

    _ensure_utf8_stdout()

    explicit_dir = _flag_value(cleaned, _CONFIG_DIR_FLAGS)
    explicit_name = _flag_value(cleaned, _CONFIG_NAME_FLAGS)
    # Mirror pipeline/__main__.py:_prepare_cli_argv — when the user picks a
    # custom ``--config-name`` but no ``--config-dir``, look in the cwd so
    # external consumer configs resolve without the user having to spell
    # ``--config-dir .`` every time.
    if explicit_dir is None and explicit_name not in (None, _DEFAULT_CONFIG_NAME):
        config_dir = Path.cwd()
    else:
        config_dir = Path(explicit_dir or _DEFAULT_CONFIG_DIR)
    config_name = explicit_name or _DEFAULT_CONFIG_NAME
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
    dev = _is_dev_install()
    uv_present = _uv_available()

    # Dispatch case → which install command to render in the preview frame.
    sync_argv: list[str] = []
    if dev and uv_present:
        case = "A"
        sync_argv, pretty = render_command(
            mode="sync", extras=extras, python_version=python_version
        )
    elif dev and not uv_present:
        # Case B: nothing to render; abort before frame.
        _case_b_no_uv()
        return cleaned  # unreachable, keeps type checkers happy
    elif not dev and uv_present:
        case = "C"
        _, pretty = render_command(mode="add", extras=extras, python_version=None)
    else:
        case = "D"
        _, pretty = render_command(mode="pip", extras=extras, python_version=None)

    if flags.dry_run:
        action = "Dry-run preview"
    elif flags.sync_only:
        action = "Sync only"
    else:
        action = "Sync then run"

    # Render frame with optional info note when the bootstrap will refuse to
    # exec (case C without --allow-project-edit, case D global w/o --exec-global).
    refusal = (case == "C" and not flags.allow_project_edit and not flags.dry_run) or (
        case == "D" and not _in_venv() and not flags.exec_global and not flags.dry_run
    )

    note_blocks = _refusal_note_blocks(case, extras, cleaned) if refusal else None

    print_deps_frame(
        hardware=hardware,
        hardware_origin="probed",
        python_version=python_version,
        extras=sorted(extras),
        pretty_command=pretty,
        action=action,
        note_blocks=note_blocks,
    )

    if flags.dry_run:
        sys.exit(0)
    if refusal:
        # Refusal path: frame already told the user what to do. Exit non-zero
        # so callers can detect "did not run".
        sys.exit(1)

    if case == "A":
        if flags.sync_only:
            sys.exit(_exec(sync_argv))
        # Sync first so the parent process never sees a stale lockfile, then
        # relaunch via ``uv run`` so the (possibly re-pinned) interpreter is
        # the one that imports torch and runs Hydra.
        rc = _exec(sync_argv)
        if rc != 0:
            sys.exit(rc)
        relaunch = ["uv", "run"]
        if python_version is not None:
            relaunch += ["-p", python_version]
        for x in sorted(extras):
            relaunch += ["--extra", x]
        relaunch += ["raitap", *cleaned[1:]]
        sys.exit(_exec(relaunch, set_sentinel=True))

    if case == "C":
        rc = _case_c_uv_add(extras, python_version, flags)
        if rc != 0 or flags.sync_only:
            sys.exit(rc)
        sys.exit(_exec([sys.executable, "-m", "raitap.cli", *cleaned[1:]], set_sentinel=True))

    # Case D
    rc = _case_d_pip(extras, python_version)
    if rc != 0 or flags.sync_only:
        sys.exit(rc)
    sys.exit(_exec([sys.executable, "-m", "raitap.cli", *cleaned[1:]], set_sentinel=True))
    return cleaned  # unreachable; satisfies static analysers that flag implicit return
