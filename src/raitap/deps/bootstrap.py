"""Auto-install raitap extras inferred from the active Hydra config.

Invoked by :mod:`raitap.cli` before Hydra (or any torch-heavy code)
takes over. Walks the composed config, picks the right backend + adapter
extras for the current host, then dispatches per install context
(:class:`BootstrapCase`):

==============================  ==========  ==========  ===========================
``BootstrapCase``               ``is_dev``  ``uv``      Action
==============================  ==========  ==========  ===========================
``DEV_WITH_UV``                 True        present     ``uv sync`` + ``uv run`` relaunch
``DEV_WITHOUT_UV``              True        missing     abort with "install uv" error
``USER_WITH_UV``                False       present     print ``uv add`` plan; only exec
                                                        when ``--allow-project-edit``
                                                        was passed (otherwise abort)
``USER_WITH_PIP``               False       missing     print ``pip install`` plan;
                                                        auto-exec when running inside a
                                                        venv, require ``--exec-global``
                                                        when targeting the base
                                                        interpreter
==============================  ==========  ==========  ===========================

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
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING


class BootstrapCase(Enum):
    """Install-context dispatch for the auto-deps flow.

    Picked by ``(_is_dev_install(), _uv_available())``; consumed by
    :func:`maybe_bootstrap` and :func:`install_raitap_deps` to route to
    the matching install command + refusal hint.
    """

    DEV_WITH_UV = "dev_with_uv"        # uv sync + uv run relaunch
    DEV_WITHOUT_UV = "dev_without_uv"  # abort with "install uv" error
    USER_WITH_UV = "user_with_uv"      # uv add (needs --allow-project-edit consent)
    USER_WITH_PIP = "user_with_pip"    # pip install (needs --exec-global outside a venv)

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
        elif a in ("--allow-project-edit", "-y"):
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


def _handle_dev_without_uv() -> None:
    """:attr:`BootstrapCase.DEV_WITHOUT_UV` — print error frame and abort."""
    print_deps_error_frame(
        label="Missing uv",
        message="Developer checkout requires uv but uv was not found on PATH.",
        details=[
            "Install uv from https://docs.astral.sh/uv/getting-started/installation/ "
            "and rerun. Or pass --custom-deps to bypass the auto-installer."
        ],
    )
    sys.exit(2)


def _run_uv_add(extras: set[str], python_version: str | None, flags: _DepFlags) -> int:
    """:attr:`BootstrapCase.USER_WITH_UV` — invoke ``uv add`` for the inferred extras."""
    _check_pin_matches_host(python_version)
    add_argv, _ = render_command(mode="add", extras=extras, python_version=None)
    return _exec(add_argv, set_sentinel=True)


def _run_pip_install(extras: set[str], python_version: str | None) -> int:
    """:attr:`BootstrapCase.USER_WITH_PIP` — invoke ``pip install`` for the inferred extras."""
    _check_pin_matches_host(python_version)
    pip_argv, _ = render_command(mode="pip", extras=extras, python_version=None)
    return _exec(pip_argv, set_sentinel=True)


def _hint_invocation(cleaned: list[str], extra_flag: str) -> str:
    """Reconstruct the user's invocation with an extra flag appended."""
    tail = " ".join(cleaned[1:])
    base = "uv run raitap" if _uv_available() else "raitap"
    return f"{base} {tail} {extra_flag}".replace("  ", " ").strip()


def _refusal_note_blocks(case: BootstrapCase, extras: set[str], cleaned: list[str]) -> list:
    """Build the yellow/white note blocks for refusal of CLI consent prompts."""
    from rich.style import Style
    from rich.text import Text

    from raitap.utils.colour import Status, colour

    warn = colour(Status.WARNING).base
    white = Style(color="white")

    match case:
        case BootstrapCase.USER_WITH_UV:
            cmd = (
                f"uv add raitap[{','.join(sorted(extras))}]" if extras else "uv add raitap"
            )
            hint = _hint_invocation(cleaned, "-y")
            return [
                Text(
                    "Running the uv add command would modify your project file. Either:",
                    style=warn,
                ),
                Text.assemble(("- Run it yourself: ", warn), (cmd, white)),
                Text.assemble(
                    ("- Add the -y flag (alias of --allow-project-edit): ", warn),
                    (hint, white),
                ),
            ]
        case BootstrapCase.USER_WITH_PIP:
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
                Text.assemble(("- Activate a venv and rerun.", warn)),
                Text.assemble(("- Run it yourself: ", warn), (pip_cmd, white)),
                Text.assemble(
                    ("- Add the --exec-global flag: ", warn),
                    (hint, white),
                ),
            ]
        case _:
            return []  # other cases never reach the refusal path


def _python_refusal_note_blocks(case: BootstrapCase, extras: set[str]) -> list:
    """Refusal frame body for :func:`install_raitap_deps` (Python entry point).

    Mirrors :func:`_refusal_note_blocks` but the hint references the Python
    function's kwarg, not a CLI flag.
    """
    from rich.style import Style
    from rich.text import Text

    from raitap.utils.colour import Status, colour

    warn = colour(Status.WARNING).base
    white = Style(color="white")

    match case:
        case BootstrapCase.USER_WITH_UV:
            cmd = (
                f"uv add raitap[{','.join(sorted(extras))}]" if extras else "uv add raitap"
            )
            return [
                Text(
                    "Running the uv add command would modify your project file. Either:",
                    style=warn,
                ),
                Text.assemble(("- Run it yourself: ", warn), (cmd, white)),
                Text.assemble(
                    ("- Call ", warn),
                    ("install_raitap_deps(cfg, allow_project_edit=True)", white),
                    (" to consent.", warn),
                ),
            ]
        case BootstrapCase.USER_WITH_PIP:
            pip_cmd = (
                f"{sys.executable} -m pip install raitap[{','.join(sorted(extras))}]"
                if extras
                else f"{sys.executable} -m pip install raitap"
            )
            return [
                Text(
                    "Running pip install would target the base interpreter (no venv detected). Either:",
                    style=warn,
                ),
                Text.assemble(("- Activate a venv and rerun.", warn)),
                Text.assemble(("- Run it yourself: ", warn), (pip_cmd, white)),
                Text.assemble(
                    ("- Call ", warn),
                    ("install_raitap_deps(cfg, exec_global=True)", white),
                    (" to consent.", warn),
                ),
            ]
        case _:
            return []  # other cases never reach the refusal path


def _config_to_mapping(config: Any) -> Mapping[str, Any]:
    """Coerce an ``AppConfig`` (or pre-built Mapping) into a plain dict the
    :mod:`raitap.deps.inference` walker can read.

    ``dataclasses.asdict`` recurses through nested dataclasses (the hydra-zen
    builders return dataclass instances) and preserves the ``_target_`` field
    the walker keys on.
    """
    import dataclasses
    from collections.abc import Mapping as _Mapping

    if isinstance(config, _Mapping):
        return config
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)
    raise TypeError(
        "install_raitap_deps expected an AppConfig (or Mapping), got "
        f"{type(config).__name__}."
    )


def install_raitap_deps(
    config: Any,
    *,
    allow_project_edit: bool = False,
    exec_global: bool = False,
) -> None:
    """Bootstrap raitap extras inferred from a Python ``AppConfig``.

    Programmatic analogue of the CLI auto-deps flow run by
    :func:`maybe_bootstrap`. Call this **before** importing
    :func:`raitap.run` (or any other raitap submodule that pulls adapter
    backends) — the function walks the config, picks the matching extras
    for the host, invokes ``uv``/``pip`` to install them, and re-execs
    the current script so the freshly-installed packages are visible.

    Idempotent: once a re-exec has happened the sentinel env var
    short-circuits subsequent calls, so the function is safe to invoke
    unconditionally at the top of a script.

    Args:
        config: An ``AppConfig`` instance (or pre-converted ``Mapping``).
            The walker reads ``_target_`` strings from ``transparency`` /
            ``robustness`` / ``metrics`` / ``reporting`` / ``tracking``
            plus ``model.source`` to pick a backend extra.
        allow_project_edit: Consent to ``uv add`` modifying the caller's
            ``pyproject.toml``. Without it, the function prints
            the planned command and exits non-zero.
        exec_global: Consent to ``pip install`` against the base
            interpreter when no venv is active

    Example::

        from raitap import AppConfig, Hardware
        from raitap.deps import install_raitap_deps

        cfg = AppConfig(hardware=Hardware.gpu, ...)
        install_raitap_deps(cfg, allow_project_edit=True)

        # Imports below pull adapter backends — safe now extras are pinned.
        from raitap import run
        run(cfg)
    """
    if os.environ.get(_SENTINEL) == "1":
        return

    _ensure_utf8_stdout()

    mapping = _config_to_mapping(config)

    hardware = detect_hardware()
    try:
        extras, origins = infer_extras(mapping, hardware=hardware)
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

    sync_argv: list[str] = []
    match (dev, uv_present):
        case (True, True):
            kind = BootstrapCase.DEV_WITH_UV
            sync_argv, pretty = render_command(
                mode="sync", extras=extras, python_version=python_version
            )
        case (True, False):
            _handle_dev_without_uv()
            return  # unreachable
        case (False, True):
            kind = BootstrapCase.USER_WITH_UV
            _, pretty = render_command(mode="add", extras=extras, python_version=None)
        case _:
            kind = BootstrapCase.USER_WITH_PIP
            _, pretty = render_command(mode="pip", extras=extras, python_version=None)

    refusal = (kind is BootstrapCase.USER_WITH_UV and not allow_project_edit) or (
        kind is BootstrapCase.USER_WITH_PIP and not _in_venv() and not exec_global
    )
    note_blocks = _python_refusal_note_blocks(kind, extras) if refusal else None

    print_deps_frame(
        hardware=hardware,
        hardware_origin="probed",
        python_version=python_version,
        extras=sorted(extras),
        pretty_command=pretty,
        action="Install then re-exec script",
        note_blocks=note_blocks,
    )

    if refusal:
        sys.exit(1)

    relaunch = [sys.executable, *sys.argv]
    flags = _DepFlags()
    flags.allow_project_edit = allow_project_edit
    flags.exec_global = exec_global

    match kind:
        case BootstrapCase.DEV_WITH_UV:
            rc = _exec(sync_argv)
            if rc != 0:
                sys.exit(rc)
            sys.exit(_exec(relaunch, set_sentinel=True))
        case BootstrapCase.USER_WITH_UV:
            rc = _run_uv_add(extras, python_version, flags)
            if rc != 0:
                sys.exit(rc)
            sys.exit(_exec(relaunch, set_sentinel=True))
        case BootstrapCase.USER_WITH_PIP:
            rc = _run_pip_install(extras, python_version)
            if rc != 0:
                sys.exit(rc)
            sys.exit(_exec(relaunch, set_sentinel=True))
        case BootstrapCase.DEV_WITHOUT_UV:  # unreachable — handled above
            return


def maybe_bootstrap(argv: list[str]) -> list[str]:
    """Run the deps-bootstrap flow if appropriate; return cleaned argv."""
    cleaned, flags = _strip_deps_flags(argv)

    if os.environ.get(_SENTINEL) == "1" or flags.custom:
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
    match (dev, uv_present):
        case (True, True):
            kind = BootstrapCase.DEV_WITH_UV
            sync_argv, pretty = render_command(
                mode="sync", extras=extras, python_version=python_version
            )
        case (True, False):
            # DEV_WITHOUT_UV: nothing to render; abort before frame.
            _handle_dev_without_uv()
            return cleaned  # unreachable, keeps type checkers happy
        case (False, True):
            kind = BootstrapCase.USER_WITH_UV
            _, pretty = render_command(mode="add", extras=extras, python_version=None)
        case _:
            kind = BootstrapCase.USER_WITH_PIP
            _, pretty = render_command(mode="pip", extras=extras, python_version=None)

    if flags.dry_run:
        action = "Dry-run preview"
    elif flags.sync_only:
        action = "Sync only"
    else:
        action = "Sync then run"

    # Render frame with optional info note when the bootstrap will refuse to
    # exec (USER_WITH_UV without --allow-project-edit, USER_WITH_PIP outside a
    # venv without --exec-global).
    refusal = (
        kind is BootstrapCase.USER_WITH_UV
        and not flags.allow_project_edit
        and not flags.dry_run
    ) or (
        kind is BootstrapCase.USER_WITH_PIP
        and not _in_venv()
        and not flags.exec_global
        and not flags.dry_run
    )

    note_blocks = _refusal_note_blocks(kind, extras, cleaned) if refusal else None

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

    match kind:
        case BootstrapCase.DEV_WITH_UV:
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
        case BootstrapCase.USER_WITH_UV:
            rc = _run_uv_add(extras, python_version, flags)
            if rc != 0 or flags.sync_only:
                sys.exit(rc)
            sys.exit(_exec([sys.executable, "-m", "raitap.cli", *cleaned[1:]], set_sentinel=True))
        case BootstrapCase.USER_WITH_PIP:
            rc = _run_pip_install(extras, python_version)
            if rc != 0 or flags.sync_only:
                sys.exit(rc)
            sys.exit(_exec([sys.executable, "-m", "raitap.cli", *cleaned[1:]], set_sentinel=True))
        case BootstrapCase.DEV_WITHOUT_UV:  # unreachable — handled above
            return cleaned
    return cleaned  # unreachable; satisfies static analysers that flag implicit return
