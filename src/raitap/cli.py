"""CLI entry: subcommand dispatch + ``--demo`` shortcut + deps bootstrap.

Pyproject's ``raitap`` console script points here so the *first* line of code
that runs in a fresh checkout is the dep inference. Importing
:mod:`raitap.pipeline` directly would pull ``torch`` (via ``forward_output``)
before the bootstrap can sync it — defeating the auto-install flow.

Order of operations:

1. Handle non-Hydra subcommands (``raitap tracking stop``) — these do not
   need the heavy deps.
2. Resolve ``--demo`` to the bundled :mod:`raitap.configs.demo` config.
3. :func:`raitap.deps.bootstrap.maybe_bootstrap` — re-exec via ``uv run``
   when needed, exit otherwise.
4. Only after step 3 sets the sentinel (or the user passed ``--custom-deps``),
   import :mod:`raitap.pipeline.__main__` and dispatch.
"""

from __future__ import annotations

import sys

_DEMO_FLAG = "--demo"
_CONFIG_FLAGS = (
    "--config-name",
    "-cn",
    "--config-dir",
    "-cd",
    "--config-path",
    "-cp",
)
_HYDRA_INTROSPECTION_FLAGS = (
    "--help",
    "-h",
    "--hydra-help",
    "--cfg",
    "--info",
    "--version",
)


def _handle_demo_flag(argv: list[str]) -> list[str]:
    """Rewrite ``--demo`` to point Hydra at the bundled ``demo`` config.

    The bundled ``raitap.configs`` directory is registered as the package
    config_path via :func:`hydra.main` in :mod:`raitap.pipeline.__main__`, so
    just emitting ``--config-name demo`` is enough — no ``--config-dir`` needed.
    """
    if _DEMO_FLAG not in argv:
        return argv
    stripped = [arg for arg in argv if arg != _DEMO_FLAG]
    if not stripped:
        return argv
    return [stripped[0], "--config-name", "demo", *stripped[1:]]


def _needs_help_frame(argv: list[str]) -> bool:
    """``True`` when ``raitap`` was invoked with no actionable config selector.

    User passed neither ``--demo`` nor any ``--config-name``/``--config-dir``
    flag, and didn't request Hydra's own help. In that case we print a
    pointer instead of silently failing in Hydra's stack.
    """
    if not argv:
        return True
    if any(arg in _HYDRA_INTROSPECTION_FLAGS for arg in argv):
        return False
    if _DEMO_FLAG in argv:
        return False
    for arg in argv:
        for flag in _CONFIG_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                return False
    return True


def _print_help_frame() -> None:
    """Print a usage frame in the standard raitap panel style."""
    from rich.panel import Panel
    from rich.style import Style
    from rich.table import Table
    from rich.text import Text

    from raitap.utils.console import Status, colour, get_console, setup_logging

    setup_logging()

    docs_url = "https://caiivs.github.io/raitap/"
    info_style = colour(Status.INFO).base

    from raitap.utils.diagnostics import is_dev_install

    invocation = "uv run raitap" if is_dev_install() else "raitap"

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column(no_wrap=False)
    table.add_row("demo", Text(f"{invocation} --demo", style="white"))
    table.add_row(
        "custom",
        Text(f"{invocation} --config-dir <dir> --config-name <name>", style="white"),
    )
    table.add_row("docs", Text(docs_url, style=info_style + Style(link=docs_url)))

    intro = Text(
        "Either run the demo, or specify your config directory and name.",
        style="dim",
    )

    from rich.console import Group

    title = Text.assemble(
        ("RAITAP", info_style + Style(bold=True)),
        (" · CLI", info_style),
    )
    panel = Panel(
        Group(intro, Text(""), table),
        title=title,
        title_align="left",
        border_style=info_style,
        padding=(1, 2),
    )
    console = get_console()
    console.print()
    console.print(panel)
    console.print()


def main() -> None:
    if sys.argv[1:2] == ["tracking", "stop"]:
        import logging

        from raitap.tracking import run_stop_command
        from raitap.utils.console import setup_logging

        setup_logging(level=logging.INFO)
        run_stop_command()
        return

    if _needs_help_frame(sys.argv[1:]):
        _print_help_frame()
        return

    sys.argv = _handle_demo_flag(list(sys.argv))

    from raitap.deps.bootstrap import maybe_bootstrap

    sys.argv = maybe_bootstrap(list(sys.argv))

    # Bootstrap either re-exec'd (and called ``sys.exit``) or set the sentinel
    # so the heavy pipeline package is now safe to import.
    from raitap.pipeline.__main__ import main as run_main

    run_main()


if __name__ == "__main__":
    main()
