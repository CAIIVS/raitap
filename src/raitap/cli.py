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

from raitap import _cli_argv


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


def _print_version() -> None:
    """Print the installed raitap version. Short-circuits before deps bootstrap."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        print(f"raitap {version('raitap')}")
    except PackageNotFoundError:
        print("raitap (version unknown — not installed as a distribution)")


def main() -> None:
    if sys.argv[1:2] == ["tracking", "stop"]:
        import logging

        from raitap.tracking import run_stop_command
        from raitap.utils.console import setup_logging

        setup_logging(level=logging.INFO)
        run_stop_command()
        return

    if sys.argv[1:2] in (["--version"], ["-V"]):
        _print_version()
        return

    if _cli_argv.needs_help_frame(sys.argv[1:]):
        _print_help_frame()
        return

    sys.argv = _cli_argv.rewrite_demo(list(sys.argv))

    from raitap.deps.bootstrap import maybe_bootstrap

    sys.argv = maybe_bootstrap(list(sys.argv))

    # Bootstrap either re-exec'd (and called ``sys.exit``) or set the sentinel
    # so the heavy pipeline package is now safe to import.
    from raitap.pipeline.__main__ import main as run_main

    run_main()


if __name__ == "__main__":
    main()
