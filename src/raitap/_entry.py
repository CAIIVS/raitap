"""Thin CLI entry that runs deps-bootstrap before importing the torch-heavy pipeline.

Pyproject's ``raitap`` console script points here so the *first* line of code
that runs in a fresh checkout is the dep inference. Importing
:mod:`raitap.run` directly would pull ``torch`` (via ``forward_output``)
before the bootstrap can sync it — defeating the auto-install flow.

Order of operations:

1. Handle non-Hydra subcommands (``raitap tracking stop``) — these do not
   need the heavy deps.
2. :func:`raitap.configs.extras.bootstrap.maybe_bootstrap` — re-exec via
   ``uv run`` when needed, exit otherwise.
3. Only after step 2 sets the sentinel (or the user passed
   ``--custom-deps``), import :mod:`raitap.run.__main__` and dispatch.
"""

from __future__ import annotations

import sys


def main() -> None:
    if sys.argv[1:2] == ["tracking", "stop"]:
        import logging

        from raitap.tracking import run_stop_command
        from raitap.utils.console import setup_logging

        setup_logging(level=logging.INFO)
        run_stop_command()
        return

    from raitap.configs.extras.bootstrap import maybe_bootstrap

    sys.argv = maybe_bootstrap(list(sys.argv))

    # Bootstrap either re-exec'd (and called ``sys.exit``) or set the sentinel
    # so the heavy run package is now safe to import.
    from raitap.run.__main__ import main as run_main

    run_main()


if __name__ == "__main__":
    main()
