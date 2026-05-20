---
title: "Using automatic deps management"
description: "RAITAP infers and installs the dependencies each config needs. Covers the --dry-run preview, --allow-project-edit / --exec-global knobs, and the per-hardware index-routing gotchas."
myst:
  html_meta:
    "description": "RAITAP infers and installs the dependencies each config needs. Covers the --dry-run preview, --allow-project-edit / --exec-global knobs, and the per-hardware index-routing gotchas."
---

# Using automatic deps management

This page explains how to use RAITAP's automatic dependency management. It lists hardware gotchas and useful flags.

## Installing the deps

RAITAP will automatically analyse your config and install the deps when you run the job.

```{install-tabs}
:uv:
uv run raitap --config-dir my-configs --config-name assessment

:pip:
raitap --config-dir my-configs --config-name assessment
```

## Useful flags

Depending on your setup (`uv`, `pip`, ...) you might need to pass additional flags to make the install fully automatic.

| Flag                                                                                                       | When                                                             |
| ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| <a href="../flags.html#flag-dry-run"><code>--dry-run</code></a>                              | Preview the inferred install                                     |
| <a href="../flags.html#flag-allow-project-edit"><code>--allow-project-edit</code></a> / `-y` | Let RAITAP modify your `pyproject.toml` and run the `uv` install |
| <a href="../flags.html#flag-exec-global"><code>--exec-global</code></a>                      | Install into the global Pip environment (not recommended)        |

## Hardware gotchas

If you use a GPU, you will need to tweak your `pyproject.toml`. This is due to PyPI and ecosystem limitations.

:::{include} _gotchas.md
:::
