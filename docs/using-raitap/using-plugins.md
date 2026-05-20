---
title: "Using plugins"
description: "Install third-party RAITAP adapters and use them like built-in ones."
myst:
  html_meta:
    "description": "Install third-party RAITAP adapters and use them like built-in ones."
---

# Using plugins

This page explains how to seamlessly use a 3rd party lib's plugin in RAITAP.

Some libraries are implemented as 1st party in RIAITAP directly, but some others simply ship a plugin. The user-facing behaviour is identical.

:::{tip}
Are you a library maintainer wanting to ship your own adapter as a plugin? See
{doc}`../contributor/writing-a-plugin`.
:::

## 1. Install

Install the plugin next to RAITAP:

```{install-tabs}
:uv:
uv add raitap raitap-superxai

:pip:
pip install raitap raitap-superxai
```

That's it. Every installed plugin is discovered automatically.

## 2. Use it

Reference the adapter by its `registry_name` (here, `superxai`), exactly like a
built-in adapter.

```{config-tabs}
:yaml:
transparency:
  my_run:
    _target_: "SuperXAIExplainer"
    algorithm: supertreeshap

:python:
from raitap.transparency import superxai

transparency = {"my_run": superxai(algorithm="supertreeshap")}
```

The `+transparency=superxai` CLI shorthand also works **if** the plugin ships a
matching preset; otherwise compose it inline as above.

## If a plugin doesn't show up

- **Version mismatch.** A plugin declares which RAITAP versions it supports. If
  your installed RAITAP is outside that range (or the plugin declares no range),
  RAITAP **skips it and logs a warning** — check your run's logs and upgrade the
  plugin or RAITAP as needed.
- **A plugin crashed on load.** A broken plugin is logged (by name) and skipped;
  it never breaks your run. The warning names the culprit.
- **Disable all plugins.** Set `RAITAP_DISABLE_PLUGINS=1` to ignore every
  installed plugin — useful for reproducing an issue without third-party code.
