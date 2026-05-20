---
title: "Using plugins"
description: "Install third-party RAITAP adapters and use them like built-in ones."
myst:
  html_meta:
    "description": "Install third-party RAITAP adapters and use them like built-in ones."
---

# Using plugins

Third-party packages can add adapters (extra explainers, attacks, trackers, …)
to RAITAP. Once installed alongside RAITAP, a plugin's adapter behaves like a
first-party one — same YAML and Python API.

:::{tip}
Are you a library maintainer wanting to ship your own adapter as a plugin? See
{doc}`../contributor/writing-a-plugin`.
:::

## Install

Install the plugin next to RAITAP:

```bash
pip install raitap raitap-superxai
```

That's it — no config flag to "enable" it. Every installed plugin is discovered
automatically.

## Use it

Reference the adapter by its `registry_name` (here, `superxai`), exactly like a
built-in adapter.

In YAML:

```yaml
transparency:
  my_run:
    _target_: SuperXAIExplainer
    algorithm: supertreeshap
```

In Python:

```python
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
