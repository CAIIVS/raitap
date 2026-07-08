# raitap-example-plugin

A minimal, runnable RAITAP plugin. Copy this directory as a starting point for
your own plugin, or install it as-is to see plugin discovery work end to end.

It registers one adapter, `identity_attack` — a robustness assessor that
returns the inputs unmodified. No real attack logic, no third-party
dependency beyond `raitap` itself; it exists to demonstrate the wiring, not
to test robustness.

See `docs/contributor/writing-a-plugin.md` in the raitap repo for the full
guide this plugin follows.

## Install

Alongside a raitap project:

```bash
uv add raitap raitap-example-plugin
# or
pip install raitap raitap-example-plugin
```

From this checkout (editable, for trying it locally):

```bash
uv pip install -e ./example-plugin
```

## It's discovered automatically

No registration step needed. RAITAP scans the `raitap.adapters` entry-point
group at config-load time and imports every installed plugin, which fires the
`@adapters.robustness(...)` decorator in `raitap_example_plugin/__init__.py`.
Confirm it resolved:

```bash
python -c "from raitap.robustness import identity_attack; print(identity_attack)"
```

## Use it in a config

YAML (`assessment.yaml`, see the fragment in this directory):

```yaml
robustness:
  identity_check:
    use: identity_attack
    algorithm: identity
```

Python:

```python
from raitap import AppConfig
from raitap.robustness import identity_attack

cfg = AppConfig(
    ...,
    robustness={
        "identity_check": identity_attack(algorithm="identity"),
    },
)
```

Both forms resolve `use: identity_attack` against the same trusted registry
a first-party adapter (e.g. `torchattacks`) resolves against — a plugin
adapter is indistinguishable from an in-tree one once installed.

## Uninstall

```bash
uv pip uninstall raitap-example-plugin
```
