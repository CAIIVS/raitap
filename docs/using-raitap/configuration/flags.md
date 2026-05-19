---
title: "Flags"
description: "Reference table mapping each RAITAP CLI flag to its Python-API kwarg and (where applicable) environment variable."
myst:
  html_meta:
    "description": "Reference table mapping each RAITAP CLI flag to its Python-API kwarg and (where applicable) environment variable."
---

(flags)=

# Flags

Every RAITAP-specific invocation knob is listed below. Each row gives the
equivalent surface across the CLI, the Python API (`raitap.run(...)`
kwargs), and the environment. The accepted truthy value for every env var
is the literal string `"1"`; any other value (including unset) is no
consent.

Hydra's own flags (`--config-name`, `--config-dir`, `--multirun`, `--help`,
`--hydra-help`) are documented by Hydra and are not repeated here.

## General CLI flags

| CLI | Python API | Env var | Description |
|---|---|---|---|
| `--demo`{#flag-demo} | — (pass `AppConfig` directly) | — | Run the bundled `demo` config. Equivalent to `--config-name demo` against the packaged `raitap.configs` directory. |
| `--version` / `-V`{#flag-version} | `raitap.__version__` | — | Print the installed RAITAP version and exit. Short-circuits before deps inference. |

## Dependency-management flags

| CLI | Python API | Env var | Description |
|---|---|---|---|
| `--dry-run`{#flag-dry-run} | — | — | Print the inferred extras for the composed config and exit without installing or running. |
| `--sync-only`{#flag-sync-only} | — | — | Install the inferred extras and exit before the pipeline runs. |
| `--custom-deps`{#flag-custom-deps} | `auto_install_deps=False` (default) | — | Skip dep inference entirely; caller is responsible for installing the right extras. |
| `--allow-project-edit`{#flag-allow-project-edit} / `-y` | `auto_install_deps=True` | — | Consent to `uv add` mutating `pyproject.toml` / `uv.lock` when extras are missing. |
| `--exec-global`{#flag-exec-global} | `exec_global=True` | — | Consent to bare-`pip install` into the active interpreter when no venv is detected. Not recommended. |
| — | — | `_RAITAP_DEPS_BOOTSTRAPPED=1`{#env-raitap-deps-bootstrapped} | Short-circuit the entire deps-bootstrap flow (both CLI and `auto_install_deps=True`). Set this when a downstream `pyproject.toml` already owns the extras (see {ref}`consuming-raitap`). The bootstrap sets this itself after a re-exec to prevent recursion; you can also set it manually. |

## Safety-consent flags

| CLI | Python API | Env var | Description |
|---|---|---|---|
| `--allow-preprocessing-exec`{#flag-allow-preprocessing-exec} / `-yp` | `acknowledge_preprocessing_exec=True` | `RAITAP_ALLOW_PREPROCESSING_EXEC=1` | Consent to importing a user-supplied preprocessing `.py` (runs arbitrary code at load time). |
| `--acknowledge-preprocessing-off`{#flag-acknowledge-preprocessing-off} | `acknowledge_preprocessing_off=True` | `RAITAP_ACKNOWLEDGE_PREPROCESSING_OFF=1` | Acknowledge running with both `data.preprocessing` and `data.model_input_transformation` set to `null`. |
| `--allow-unsafe-pickle`{#flag-allow-unsafe-pickle} | `allow_unsafe_pickle=True` | `RAITAP_ALLOW_UNSAFE_PICKLE=1` | Consent to loading legacy checkpoints with `torch.load(..., weights_only=False)` (runs arbitrary code at load time). |

## Bootstrap dispatch cases

The deps-bootstrap (CLI + `raitap.run(..., auto_install_deps=True)`) picks
one of four code paths from `(is_dev_install, uv_available)`. Each path
has its own install command and consent requirement.

| `BootstrapCase` | Dev install? | `uv` present? | Action | Consent flag |
|---|---|---|---|---|
| `DEV_WITH_UV` | yes | yes | `uv sync` + `uv run` relaunch. | — |
| `DEV_WITHOUT_UV` | yes | no | Abort with an "install uv" error frame. | — |
| `USER_WITH_UV` | no | yes | Print the planned `uv add` command; only execute when consent is given. | [`--allow-project-edit`](#flag-allow-project-edit) |
| `USER_WITH_PIP` | no | no | Print the planned `pip install` command; auto-execute inside a venv, require consent against the base interpreter. | [`--exec-global`](#flag-exec-global) (only when no venv is active) |

"Dev install" means an editable checkout of the RAITAP repo
(`pyproject.toml` next to the source tree). "User install" means a wheel
from PyPI / an internal index.

## Setting env vars

POSIX shell:

```bash
export RAITAP_ALLOW_PREPROCESSING_EXEC=1
uv run raitap --config-name assessment
```

PowerShell:

```powershell
$env:RAITAP_ALLOW_PREPROCESSING_EXEC = "1"
uv run raitap --config-name assessment
```

GitHub Actions:

```yaml
- run: uv run raitap --config-name assessment
  env:
    RAITAP_ALLOW_PREPROCESSING_EXEC: "1"
```
