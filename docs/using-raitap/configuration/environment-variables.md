---
title: "Environment variables"
description: "RAITAP-specific environment variables. These mirror the safety-consent CLI flags and let non-CLI entry points (Python API, CI runners, schedulers) opt in without parsing argv."
myst:
  html_meta:
    "description": "RAITAP-specific environment variables. These mirror the safety-consent CLI flags and let non-CLI entry points (Python API, CI runners, schedulers) opt in without parsing argv."
---

(environment-variables)=

# Environment variables

RAITAP reads a small set of `RAITAP_*` environment variables. They all gate
behaviour that carries an arbitrary-code or data-correctness risk, so they
are off by default and must be opted into explicitly.

Each variable mirrors a CLI flag and a Python-API kwarg. Prefer the flag or
kwarg in interactive use; reach for the env var when invoking RAITAP from a
wrapper script, CI job, or scheduler that cannot easily mutate `sys.argv`.

The accepted truthy value is the literal string `"1"`. Any other value
(including `"true"`, `"yes"`, an empty string, or the variable being unset)
is treated as no consent.

## `RAITAP_ALLOW_PREPROCESSING_EXEC`

Consents to executing a user-supplied preprocessing factory loaded from a
`.py` file pointed at by `data.preprocessing` or
`data.model_input_transformation`. Without this consent, RAITAP refuses to
import the file because doing so runs arbitrary Python at module load.

| Surface | Equivalent |
|---|---|
| CLI | `--allow-preprocessing-exec` / `-yp` |
| Python API | `acknowledge_preprocessing_exec=True` on `raitap.run(...)` |
| Env var | `RAITAP_ALLOW_PREPROCESSING_EXEC=1` |

See {doc}`../../modules/data/preprocessing` for the underlying mechanism.

## `RAITAP_ACKNOWLEDGE_PREPROCESSING_OFF`

Acknowledges that the run will proceed with no preprocessing on a code path
where RAITAP would otherwise refuse to start (typical for pretrained image
models that expect ImageNet normalisation). Use this when you genuinely want
raw inputs to reach the model.

| Surface | Equivalent |
|---|---|
| CLI | `--acknowledge-preprocessing-off` |
| Python API | `acknowledge_preprocessing_off=True` on `raitap.run(...)` |
| Env var | `RAITAP_ACKNOWLEDGE_PREPROCESSING_OFF=1` |

See {doc}`../../modules/data/preprocessing` for the conditions that trigger
the refusal.

## `RAITAP_ALLOW_UNSAFE_PICKLE`

Consents to loading legacy PyTorch checkpoints with
`torch.load(..., weights_only=False)`. The default `weights_only=True` path
rejects pickled Python objects; opting out re-enables arbitrary-code
execution at load time, so only set this for checkpoints you trust.

| Surface | Equivalent |
|---|---|
| CLI | `--allow-unsafe-pickle` |
| Python API | `allow_unsafe_pickle=True` on `raitap.run(...)` |
| Env var | `RAITAP_ALLOW_UNSAFE_PICKLE=1` |

See {doc}`../../modules/model/own-vs-built-in` for when this is necessary.

## Examples

POSIX shell:

```bash
export RAITAP_ALLOW_PREPROCESSING_EXEC=1
export RAITAP_ALLOW_UNSAFE_PICKLE=1
uv run raitap --config-name assessment
```

PowerShell:

```powershell
$env:RAITAP_ALLOW_PREPROCESSING_EXEC = "1"
$env:RAITAP_ALLOW_UNSAFE_PICKLE = "1"
uv run raitap --config-name assessment
```

GitHub Actions:

```yaml
- run: uv run raitap --config-name assessment
  env:
    RAITAP_ALLOW_PREPROCESSING_EXEC: "1"
    RAITAP_ALLOW_UNSAFE_PICKLE: "1"
```
