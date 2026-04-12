# RAITAP

RAITAP is a Python library to assess the responsibility level of AI models. It is designed to be easily integrated into existing MLOps workflows.

## What does RAITAP assess?

RAITAP currently assesses the following 2 responsible AI dimensions:

- Transparency
- Robustness

as defined in [Towards the certification of AI-based systems](https://doi.org/10.1109/SDS60720.2024.00020) and [MLOps as enabler of trustworthy AI](https://doi.org/10.1109/SDS60720.2024.00013)

## How is RAITAP structured?

RAITAP is a wrapper around existing XAI frameworks, which provides a consistent API, allowing you to easily switch your configuration, combine frameworks, and obtain consolidated outputs.

## Table of contents

```{toctree}
:maxdepth: 1
:caption: Using RAITAP

using-raitap/standalone-integrated
using-raitap/installation
using-raitap/get-it-running
using-raitap/configuration/index
using-raitap/understanding-outputs
```

```{toctree}
:maxdepth: 1
:caption: Modules

modules/model/index
modules/data/index
modules/transparency/index
modules/metrics/index
modules/tracking/index
```

```{toctree}
:maxdepth: 1
:caption: Reference

reference/api/index
license
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Contributor documentation

contributor/index
```
