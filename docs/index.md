# RAITAP

RAITAP is a Python library to assess the responsibility level of AI models. It is designed to be easily integrated into existing MLOps workflows.

## What does it assess?

RAITAP currently assesses the following 2 responsible AI dimensions:

- Transparency
- Robustness

as defined in [Towards the certification of AI-based systems](https://doi.org/10.1109/SDS60720.2024.00020) and [MLOps as enabler of trustworthy AI](https://doi.org/10.1109/SDS60720.2024.00013)

## Where does it fit in my workflow?

RAITAP is configured via YAML [Hydra](https://hydra.cc/) configs or CLI flags, and then ran via a CLI command.

This means it can be used either as:

- a standalone Python package, which stores the assessment outputs in the directory you specify. See {doc}`./using-raitap/understanding-outputs` for more details.
- a step in a larger MLOps pipeline, which forwards the assessment outputs to your tracking software (e.g. MLflow). See {doc}`the tracking module <./modules/tracking/index>` for more details.

This gives you full flexibility to choose how you want to use RAITAP in your workflow.

## How is it structured?

RAITAP is a wrapper around existing XAI frameworks, which provides a consistent API, allowing you to easily switch your configuration, combine frameworks, and obtain consolidated outputs.

## Table of contents

```{toctree}
:maxdepth: 1
:caption: Using RAITAP

using-raitap/installation
using-raitap/get-it-running
using-raitap/configuration/index
using-raitap/understanding-outputs
using-raitap/job-launcher
```

```{toctree}
:maxdepth: 1
:caption: Modules

modules/model/index
modules/data/index
modules/transparency/index
modules/metrics/index
modules/tracking/index
modules/reporting/index
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
