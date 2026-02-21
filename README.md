<img src="assets/images/tech_assessment_platform_logo.png" width="400">

## Purpose

The certainty pipeline `raitap` is dedicated to the certification of responsible AI.
Four dimensions can be certified following MLOps practices.

The four dimensions are:

* Transparency
* Robustness
* Security
* Autonomy and control

For more information read

* [Towards the certification of AI-based systems](https://doi.org/10.1109/SDS60720.2024.00020)
* [MLOps as enable of trustworthy AI](https://doi.org/10.1109/SDS60720.2024.00013)

Additional references on XAI aspects:

* [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/).
* [One Explanation Does Not Fit All](https://doi.org/10.48550/arXiv.1909.03012)

## Getting it running

### Prerequisites

* Python 3.13 or higher
* [uv](https://docs.astral.sh/uv/) package manager

### Running the project

1. Clone the repository
2. Install the dependencies by running

    ```bash
    uv sync
    ```

3. Run the example use case with `TODO`

For development setup and contribution guidelines, refer to [CONTRIBUTING.md](CONTRIBUTING.md).

## Design Decisions

### Key Characteristics

* Easily executable
* Easily maintainable
* Easily extendable

### Core technologies

**hydra** is a flexible application configuration framework. It provides a solid backbone including CLI integration,
basic logging, structured configuration, and plugins for job orchestration.

**uv** is a "blazingly" fast Python package and project manager, facilitating dependency installation, package building,
and ultimately development.

**mlflow** provides comprehensive support for AI-based workflows and MLOps core functionalities, such experiment
tracking, model versioning, and monitoring.

**ONNX Standard.** This standard is framework-independent, working well for tensorflow and pytorch. With few adjustments
to the code this standard should be attainable for every use case.

## Configuration (Hydra)

This projectt uses `hydra` for configuration management. The configuration files are located in the `configs` directory.
Configs are organized hierachically:

* `config.yaml` is the **main configuration file**, which includes the default values for all parameters and references to other config files.
* **config groups** (swappable components) are located in subdirectories of `configs` and contain specific configurations for different components of the project (e.g., models, data, transparency methods).
* **config overrides** can be specified at runtime to modify the behavior of the application without changing the config files.

### Where configs live

* Base config: `src/raitap/configs/config.yaml`
* Config groups:
  * `src/raitap/configs/model/` (e.g., `resnet50.yaml`, `vit_b32.yaml`)
  * `src/raitap/configs/data/` (e.g., `isic2018.yaml`, `malaria.yaml`)
  * `src/raitap/configs/transparency/` (e.g., `shap.yaml`, `captum.yaml`)

### Running with the default configuration

```bash
uv run python -m raitap.run
```

### Running with config overrides

```bash
uv run python -m raitap.run model=resnet50 data=isic2018 transparency=shap
```
