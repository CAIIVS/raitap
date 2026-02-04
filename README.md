<img src="assets/images/tech_assessment_platform_logo.png" width="800">

## Purpose
The certainty pipeline `raitap` is dedicated to the certification of responsible AI.
Four dimensions can be certified following MLOps practices.

The four dimensions are:
* Transparency
* Reliability
* Safety and security
* Autonomy and control

For more information read
- [Towards the certification of AI-based systems](https://doi.org/10.1109/SDS60720.2024.00020)
- [MLOps as enable of trustworthy AI](https://doi.org/10.1109/SDS60720.2024.00013)

Additional references on XAI aspects:
- [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/).
- [One Explanation Does Not Fit All](https://doi.org/10.48550/arXiv.1909.03012)


## Design Decisions

### Key Characteristics
* Easily executable
* Easily maintainable
* Easily extendable

### Core technologies
**hydra** is a flexible application configuration framework. It provides a solid backbone including CLI integration, basic logging, structured configuration, and plugins for job orchestration.

**uv** is a "blazingly" fast Python package and project manager, facilitating dependency installation, package building, and ultimately development.

**mlflow** provides comprehensive support for AI-based workflows and MLOps core functionalities, such experiment tracking, model versioning, and monitoring.

**ONNX Standard.** This standard is framework-independent, working well for tensorflow and pytorch. With few adjustments to the code this standard should be attainable for every use case.
