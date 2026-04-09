# Where does RAITAP fit in my workflow?

RAITAP is configured via YAML [Hydra](https://hydra.cc/) configs or CLI flags, and then ran via a CLI command.

This means it can be used either as:

- a standalone Python package, which stores the assessment outputs in the directory you specify. See [understanding outputs](understanding-outputs.md) for more details.
- a step in a larger MLOps pipeline, which forwards the assessment outputs to your tracking software (e.g. MLflow). See [tracking](../modules/tracking/configuration.md) for more details.

This gives you full flexibility to choose how you want to use RAITAP in your workflow.
