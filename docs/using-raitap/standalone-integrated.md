# Standalone & integrated MLOps usages

RAITAP is can be configured via YAML Hydra configs or CLI flags, and is ran via a CLI command.

This means it can be used either as:

- a standalone Python package, which stores the assessment outputs in the directory you specified.
- a step in a larger MLOps pipeline, which forwards the assessment outputs to your tracking software (e.g. MLFLow).
