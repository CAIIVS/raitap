# Tracking

The tracking module selects an optional backend that records run metadata,
metrics, and artifacts after the local assessment finishes.

Unlike metrics and transparency, tracking does not create its own dedicated
subdirectory under the run output. Instead, it exports the config, dataset
metadata, scalar metrics, and existing artifacts to the configured backend.

Tracking is disabled by default. Enable it by selecting a tracking preset such
as `tracking=mlflow`.

```{toctree}
:maxdepth: 1
:caption: Tracking module documentation

configuration
frameworks-and-libraries
output
```
