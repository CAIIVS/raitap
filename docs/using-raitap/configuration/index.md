# Creating & running your own configuration

This section explains high level configuration principles for your use case.

```{toctree}
:maxdepth: 1
:caption: Configuration pages

general
kitchen-sink
global-config-options
```

(module-specific-configurations)=

## Module-specific configurations

RAITAP's pipeline is composed of modules. Each has their own configuration options. You can view them here:

```{toctree}
:maxdepth: 1
:caption: Module configuration docs
:hidden:

Model <../../modules/model/configuration>
Data <../../modules/data/configuration>
Transparency <../../modules/transparency/configuration>
Metrics <../../modules/metrics/configuration>
Tracking <../../modules/tracking/configuration>
```

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Model
:link: ../../modules/model/configuration
:link-type: doc

Configure which model RAITAP assesses.
:::

:::{grid-item-card} Data
:link: ../../modules/data/configuration
:link-type: doc

Configure the input data and labels.
:::

:::{grid-item-card} Transparency
:link: ../../modules/transparency/configuration
:link-type: doc

Configure explainers and visualisers.
:::

:::{grid-item-card} Metrics
:link: ../../modules/metrics/configuration
:link-type: doc

Configure evaluation metrics.
:::

:::{grid-item-card} Tracking
:link: ../../modules/tracking/configuration
:link-type: doc

Configure logging and experiment tracking.
:::

:::{grid-item-card} Reporting
:link: ../../modules/reporting/configuration
:link-type: doc

Configure report generation.
:::

::::
