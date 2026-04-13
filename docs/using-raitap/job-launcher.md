# Using job launchers (Slurm...)

RAITAP supports launcher-based remote execution via [Hydra's Submitit plugin](https://hydra.cc/docs/plugins/submitit_launcher/).

This is useful for Slurm-managed HPC clusters, shared GPU servers, and other environments where you want Hydra to submit or fan out jobs for you.

## Prerequisites

1. Install the `launcher` extra when setting up RAITAP:

    ```{install-tabs}
    :uv:
    uv add "raitap[launcher]"

    :pip:
    pip install "raitap[launcher]"
    ```

2. Ensure you have access to a Slurm-managed remote server and know your:
   - **Partition name** (e.g., `gpu`, `compute`)
   - **Account name** (required by most clusters for job accounting)

(job-launcher-slurm-sweeps)=

## Slurm sweeps

Use Hydra's `submitit_slurm` launcher for multi-runs: each combination of config choices becomes a Slurm array task.

### YAML example

Compose the sweep, launcher selection, and Slurm resources in one experiment YAML. Use Hydra's `defaults` list to choose the dataset, transparency presets, and the `submitit_slurm` launcher; see {ref}`composing-yaml-files`. Submitit's Slurm resource fields live under `hydra.launcher`.

```yaml
# assessment.yaml
defaults:
  - _self_
  - hydra/launcher: submitit_slurm # this is a preset from the plugin
  - data: my_dataset
  - transparency:
      - captum_ig
      - shap_gradient
  # model, metrics, tracking, ... — see the configuration guides.

hydra:
    launcher:
        partition: gpu
        account: myproject
        timeout_min: 240
        cpus_per_task: 8
        gpus_per_node: 1
        mem_gb: 48
        array_parallelism: 8
    sweep:
        dir: outputs/my_experiment/${now:%Y-%m-%d}/${now:%H-%M-%S} # see the output section lower on this page
        subdir: ${hydra.job.num}
```

```{install-tabs}
:uv:
uv run raitap --multirun --config-name assessment

:pip:
raitap --multirun --config-name assessment
```

### CLI override example

The same sweep and resource knobs expressed as overrides:

```{install-tabs}
:uv:
uv run raitap \
  --multirun \
  hydra/launcher=submitit_slurm \
  transparency=captum_ig,shap_gradient \
  data=my_dataset \
  hydra.launcher.partition=gpu \
  hydra.launcher.account=myproject \
  hydra.launcher.timeout_min=240 \
  hydra.launcher.cpus_per_task=8 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=48 \
  hydra.launcher.array_parallelism=8

:pip:
raitap \
  --multirun \
  hydra/launcher=submitit_slurm \
  transparency=captum_ig,shap_gradient \
  data=my_dataset \
  hydra.launcher.partition=gpu \
  hydra.launcher.account=myproject \
  hydra.launcher.timeout_min=240 \
  hydra.launcher.cpus_per_task=8 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=48 \
  hydra.launcher.array_parallelism=8
```

YAML keeps Slurm settings next to the rest of the experiment under version control. CLI overrides are handy for one-off runs or quick resource adjustments.

(job-launcher-launcher-preset)=

### Reusing the same launcher preset

If you run sweeps across several experiments on the same cluster, extract the `hydra.launcher.*` settings into a shared preset under `configs/hydra/launcher/` instead of repeating them in every experiment YAML.

```yaml
# configs/hydra/launcher/my_launcher.yaml
# @package hydra.launcher
defaults:
  - submitit_slurm

submitit_folder: ${hydra.sweep.dir}/.submitit/%j
timeout_min: 240
cpus_per_task: 8
gpus_per_node: 1
tasks_per_node: 1
mem_gb: 48
nodes: 1
name: ${hydra.job.name}
partition: gpu
account: myproject
qos: null
constraint: null
gres: null
array_parallelism: 8
signal_delay_s: 120
```

Reference it in your experiment YAML in place of `submitit_slurm` and the inline `hydra.launcher.*` block:

```yaml
# assessment.yaml
defaults:
  - _self_
  - hydra/launcher: my_launcher  # replaces submitit_slurm + inline hydra.launcher.*
  - data: my_dataset
  - transparency:
      - captum_ig
      - shap_gradient
```

The run command is unchanged:

```{install-tabs}
:uv:
uv run raitap --multirun --config-name assessment

:pip:
raitap --multirun --config-name assessment
```

## Remote environment setup

Many shared servers and HPC systems require some environment setup before running jobs. Here's a typical workflow:

**1. Configure UV cache location (if using uv):**

Some clusters have shared `/tmp` or home directory quotas. Set a user-specific cache location:

```bash
export UV_PATH="/cluster/home/$USER/.uv"
```

**2. Load required modules:**

```bash
# Load Python (adjust version to what's available)
module load python/3.13.2

# Load uv if provided as a module
VENV=raitap-env module load uv/0.6.12
```

**3. Add dependencies:**

See {ref}`execution-dependencies` for the dependencies you need to add. Below is an example:

```bash
uv add "raitap[launcher,torch-cuda,captum,shap,metrics]"
```

**4. Submit your multirun:**

```{install-tabs}
:uv:
uv run raitap --multirun --config-name assessment

:pip:
raitap --multirun --config-name assessment
```

## Best practices

**Resource management:**

- Set `timeout_min` generously; some explainability methods (especially SHAP) can take quite a while
- Adjust `mem_gb` based on your model size and batch configuration
- Use `cpus_per_task` to match the number of dataloader workers
- For single-GPU jobs, set `gpus_per_node=1`, `tasks_per_node=1`, `nodes=1`
- Use `array_parallelism` to limit concurrent jobs (e.g., 8 for an 8-GPU node)

## Output

Each job in the array writes its results under `hydra.sweep.dir`. Configure it in your experiment YAML to organize outputs by experiment:

```yaml
hydra:
  sweep:
    dir: outputs/my_experiment/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
```

This creates a structure like:

```text
outputs/my_experiment/2026-04-13/14-30-00/
├── 0/                                     # First job in the array
│   ├── transparency/
│   └── .hydra/
├── 1/                                     # Second config combination
├── ...
└── .submitit/
    ├── 12345_0_log.out
    ├── 12345_0_log.err
    └── 12345_submission.sh
```

## Useful Slurm advice

The following tips apply to Slurm generally and are not specific to RAITAP.

**Data staging:**

- For large datasets, consider copying data to fast local storage before processing
- Many clusters provide job-local scratch space (e.g., `/scratch`, `/tmp`)
- Use a setup script in your Slurm job to handle staging logic
- Clean up temporary files after job completion

**Monitoring jobs:**

```bash
# Check job status
squeue -u $USER

# Check detailed job info
scontrol show job <job-id>

# Cancel a job / all your jobs
scancel <job-id>
scancel -u $USER

# View job logs (Hydra writes these under the sweep dir)
tail -f outputs/multirun/.../submitit/<job-id>/<job-id>_0_log.out
```

For more details on Submitit configuration, see the [Hydra Submitit documentation](https://hydra.cc/docs/plugins/submitit_launcher/).
