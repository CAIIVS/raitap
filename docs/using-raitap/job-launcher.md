# Using job launcher (Slurm...)

RAITAP supports launcher-based remote execution via [Hydra's Submitit plugin](https://hydra.cc/docs/plugins/submitit_launcher/).
This is useful for Slurm-managed HPC clusters, shared GPU servers, and other environments where you want Hydra to submit or fan out jobs for you.

## 0. Prerequisites

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

## Quick usage: Running sweeps with CLI overrides

The simplest approach is to use Hydra's built-in `submitit_slurm` launcher and override parameters directly from the CLI:

```{install-tabs}
:uv:
uv run --extra launcher --extra torch-cuda --extra captum --extra shap python \
  path/to/your_entrypoint.py \
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
python path/to/your_entrypoint.py \
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

This approach works well when:

- You don't need to repeat the same launcher settings across multiple runs
- You want to quickly test different resource configurations
- Your remote execution setup changes between experiments

Hydra will:

1. Create a Slurm job array with one job per configuration combination
2. Submit up to `array_parallelism` jobs concurrently
3. Store logs in `${hydra.sweep.dir}/.submitit/<job_id>/`

## Advanced usage: Creating a reusable launcher configuration

If you run sweeps frequently with the same resource settings, create a Hydra launcher preset.
For example, create `configs/hydra/launcher/my_launcher.yaml`:

```yaml
# @package hydra.launcher
defaults:
  - submitit_slurm

# Job resource defaults
submitit_folder: ${hydra.sweep.dir}/.submitit/%j
timeout_min: 240
cpus_per_task: 8
gpus_per_node: 1
tasks_per_node: 1
mem_gb: 48
nodes: 1
name: ${hydra.job.name}

# Scheduler-specific defaults (still override at runtime if needed)
partition: gpu
account: myproject

# Optional parameters
qos: null
constraint: null
gres: null

# Array job parallelism
array_parallelism: 8
signal_delay_s: 120
```

Then use it with shorter commands:

```{install-tabs}
:uv:
uv run raitap --multirun \
  hydra/launcher=my_launcher \
  transparency=captum_ig,shap_gradient \
  data=my_dataset

:pip:
raitap --multirun \
  hydra/launcher=my_launcher \
  transparency=captum_ig,shap_gradient \
  data=my_dataset
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

**3. Sync dependencies:**

```bash
uv sync --extra launcher --extra torch-cuda --extra captum --extra shap --extra metrics
```

**4. Submit your multirun:**

Use the command from the "Quick start" section above.

## Best practices for launcher-based execution

**Resource management:**
- Set `timeout_min` generously; some explainability methods (especially SHAP) can take quite a while
- Adjust `mem_gb` based on your model size and batch configuration
- Use `cpus_per_task` to match the number of dataloader workers
- For single-GPU jobs, set `gpus_per_node=1`, `tasks_per_node=1`, `nodes=1`
- Use `array_parallelism` to limit concurrent jobs (e.g., 8 for an 8-GPU node)

**Data staging:**
- For large datasets, consider copying data to fast local storage before processing
- Many clusters provide job-local scratch space (e.g., `/scratch`, `/tmp`)
- Use a custom entry point script to handle staging logic (see "Advanced" section)
- Clean up temporary files after job completion

**Monitoring jobs:**
```bash
# Check job status
squeue -u $USER

# Check detailed job info
scontrol show job <job-id>

# Cancel a job
scancel <job-id>

# Cancel all your jobs
scancel -u $USER

# View job logs
tail -f outputs/multirun/.../submitit/<job-id>/<job-id>_0_log.out
```

**Output organization:**
Configure `hydra.sweep.dir` to organize outputs by experiment:

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

For more details on Submitit configuration, see the [Hydra Submitit documentation](https://hydra.cc/docs/plugins/submitit_launcher/).
