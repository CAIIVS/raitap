# General configuration guide

RAITAP is built on top of [Hydra](https://hydra.cc/), a powerful configuration framework for Python. It allows to configure all options via YAML files, and override them when running via the CLI.

These docs will explain just enough about Hydra to use RAITAP effectively. However, you might want to dive deeper into the [Hydra documentation](https://hydra.cc/docs/intro/).

## Guide to writing and using your own configuration

### 1. Write your configuration YAML

Hydra parses YAML files to understand which options to apply to the pipeline. Create a YAML file with the options you need.
You may find useful to refer to:

- the {doc}`global-config-options`
- the {ref}`module-specific-configurations`
- the {doc}`kitchen-sink`

If your workflow does not make it easy to use YAML files, you can rely 100% on a CLI command. See {ref}`cli-overriding` for more details.

### 2. Preview your configuration

You can preview the final, Hydra-parsed configuration before executing it. Run the following from the same directory:

```{install-tabs}
:uv:
uv run raitap --config-name assessment --cfg job # assuming your config is at `./assessment.yaml`

:pip:
raitap --config-name assessment --cfg job # assuming your config is at `./assessment.yaml`
```

### 3. Execute your configuration

```{install-tabs}
:uv:
uv run raitap --config-name assessment # assuming your config is `./assessment.yaml`

:pip:
raitap --config-name assessment # assuming your config is `./assessment.yaml`
```

## Some advanced Hydra features

(cli-overriding)=

### CLI overriding

Hydra does not only read from YAML files. It can also parse CLI option overrides.
In the following, we override some options from the
{doc}`../../modules/transparency/configuration`.

You can either set individual options:

```{install-tabs}
:uv:
uv run raitap --config-name assessment hardware=cpu transparency.myexplainer1.call.target=0

:pip:
raitap --config-name assessment hardware=cpu transparency.myexplainer1.call.target=0
```

Or override an entire nested value at once:

```{install-tabs}
:uv:
uv run raitap --config-name assessment "transparency.captum_saliency.visualisers=[{_target_: CaptumImageVisualiser, call: {show_sample_names: true}}]"

:pip:
raitap --config-name assessment "transparency.captum_saliency.visualisers=[{_target_: CaptumImageVisualiser, call: {show_sample_names: true}}]"
```

### Composing YAML files

Hydra allows you to compose multiple YAML files into a single configuration.
This is useful to avoid repeating the same options in multiple files.

The main mechanism for this is the `defaults` list.

```yaml
# assessment.yaml
defaults:
  - _self_ # inserts experiment_name and hardware from the current file into the final config
  - model: resnet50 # imports the other YAML file, see below
  - data: isic2018
  - transparency: shap_gradient
  - metrics: classification

experiment_name: "my-exp"
hardware: cpu

# resnet50.yaml
source: resnet50 # built-in torch model, see the Model module docs
```

Hydra composition is cascading top-down. Hence, you might want to control the
order of composition. This can be achieved using the `_self_` keyword (note the single underscores).

```yaml
defaults:
  - model: resnet50
  - model: vitb32
  - _self_

model:
  source: "./my-custom-model.onnx"
```

In the above example, the final config will use the custom ONNX model, because `_self_` is applied last.

### Batch runs

Hydra can execute multiple runs from a single command using `--multirun`.
This is useful when you want to compare several presets or override values in one go.

```{install-tabs}
:uv:
uv run raitap --multirun transparency=demo,shap_gradient experiment_name=demo,shap

:pip:
raitap --multirun transparency=demo,shap_gradient experiment_name=demo,shap
```

Hydra expands the comma-separated values into multiple runs. To inspect where each run
writes its outputs, see {doc}`../understanding-outputs`.

### Slurm integration

RAITAP supports distributed execution on HPC clusters via [Hydra's Submitit plugin](https://hydra.cc/docs/plugins/submitit_launcher/).
This allows you to run parameter sweeps as Slurm job arrays without writing custom submission scripts.

#### Prerequisites

1. Install the cluster extra when setting up RAITAP:

```{install-tabs}
:uv:
uv sync --extra cluster

:pip:
pip install raitap[cluster]
```

2. Ensure you have access to a Slurm-managed cluster and know your:
   - **Partition name** (e.g., `gpu`, `compute`)
   - **Account name** (required by most clusters for job accounting)

#### Quick start: Running sweeps with CLI overrides

The simplest approach is to use Hydra's built-in `submitit_slurm` launcher and override parameters directly from the CLI:

```{install-tabs}
:uv:
uv run --extra cluster --extra torch-cuda --extra captum --extra shap python \
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
- Your cluster setup changes between experiments

Hydra will:
1. Create a Slurm job array with one job per configuration combination
2. Submit up to `array_parallelism` jobs concurrently
3. Store logs in `${hydra.sweep.dir}/.submitit/<job_id>/`

#### Alternative: Creating a reusable launcher configuration

If you run sweeps frequently with the same resource settings, create a Hydra launcher preset.
For example, create `configs/hydra/launcher/my_cluster.yaml`:

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

# Cluster-specific (still override at runtime if needed)
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
  hydra/launcher=my_cluster \
  transparency=captum_ig,shap_gradient \
  data=my_dataset

:pip:
raitap --multirun \
  hydra/launcher=my_cluster \
  transparency=captum_ig,shap_gradient \
  data=my_dataset
```

#### Cluster environment setup

Many HPC clusters require environment setup before running jobs. Here's a typical workflow:

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
uv sync --extra cluster --extra torch-cuda --extra captum --extra shap --extra metrics
```

**4. Submit your multirun:**

Use the command from the "Quick start" section above.

#### Best practices for cluster execution

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
```
outputs/my_experiment/2026-04-13/14-30-00/
├── 0/  # First config combination
│   ├── transparency/
│   └── .hydra/
├── 1/  # Second config combination
├── ...
└── .submitit/
    ├── 12345_0_log.out
    ├── 12345_0_log.err
    └── 12345_submission.sh
```

#### Advanced: Custom cluster entry point

For complex setups (custom preprocessing, data staging, cleanup), create a dedicated entry point.
This pattern is useful when you need to:
- Stage large datasets from network storage to local scratch
- Dynamically resolve model or data paths based on environment variables
- Clean up temporary files after runs
- Set environment variables or configure libraries

Example structure:

```python
# my_cluster_entrypoint.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import hydra

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def prepare_environment(config: DictConfig) -> None:
    """Set up cluster-specific environment and stage data."""
    # Example: Configure matplotlib cache for headless environments
    cache_dir = Path(os.environ.get("SCRATCH", "/tmp")) / "mpl-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    
    # Example: Resolve relative paths from repo root
    repo_root = Path(__file__).resolve().parents[2]
    if hasattr(config.model, "source"):
        model_path = repo_root / config.model.source
        config.model.source = str(model_path)
    
    logger.info("Environment prepared for cluster execution")

def cleanup_temp_files(config: DictConfig) -> None:
    """Clean up temporary files after job completion."""
    # Example: Clean up job-local scratch space
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        scratch_dir = Path(os.environ.get("SCRATCH", "/scratch")) / f"job-{job_id}"
        if scratch_dir.exists():
            import shutil
            shutil.rmtree(scratch_dir)
            logger.info("Cleaned up temporary files in %s", scratch_dir)

@hydra.main(version_base="1.3", config_path="configs", config_name="my_config")
def main(config: DictConfig) -> None:
    from raitap.run import run
    
    # Pre-processing
    prepare_environment(config)
    
    # Run assessment
    run(config)
    
    # Post-processing
    try:
        cleanup_temp_files(config)
    except Exception as e:
        logger.warning("Cleanup encountered an error (non-fatal): %s", e)
    
    logger.info("Cluster run complete!")

if __name__ == "__main__":
    main()
```

Then submit via:
```bash
uv run --extra cluster --extra torch-cuda --extra captum --extra shap python \
  my_cluster_entrypoint.py \
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

#### Troubleshooting

**Jobs fail immediately:**
- Check `partition` and `account` are valid for your cluster (use `sinfo` and `scontrol show partition`)
- Verify resource requests don't exceed partition limits
- Review `.submitit/<job-id>/<job-id>_0_log.err` for error messages
- Ensure required Python modules are loaded before submission

**Jobs run out of memory:**
- Some transparency methods (Occlusion, SHAP) are particularly memory-intensive
- Reduce batch sizes in transparency configs

**Module or import errors:**
- Ensure all required extras are installed: `--extra cluster --extra torch-cuda --extra captum --extra shap`

**UV cache issues:**
- Set `UV_PATH` to a user-specific directory with sufficient quota
- Clear cache if needed: `rm -rf $UV_PATH/cache`
- Check available disk space: `df -h $UV_PATH`

For more details on Submitit configuration, see the [Hydra Submitit documentation](https://hydra.cc/docs/plugins/submitit_launcher/).
