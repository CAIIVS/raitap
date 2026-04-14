# Development environment setup

## Setup steps

### 0. Prerequisites

- The repository cloned.
- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Install dependencies

RAITAP supports many machine and model configurations. To avoid conflicts, only install the dependencies that match your setup. Linting and testing of all configurations is done on the CI.

1. Choose your execution dependency group from the following table:

    |       | CPU         | CUDA         | Intel GPU     |
    | ----- | ----------- | ------------ | ------------- |
    | Torch | `torch-cpu` | `torch-cuda` | `torch-intel` |
    | ONNX  | `onnx-cpu`  | `onnx-cuda`  | `onnx-intel`  |

    :::{note}

    - CUDA corresponds to NVIDIA GPUs.
    - `torch-intel` uses the Intel XPU API directly.
    - `onnx-intel` uses the OpenVINO ONNX Runtime.
    - Apple MPS support is coming soon.
    :::

2. Decide which optional dependencies you need. It can either be a whole module (e.g. `transparency`, `metrics`, `tracking`) or a specific framework/integration (e.g. `shap`, `captum`, `mlflow`).

3. Consolidate all the dependency groups into a single command and run it. Notice the `--group dev`   flag to install the contributor environment. Here an example:

    ```shell
    uv sync --group dev --extra onnx-cpu --extra transparency
    ```

    :::{warning}
    Do not run the `sync` commands separately. The latest run will override the previous ones.
    :::

### 2. Install the commit message hook

```bash
uv run pre-commit install --hook-type commit-msg
```

It will ensure your commit messages follow the [conventional commits](https://www.conventionalcommits.org/) format.

### 3. Install the VSCode extensions

If you use VSCode, install the recommended extensions (open command palette > search for "Extensions: Show Recommended Extensions").

If not, install the equivalents for your IDE (see `.vscode/extensions.json`), and ensure it uses the Python interpreter from the `.venv`, NOT your global Python installation.

## Useful commands

- `uv run pytest` runs the test suite. Depending on the dependencies installed, some tests might fail, which is fine. The CI is the real source of truth.
- `uv run ruff check --fix .` lints all Python files and applies auto-fixes where possible. Omit the `--fix` flag to only check for issues without modifying files.
- `uv run ruff format .` formats all Python files according to the configured style.
- `uv run pyright` runs type checking on all Python files to catch type errors.
- `uv run cz commit` opens an interactive prompt to create a commit message following the conventional commits format. You can also use the VSCode extension instead.
- `uv run docs-preview` starts a local server to preview the documentation, which supports hot-reloading.
