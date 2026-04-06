# Contributing to this project

## Setting up the environment

### Prerequisites

* Python 3.13 or higher
* [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone the repository
2. Install the dependencies by running

    ```bash
    uv sync --group dev --extra torch-cpu --extra mlflow --extra shap --extra captum --extra metrics --extra onnx-cpu
    ```

    On Apple Silicon, this same environment also covers MPS for Torch and
    CoreML-capable `onnxruntime` builds for ONNX validation.

3. Install the commit message hook

    ```bash
    uv run pre-commit install --hook-type commit-msg
    ```

    It will ensure your commit messages follow the [conventional commits](https://www.conventionalcommits.org/) format.

4. If you use VSCode, install the recommended extensions. If not, install the equivalents for your IDE, and ensure it uses the Python interpreter from the `.venv` folder.

## Workflow

1. To modify the `main` branch, create a new branch and open a pull request.
2. Ensure all GH Action checks pass before merging. The actions will handle formatting and linting, pushing auto-fixes if possible.

## Adding Features

### Adding New XAI Methods

The most common contribution is adding new explainability methods:

**Quick example (Captum method):**

1. Test compatibility:
   ```python
   # tests/transparency/test_captum_explainer.py
   def test_new_method(simple_cnn, sample_images):
       explainer = CaptumExplainer("NewMethod")
       attributions = explainer.explain(simple_cnn, sample_images, target=0)
       assert attributions.shape == sample_images.shape
   ```

2. Add to registry (one line):
   ```python
   # src/raitap/transparency/methods.py
   class Captum:
       NewMethod = ExplainerMethod()  # ← Add this line
   ```

3. Done! Use immediately:
   ```python
   from raitap.transparency.methods import Captum
   explainer = create_explainer(Captum.NewMethod)
   ```

**For complete walkthroughs, see:**
- **[Extending Guide](docs/extending.md)** - Step-by-step instructions for:
  - Adding Captum/SHAP methods
  - Adding new frameworks (OmniXAI, Alibi, etc.)
  - Adding visualisers and datasets
- **[Architecture Guide](docs/architecture.md)** - Understand the design patterns

### Understanding the Codebase

Before making significant changes, read:
- **[docs/architecture.md](docs/architecture.md)** - Design patterns, extension points, and why things work the way they do
- **[docs/configuration.md](docs/configuration.md)** - How Hydra configs work
- **[Development.md](Development.md)** - MVP roadmap and planned features

## Useful commands

* `uv sync --group dev --extra torch-cpu --extra mlflow --extra shap --extra captum --extra metrics --extra onnx-cpu` installs the default contributor environment. On Apple Silicon, it is also the standard setup for MPS/CoreML checks.
* `uv run pytest` runs the test suite.
* `uv run ruff check --fix .` lints all Python files and applies auto-fixes where possible. Omit the `--fix` flag to only check for issues without modifying files.
* `uv run ruff format .` formats all Python files according to the configured style.
* `uv run pyright` runs type checking on all Python files to catch type errors.
* `uv run cz commit` opens an interactive prompt to create a commit message following the conventional commits format. You can also use the VSCode extension.
