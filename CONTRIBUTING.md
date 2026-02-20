# Contributing to this project

## Setting up the environment

### Prerequisites

* Python 3.13 or higher
* [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone the repository
2. Install the dependencies by running

    ```bash
    uv sync
    ```

3. Install the commit message hook

    ```bash
    uv run pre-commit install --hook-type commit-msg
    ```

    It will ensure your commit messages follow the [conventional commits](https://www.conventionalcommits.org/) format.

4. If you use VSCode, install the recommended extensions. If not, install the equivalents for your IDE, and ensure it uses the Python interpreter from the `.venv` folder.

## Workflow

1. To modify the `main` branch, create a new branch and open a pull request.
2. Ensure all GH Action checks pass before merging. The actions will handle formatting and linting, pushing auto-fixes if possible.

## Useful commands

* `uv sync` installs the dependencies defined in `uv.lock`.
* `uv run ruff check --fix .` lints all Python files and applies auto-fixes where possible. Omit the `--fix` flag to only check for issues without modifying files.
* `uv run ruff format .` formats all Python files according to the configured style.
* `uv run pyright` runs type checking on all Python files to catch type errors.
* `uv run cz commit` opens an interactive prompt to create a commit message following the conventional commits format. You can also use the VSCode extension.
