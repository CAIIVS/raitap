# Pull requests and commit messages

This project relies on **Conventional Commits** for automation: **Release Please** builds the changelog and release versions from what lands on `main`, and CI enforces the same rules on **pull request titles**.

:::{warning}
You should read this page before opening a PR.
:::

## Why the PR title matters

We use **squash and merge** by default. That means each merged PR becomes **a single commit** on `main`, and GitHub usually takes the **PR title** as the **squash commit subject**.

So the **PR title is not cosmetic**: it is what **Release Please** and the history on `main` see. A vague title (for example `Update stuff` or `Fix bug`) breaks changelog quality and can make releases harder to reason about.

## Required format (Conventional Commits)

Use this shape:

```text
<type>(<scope>): <subject>
```

- **`<type>`** — required. One of: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- **`<scope>`** — optional. If you use a scope, it **must** be one of: `transparency`, `tracking`, `metrics`, `docs`, `infra`, `deps`, `misc`, `config`, `model`, `data`, `reporting`.
- **`<subject>`** — required, non-empty (describe the change in the imperative mood, e.g. “add SHAP batch helper”, not “added…”).

The canonical regex lives in **`pyproject.toml`** under **`[tool.commitizen.customize]`** as **`schema_pattern`**. Local checks can use Commitizen; CI runs a dedicated **Semantic PR title** workflow on PRs targeting `main`.

### Examples

| Valid                                           | Notes                                           |
| ----------------------------------------------- | ----------------------------------------------- |
| `feat(transparency): add batch SHAP visualiser` | Type + scope + subject                          |
| `fix: correct dtype in metrics export`          | Scope omitted (allowed)                         |
| `docs: describe reporting PDF options`          | `docs` as type for documentation-only changes   |
| `feat(api)!: remove deprecated explainer alias` | **`!`** marks a **breaking** change (see below) |

### Breaking changes

Because the PR title is a **single line**, breaking changes must be visible there: put **`!`** immediately after the **type** or after the **closing parenthesis of the scope**, for example `feat!:` or `feat(api)!:`.

For background, see [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

## What CI enforces

- A workflow validates the **PR title** on pull requests to **`main`**. If the title does not match the rules, the check fails and the branch protection (when enabled) blocks merge.
- PRs from **Release Please** automation (branches whose names start with **`release-please--`**) **skip** this check so release PR titles can follow Release Please’s own pattern.

## Tips

- Prefer **editing the PR title** when the bot or a reviewer asks — faster than rewriting branch history.
- In GitHub, enable **“Default to PR title for squash merge commit messages”** for the repository so the squash subject stays aligned with the title.
- Individual commits on your branch can still be messy during development; with squash merge, reviewers focus on the **final PR title**. Keeping commits conventional anyway helps if you ever rebase or use merge commits elsewhere.
