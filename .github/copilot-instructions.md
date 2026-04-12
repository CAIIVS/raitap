# Copilot code review (repository)

When reviewing pull requests on GitHub:

1. **PR title:** Must follow [Conventional Commits](https://www.conventionalcommits.org/) with the same **types** and optional **scopes** as `[tool.commitizen.customize] schema_pattern` in `pyproject.toml` (types include feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert; scopes include transparency, tracking, metrics, docs, infra, deps, misc, config, model, data, reporting). Scope is optional. Breaking changes should use `!` after the type or scope (e.g. `feat!:` or `feat(api)!:`) because the PR title becomes the squash merge subject.

2. **Breaking changes:** If the diff removes or renames public APIs, changes config defaults, or migrates persisted formats in a non-backward-compatible way, the PR title should signal breaking (`!`) or the description should state **BREAKING CHANGE**. CI enforces title shape; you supplement by flagging mismatches between diff and title.

3. **Limits:** Copilot does not replace required checks. GitHub includes the PR title and body in the review prompt per GitHub Copilot code review documentation.
