# Moving from 0.x to stable 1.0.0

This page explains what to change when the project moves from **major version 0** to **stable 1.0.0**.

## 1. Release Please

In **`release-please-config.json`**, under **`packages["."]`**, remove **`bump-minor-pre-major`** and **`bump-patch-for-minor-pre-major`**, or set both to **`false`**.

After that, a **breaking** conventional commit (`feat!:`, `fix!:`, or a **`BREAKING CHANGE`** footer) will increase the **major** version as usual (for example **0.5.0 → 1.0.0**), and **`feat`** will bump **minor** again per normal SemVer.

## 2. Commitizen

In **`pyproject.toml`**, under **`[tool.commitizen]`**, set **`major_version_zero = false`** (or remove the line; the default is **`false`**).

Otherwise **`cz bump`** can keep treating breaking changes on **0.x** as minor bumps while Release Please does not, which is confusing.

## 3. Update the contributor guide

Delete this file and update {doc}`./index` to remove the `going-stable-1-0`toctree entry.

## 4. Cutting 1.0.0

Pick one approach:

- **Natural bump:** After the two changes above, merge a PR whose squash subject marks a **breaking** change; the next Release Please release PR should propose **1.0.0**.
- **Explicit version:** Put **`Release-As: 1.0.0`** in the **body** of the commit you merge to **`main`** (see [Release Please](https://github.com/googleapis/release-please) docs). That forces the next release version without relying on a breaking commit alone.

Review **changelog**, **docs**, and **PyPI** metadata before merging the **1.0.0** release PR.
