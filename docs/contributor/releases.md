# Releases

This project supports two ways to get a version onto PyPI. You only need one of them per release.

## Standard path: release pull request

Use this for normal releases.

1. Merge your work into `main` as usual, using [conventional commits](pull-requests.md) as titles.
2. Whenever one of those PRs is merged to main, the automation will create a new release PR, automatically handling version numbering and including all changes since the last release.
3. Once you want to release the new version, merge that automated release PR back into main.
4. The automation will publish to PyPI, then deploy the GitHub Pages documentation from the same release tag so the public site matches the package on PyPI.

## Alternate path: publish from a version tag

Use this only when you **already** have a correct **`v*.*.*`** tag on the commit you want on PyPI — for example if you must publish from a tag the standard path did not drive, or you are correcting a one-off situation your team agrees on.

1. Ensure the commit you want is tagged with a semver tag such as **`v1.2.3`** (leading `v`, three numeric parts).
2. **Push that tag** to GitHub (for example `git push origin v1.2.3`).
3. Publishing to PyPI is triggered from that tag push, then the documentation site is deployed from the same tag. You still do not upload the package from your laptop unless your team explicitly chooses a different process.

Coordinate with maintainers before using this path so you do not double-publish or publish the wrong commit.

## Moving from 0.x to stable 1.0.0

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
