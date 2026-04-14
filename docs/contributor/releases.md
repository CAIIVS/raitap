# Releases

This project supports two ways to get a version onto PyPI. You only need one of them per release.

## Standard path: release pull request

Use this for normal releases.

1. Land work on `main` using [conventional commits](pull-requests.md) and good PR titles, as usual.
2. When enough changes are on `main`, a **release pull request** appears (opened and updated by automation). It proposes the next version and release notes.
3. **Review** that PR. When it looks right, **merge** it into `main`.
4. After the merge, **you do not run a local publish command**. The GitHub release and PyPI upload are handled automatically from that merge.

You do **not** hand-edit the package version on `main` for this path—the release PR is the place the version is set for publication.

## Alternate path: publish from a version tag

Use this only when you **already** have a correct **`v*.*.*`** tag on the commit you want on PyPI—for example if you must publish from a tag the standard path did not drive, or you are correcting a one-off situation your team agrees on.

1. Ensure the commit you want is tagged with a semver tag such as **`v1.2.3`** (leading `v`, three numeric parts).
2. **Push that tag** to GitHub (for example `git push origin v1.2.3`).
3. Publishing to PyPI is triggered from that tag push. You still do not upload the package from your laptop unless your team explicitly chooses a different process.

Coordinate with maintainers before using this path so you do not double-publish or publish the wrong commit.

## Which path should I use?

- **Day to day:** wait for the release PR, then merge it when you are ready to ship.
- **Exceptional cases only:** tag push, and only with maintainer alignment.
