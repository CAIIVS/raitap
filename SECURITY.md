# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** via GitHub's
[private security advisories](https://github.com/CAIIVS/raitap/security/advisories/new)
rather than opening a public issue.

When reporting, please include:

- Affected version (release tag or commit SHA).
- A minimal reproducer (config, command, traceback).
- Impact and any known workaround.

We will acknowledge receipt as soon as we can and work with you on a fix and
disclosure timeline. Please do not publicly disclose the issue until a fix is
available.

## Supported versions

Pre-1.0 releases: only the latest released minor version receives security
fixes. Older versions are unsupported.

## Out of scope

- Vulnerabilities in third-party libraries that raitap wraps — please report
  those upstream. Open an issue here only if raitap's adapter exposes the
  vulnerable surface in a way the upstream library would not on its own.
- Issues that depend on running adversarial models or untrusted code that the
  user explicitly loads (e.g. arbitrary `torch.load` of untrusted checkpoints).
  raitap inherits the trust model of the model file format the user chooses.
