# Quantus explanation-quality grading demo entry point (Windows PowerShell).
# Syncs the required extras and runs raitap against the bundled assessment.yaml.
# Forwards extra CLI args.

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Run from the repo root: raitap's dev-install detection checks the cwd's
# pyproject.toml, and only then uses `uv sync` (a bare `uv add raitap[...]`
# inside the repo fails as a self-dependency).
Set-Location (Join-Path $ScriptDir '..' '..')

$Extras = @(
    '--extra', 'torch-intel',
    '--extra', 'captum',
    '--extra', 'quantus',
    '--extra', 'metrics',
    '--extra', 'html'
)

& uv sync @Extras
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& uv run @Extras raitap `
    --config-dir $ScriptDir `
    --config-name assessment `
    @args
exit $LASTEXITCODE
