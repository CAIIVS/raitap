# Faster R-CNN / Udacity detection demo entry point (Windows PowerShell).
# Syncs the required extras and runs raitap against the bundled
# assessment.yaml. Forwards extra CLI args.

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Extras = @(
    '--extra', 'torch-cpu',
    '--extra', 'captum',
    '--extra', 'metrics',
    '--extra', 'reporting'
)

& uv sync @Extras
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# --acknowledge-preprocessing-off silences the no-preprocessing warning;
# torchvision detection models do their own internal resize/normalise, and
# adding a RAITAP-level preprocessing step would break box coord alignment.
& uv run @Extras raitap `
    --config-dir $ScriptDir `
    --config-name assessment `
    --acknowledge-preprocessing-off `
    @args
exit $LASTEXITCODE
