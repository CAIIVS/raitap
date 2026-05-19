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

# Materialise sample images at native resolution (idempotent). The demo
# sample loader resizes to 224x224 which is fatal for detection; this step
# copies the cached images into ``images/`` so ``data.source`` (a real path)
# bypasses the demo loader.
& uv run python "$ScriptDir\scripts\fetch_images.py"
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
