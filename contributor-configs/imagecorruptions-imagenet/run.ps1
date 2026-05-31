# Average-case (ImageNet-C corruptions) demo entry point (Windows PowerShell).
# Syncs the required extras and runs raitap against the bundled assessment.yaml.
# Forwards extra CLI args.

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Extras = @(
    '--extra', 'torch-intel',
    '--extra', 'imagecorruptions',
    '--extra', 'metrics',
    '--extra', 'reporting'
)

& uv sync @Extras
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& uv run @Extras raitap `
    --config-dir $ScriptDir `
    --config-name assessment `
    @args
exit $LASTEXITCODE
