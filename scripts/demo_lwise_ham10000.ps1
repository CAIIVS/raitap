# Thesis demo: layer-wise HAM10000 assessment example (PowerShell).
#
# Drops the Marabou block at the CLI (`~robustness.marabou_linf`) so the
# demo only runs Captum explainers + empirical attacks — Marabou is in
# scope for the separate UC1 MNIST demo.
$ErrorActionPreference = 'Stop'

uv run `
    --extra torch-cpu `
    --extra captum `
    --extra torchattacks `
    --extra metrics `
    --extra reporting `
    raitap `
        --config-dir "$PWD/examples/lwise-ham10000" `
        --config-name assessment `
        '~robustness.marabou_linf' `
        @args
