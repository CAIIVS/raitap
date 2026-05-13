:<<"::CMD"
@echo off
REM ====================================================================
REM Thesis demo: layer-wise HAM10000 assessment example.
REM
REM Polyglot cmd / bash script — invoke as `.\scripts\demo_lwise_ham10000.cmd`
REM from cmd.exe / PowerShell, or `bash scripts/demo_lwise_ham10000.cmd`
REM from any POSIX shell. Bash sees the cmd block as a heredoc to `:` and
REM skips it; cmd parses `:<<"::CMD"` as a label and runs the cmd block.
REM
REM Drops the Marabou block at the CLI (`~robustness.marabou_linf`) so
REM the demo only runs Captum explainers + empirical attacks — Marabou
REM is in scope for the separate UC1 MNIST demo.
REM ====================================================================
uv run ^
    --extra torch-intel ^
    --extra captum ^
    --extra torchattacks ^
    --extra metrics ^
    --extra reporting ^
    raitap ^
        --config-dir "%CD%\examples\lwise-ham10000" ^
        --config-name assessment ^
        "~robustness.marabou_linf" ^
        %*
exit /b
::CMD

# Bash branch — defaults to torch-cpu so Linux/WSL doesn't pull the
# multi-GB Intel oneAPI runtime. Override with `TORCH_EXTRA=torch-cuda
# bash scripts/demo_lwise_ham10000.cmd` etc.
set -euo pipefail
TORCH_EXTRA="${TORCH_EXTRA:-torch-cpu}"
uv run \
    --extra "$TORCH_EXTRA" \
    --extra captum \
    --extra torchattacks \
    --extra metrics \
    --extra reporting \
    raitap \
        --config-dir "$PWD/examples/lwise-ham10000" \
        --config-name assessment \
        '~robustness.marabou_linf' \
        "$@"
