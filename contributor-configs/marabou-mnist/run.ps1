# Marabou MNIST entry point (Windows). Marabou wheels are Linux-only +
# Python 3.11, so this delegates to WSL. Requires `wsl` on PATH and a
# distro with `uv` installed.

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Translate the Windows script path into the WSL view (D:\... -> /mnt/d/...).
# wsl.exe strips backslashes from forwarded args, so feed wslpath a
# forward-slash path (D:/... which wslpath -a also accepts).
$WslDir = (& wsl wslpath -a -u ($ScriptDir -replace '\\', '/')).Trim()

# Quote each forwarded arg for the bash side.
$ForwardedArgs = ($args | ForEach-Object {
    "'" + ($_ -replace "'", "'\''") + "'"
}) -join ' '

$BashCmd = "bash '$WslDir/run.sh' $ForwardedArgs"

& wsl -- bash -lc $BashCmd
exit $LASTEXITCODE
