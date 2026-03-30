# Wrapper to run JAX tests via WSL.

param(
    [string]$Distro = "Ubuntu",
    [double]$JaxMemFraction = 0.8,
    [string]$CondaEnv = "turboquant-jax",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)

    $ResolvedPath = (Resolve-Path $WindowsPath).Path
    if ($ResolvedPath -match '^([A-Za-z]):\\(.*)$') {
        $drive = $Matches[1].ToLower()
        $suffix = ($Matches[2] -replace '\\', '/')
        return "/mnt/$drive/$suffix"
    }

    throw "Unsupported Windows path format: $ResolvedPath"
}

Write-Host "Running tests/test_turboquant.py via WSL distro '$Distro'..." -ForegroundColor Cyan

try {
    wsl.exe --status | Out-Null
} catch {
    Write-Host "Could not find wsl.exe. Is WSL installed?" -ForegroundColor Red
    exit 1
}

$RepoPathWsl = Convert-ToWslPath -WindowsPath $RepoRoot
if ([string]::IsNullOrWhiteSpace($RepoPathWsl)) {
    Write-Host "Could not resolve WSL path for $RepoRoot" -ForegroundColor Red
    exit 1
}

$SkipInstallFlag = if ($SkipInstall) { "1" } else { "0" }
$bashCommand = "set -euo pipefail; cd '$RepoPathWsl'; if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then source ~/miniconda3/etc/profile.d/conda.sh; if conda env list | grep -E '^$CondaEnv[[:space:]]' >/dev/null; then conda activate $CondaEnv; else conda env create -f environment.yml; conda activate $CondaEnv; fi; fi; export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_MEM_FRACTION=$JaxMemFraction; if [ '$SkipInstallFlag' != '1' ]; then if command -v python >/dev/null 2>&1; then if ! python -c 'import jax, scipy' >/dev/null 2>&1; then echo 'Installing missing test dependencies...'; python -m pip install -U scipy 'jax[cuda12]'; fi; else if ! python3 -c 'import jax, scipy' >/dev/null 2>&1; then echo 'Installing missing test dependencies...'; python3 -m pip install -U scipy 'jax[cuda12]'; fi; fi; fi; if command -v python >/dev/null 2>&1; then python tests/test_turboquant.py; else python3 tests/test_turboquant.py; fi"

wsl.exe -d $Distro bash -lc $bashCommand

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nTest completed successfully." -ForegroundColor Green
} else {
    Write-Host "`nTest failed or could not run." -ForegroundColor Red
    exit $LASTEXITCODE
}
