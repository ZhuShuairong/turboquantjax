param(
    [string]$Distro = "Ubuntu"
)

$ErrorActionPreference = "Stop"

function Invoke-WSLCommand {
    param(
        [string]$Command
    )

    $full = "wsl -d $Distro -- bash -lc `"$Command`""
    Write-Host "Running: $full"
    & wsl -d $Distro -- bash -lc $Command
    return $LASTEXITCODE
}

Write-Host "=== TurboQuant WSL Setup Check ==="
Write-Host "Distro: $Distro"

[void](wsl --version 2>$null)
if (-not $?) {
    Write-Error "WSL does not appear to be installed or available in PATH."
    exit 1
}

Write-Host "WSL version detected."

$checks = @(
    @{ Name = "GPU visibility"; Cmd = "nvidia-smi" },
    @{ Name = "Python"; Cmd = "python3 --version" },
    @{ Name = "Pip"; Cmd = "python3 -m pip --version" },
    @{ Name = "JAX import"; Cmd = "python3 - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY" },
    @{ Name = "llama.cpp binaries"; Cmd = "command -v llama-cli || true; command -v llama-server || true" }
)

$failed = $false
foreach ($check in $checks) {
    Write-Host "`n--- $($check.Name) ---"
    $rc = Invoke-WSLCommand -Command $check.Cmd
    if ($rc -ne 0) {
        Write-Warning "Check failed: $($check.Name) (exit code $rc)"
        $failed = $true
    }
}

if ($failed) {
    Write-Host "`nOne or more checks failed. Review output above."
    exit 1
}

Write-Host "`nAll checks passed."
exit 0
