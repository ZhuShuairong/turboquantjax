$ErrorActionPreference = "Stop"

$bashCommand = "set -euo pipefail; mkdir -p ~/TQ-Experimentation; rm -rf ~/TQ-Experimentation/turboquant-jax; cp -r /mnt/c/Users/zshua/Downloads/TQ-Experimentation/turboquant-jax ~/TQ-Experimentation/turboquant-jax; source ~/miniconda3/etc/profile.d/conda.sh; if conda env list | grep -E '^turboquant-jax[[:space:]]'; then conda env update -f ~/TQ-Experimentation/turboquant-jax/environment.yml --prune; else conda env create -f ~/TQ-Experimentation/turboquant-jax/environment.yml; fi; conda activate turboquant-jax; python -m pip install -U 'jax[cuda12]'; python ~/TQ-Experimentation/turboquant-jax/benchmark_jax.py --device gpu --bits 3 --dim 128 --num-vectors 4096 --num-queries 64; python ~/TQ-Experimentation/turboquant-jax/benchmark_jax_qwen35.py --device gpu --bits 2 3 4 --seq-len 2048 --report-path /mnt/c/Users/zshua/Downloads/TQ-Experimentation/benchmark_qwen35_turboquant_rotorquant.md"

Write-Host "Syncing and benchmarking TurboQuant JAX in WSL Ubuntu..."
wsl.exe -d Ubuntu bash -lc $bashCommand
Write-Host "Done."
