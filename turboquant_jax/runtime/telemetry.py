from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency for constrained environments
    psutil = None


class ResourceTracker:
    """Lightweight wall-clock and memory tracker for inference/benchmark runs."""

    def __init__(self, sample_interval_s: float = 0.05) -> None:
        self.sample_interval_s = sample_interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._process = psutil.Process() if psutil is not None else None
        self._rss_peak_mb = self._rss_now_mb()
        self._gpu_peak_mb = self._gpu_now_mb()
        self.started_at = 0.0

    def _rss_now_mb(self) -> float | None:
        if self._process is None:
            return None
        try:
            return float(self._process.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            return None

    def _gpu_now_mb(self) -> float | None:
        if shutil.which("nvidia-smi") is None:
            return None
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            return None
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        try:
            return float(lines[0])
        except ValueError:
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._rss_now_mb()
            gpu = self._gpu_now_mb()
            if rss is not None:
                self._rss_peak_mb = max(self._rss_peak_mb or 0.0, rss)
            if gpu is not None:
                self._gpu_peak_mb = max(self._gpu_peak_mb or 0.0, gpu)
            time.sleep(self.sample_interval_s)

    def __enter__(self) -> "ResourceTracker":
        self.started_at = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def peak_rss_mb(self) -> float | None:
        return self._rss_peak_mb

    @property
    def peak_gpu_mb(self) -> float | None:
        return self._gpu_peak_mb

    @property
    def elapsed_s(self) -> float:
        if self.started_at <= 0:
            return 0.0
        return time.perf_counter() - self.started_at


def export_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload) and not isinstance(payload, type):
        data = asdict(payload)
    else:
        data = payload
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")


def export_markdown_table(path: str | Path, title: str, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [f"## {title}", ""]
    if not rows:
        lines.append("No rows generated.")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
