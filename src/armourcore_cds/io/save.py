from __future__ import annotations

from pathlib import Path


def ensure_run_dir(base_dir: Path, run_name: str) -> Path:
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
