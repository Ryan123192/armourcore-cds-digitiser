from __future__ import annotations

import json
from pathlib import Path


def write_run_report(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
