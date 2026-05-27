"""Structured run report for self-diagnosis.

Every pipeline run records a ``run_report.json`` next to the output
files.  When something goes wrong on the work computer, you can hand
me the JSON and I can read exactly what happened at every stage
without re-running.

Captured per stage:
  * elapsed wall-clock time
  * process resident-memory (RSS) in MB, sampled via psutil
  * key metrics specific to that stage (image dims, mask sizes,
    component counts, etc.)

Captured per tool:
  * bounding box (x, y, w, h)
  * area, foreground pixel count
  * connected-component counts before/after filtering
  * loop / Bezier-segment counts produced
  * elapsed seconds for that tool's vectorisation
  * a ``slow`` flag if elapsed exceeds a threshold

Captured at top level:
  * input path + size + format + whether HEIC was decoded
  * auto-orient decision
  * template id, design-area mm
  * px-per-mm scale
  * output paths
  * final status (success / partial / cancelled / failed)
  * total wall time and peak RSS over the whole run

Pure data-collection; nothing here changes pipeline behaviour.
"""
from __future__ import annotations

import json
import os
import platform
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:        # pragma: no cover -- psutil should always be present
    _PROC = None


# Threshold above which a per-tool vectorisation is flagged "slow".
SLOW_TOOL_SECONDS = 10.0


def _rss_mb() -> float:
    """Resident memory of this process in MB, or 0.0 if unavailable."""
    if _PROC is None:
        return 0.0
    try:
        return _PROC.memory_info().rss / 1_000_000.0
    except Exception:
        return 0.0


@dataclass
class StageRecord:
    """One stage's timing + memory + free-form metrics."""
    name: str
    started_at: float = field(default_factory=time.time)
    elapsed_s: float = 0.0
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    rss_delta_mb: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "running"        # running / ok / skipped / failed / cancelled
    error: str | None = None

    def finish(self, status: str = "ok", error: str | None = None) -> None:
        self.elapsed_s = round(time.time() - self.started_at, 3)
        self.rss_after_mb = _rss_mb()
        self.rss_delta_mb = round(self.rss_after_mb - self.rss_before_mb, 1)
        self.rss_before_mb = round(self.rss_before_mb, 1)
        self.rss_after_mb  = round(self.rss_after_mb,  1)
        self.status = status
        if error:
            self.error = error


@dataclass
class ToolRecord:
    """Per-tool vectorisation stats."""
    index: int
    bbox: tuple[int, int, int, int]
    area_px: int = 0
    fg_px: int = 0
    components_in_crop: int = 0
    contours_before: int = 0
    contours_after: int = 0
    loops: int = 0
    bezier_segments: int = 0
    elapsed_s: float = 0.0
    slow: bool = False
    status: str = "ok"
    error: str | None = None


@dataclass
class RunReport:
    """Top-level run report serialised as run_report.json."""
    app_version: str = "0.1.0"
    schema_version: int = 1
    machine: str = field(default_factory=platform.node)
    os: str = field(default_factory=platform.platform)
    python: str = field(default_factory=platform.python_version)
    started_at_iso: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S")
    )

    # Top-level input metadata.
    input_path: str = ""
    input_format: str = ""
    input_size_mb: float = 0.0
    input_dimensions_px: tuple[int, int] | None = None
    heic_decoded: bool = False
    auto_oriented: bool = False
    rotated_deg: int = 0

    # Template / scale.
    template_id: str = ""
    design_area_mm: tuple[float, float] | None = None
    rectified_dimensions_px: tuple[int, int] | None = None
    px_per_mm: float = 0.0

    # Stage records keyed by stage name (preserves insertion order).
    stages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)

    # Headline result fields.
    n_tools: int = 0
    n_loops: int = 0
    n_beziers: int = 0
    n_slow_tools: int = 0
    total_elapsed_s: float = 0.0
    peak_rss_mb: float = 0.0
    final_status: str = "running"   # success | partial | cancelled | failed | running
    error: str | None = None

    # Output paths produced.
    outputs: dict[str, str] = field(default_factory=dict)

    # ----- Helpers used during a live run -----------------------------------

    def begin_stage(self, name: str) -> StageRecord:
        s = StageRecord(name=name)
        s.rss_before_mb = _rss_mb()
        return s

    def end_stage(self, record: StageRecord,
                  status: str = "ok",
                  error: str | None = None,
                  **metrics: Any) -> None:
        record.metrics.update(metrics)
        record.finish(status=status, error=error)
        self.stages.append(asdict(record))
        self.peak_rss_mb = max(self.peak_rss_mb, record.rss_after_mb)

    def add_tool(self, tool: ToolRecord) -> None:
        if tool.elapsed_s >= SLOW_TOOL_SECONDS:
            tool.slow = True
            self.n_slow_tools += 1
        self.tools.append(asdict(tool))

    # ----- Finalisation -----------------------------------------------------

    def write(self, output_path: Path) -> None:
        """Serialise to JSON next to the output files."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2, default=str)
        except Exception:
            # Diagnostics must never break the run itself.
            traceback.print_exc()


def safe_format_exception() -> str:
    """Returns the active exception's formatted traceback as a string."""
    return "".join(traceback.format_exc()).strip()
