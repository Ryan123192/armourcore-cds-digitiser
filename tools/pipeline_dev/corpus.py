"""Test-corpus discovery + shared dev helpers.

Single source of truth for the BLUE_* test files we run every step's
dev experiments against.  Anything used by more than one step's
script lives here so per-step scripts stay focused.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


REPO = Path(__file__).parent.parent.parent
RAW_IMAGES_DIR = REPO / "data" / "inputs" / "raw_images"
PIPELINE_DEV_ROOT = REPO / "data" / "outputs" / "pipeline_dev"

# Filename pattern for the test corpus
_CASE_RE = re.compile(
    r"^BLUE_(?P<medium>PEN|PENCIL)_(?P<paper>FLAT|CREASED)_(?P<idx>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CorpusFile:
    """One discovered test file with its parsed metadata."""
    path: Path
    medium: str          # "PEN" / "PENCIL"
    paper: str           # "FLAT" / "CREASED"
    index: int           # 01 -> 1
    stem: str            # original stem, e.g. "BLUE_PEN_FLAT_01"

    @property
    def label(self) -> str:
        """Compact label for grid summaries."""
        return f"{self.medium[:2]}-{self.paper[:4]}-{self.index:02d}"

    @property
    def difficulty(self) -> int:
        """Rough difficulty rank — used only for stable sort order."""
        return (
            (0 if self.medium == "PEN" else 1) * 100
            + (0 if self.paper == "FLAT" else 50)
            + self.index
        )


def discover_corpus(
    medium: str | None = None,
    paper: str | None = None,
) -> list[CorpusFile]:
    """Return every test file matching the corpus naming convention.

    Optional filters by medium ("PEN" / "PENCIL") and paper
    ("FLAT" / "CREASED").  Sorted by difficulty (pen-flat first,
    pencil-creased last).
    """
    out: list[CorpusFile] = []
    if not RAW_IMAGES_DIR.exists():
        return out
    for p in RAW_IMAGES_DIR.iterdir():
        if not p.is_file():
            continue
        m = _CASE_RE.match(p.stem)
        if not m:
            continue
        case = CorpusFile(
            path=p,
            medium=m.group("medium").upper(),
            paper=m.group("paper").upper(),
            index=int(m.group("idx")),
            stem=p.stem,
        )
        if medium is not None and case.medium != medium.upper():
            continue
        if paper is not None and case.paper != paper.upper():
            continue
        out.append(case)
    out.sort(key=lambda c: c.difficulty)
    return out


def make_run_dir(step_name: str, label: str) -> Path:
    """Create a fresh timestamped output folder for this dev run.

    Returns the path; the caller writes per-case subfolders into it.
    Layout::

        data/outputs/pipeline_dev/<step_name>/<YYYYMMDD-HHMMSS>_<label>/
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    run_dir = PIPELINE_DEV_ROOT / step_name / f"{stamp}_{safe_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Visualisation helpers shared across steps
# ---------------------------------------------------------------------------

def label_image(
    img: np.ndarray,
    text: str,
    *,
    height_px: int = 40,
    scale: float = 0.8,
    bg: tuple[int, int, int] = (15, 17, 22),
    fg: tuple[int, int, int] = (230, 232, 235),
) -> np.ndarray:
    """Return a copy of *img* with a dark label strip painted across the top."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], height_px), bg, -1)
    cv2.putText(
        out, text, (10, height_px - 12),
        cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 2, cv2.LINE_AA,
    )
    return out


def grid_montage(
    tiles: Iterable[np.ndarray],
    *,
    cols: int = 4,
    tile_max_dim: int = 500,
    bg: tuple[int, int, int] = (10, 12, 16),
) -> np.ndarray:
    """Tile a list of images into a single PNG-friendly grid montage.

    Each tile is resized so its long edge is *tile_max_dim* px, then
    padded so every tile in the grid is the same size for a clean
    grid look.
    """
    items = list(tiles)
    if not items:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Step 1: resize each to fit tile_max_dim long edge
    scaled = []
    for t in items:
        h, w = t.shape[:2]
        s = tile_max_dim / max(h, w)
        if s != 1.0:
            t = cv2.resize(t, (int(round(w * s)), int(round(h * s))),
                           interpolation=cv2.INTER_AREA)
        scaled.append(t)

    # Step 2: pad every tile to the largest size in the batch
    max_h = max(t.shape[0] for t in scaled)
    max_w = max(t.shape[1] for t in scaled)
    padded = []
    for t in scaled:
        h, w = t.shape[:2]
        canvas = np.full((max_h, max_w, 3), bg, dtype=np.uint8)
        canvas[:h, :w] = t
        padded.append(canvas)

    rows = (len(padded) + cols - 1) // cols
    montage = np.full((rows * max_h, cols * max_w, 3), bg, dtype=np.uint8)
    for i, tile in enumerate(padded):
        r, c = divmod(i, cols)
        montage[r * max_h:(r + 1) * max_h, c * max_w:(c + 1) * max_w] = tile
    return montage
