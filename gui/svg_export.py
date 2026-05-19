"""SVG export that matches the ArmourCore production workflow.

Three layered groups in the output:

    1. ``reference_image``  -- the rectified CDS PNG, embedded as base64
       (so the SVG is self-contained and can be moved between machines
       without dragging the PNG with it).  Slot 1: just a visual aid
       in Affinity Publisher.
    2. ``outer_border``     -- a thin black rectangle at the design-area
       bounds.  This is the workflow hook: "Select All -> scale group
       to 500 x 600 mm" inside Affinity lands every vector at the
       correct real-world size in one click.
    3. ``tool_vectors``     -- one closed cubic-Bezier path per detected
       tool loop, drawn in red.

SVG ``width`` / ``height`` are in **millimetres** (computed from the
template's design-area definition) so any vector application that
respects unit hints (Affinity, Illustrator, Inkscape) opens the file
at the correct real-world size from the start.  The viewBox is in
pixels (matching the rectified image's pixel dimensions) so vector
coordinates stay in pixel space and don't need conversion.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable

import numpy as np


def export_vector_svg(
    output_path: Path,
    rectified_png_path: Path | None,
    segments_per_loop: list[list],
    image_size: tuple[int, int],
    design_area_mm: tuple[float, float],
    *,
    embed_image: bool = True,
    border_stroke_px: float = 2.0,
    vector_stroke_px: float = 2.0,
) -> None:
    """Write a self-contained SVG with reference image + outer border + vectors.

    Parameters
    ----------
    output_path:
        File path to write the .svg to.
    rectified_png_path:
        Path to the rectified CDS PNG that will sit underneath the
        vectors.  Pass ``None`` to omit the image layer entirely
        (lean vector-only output).
    segments_per_loop:
        List of loops.  Each loop is a list of ``(p1, cp1, cp2, p2)``
        tuples representing one cubic Bezier segment.  Coordinates are
        in pixel-space matching ``image_size``.
    image_size:
        ``(height, width)`` of the rectified raster in pixels.
    design_area_mm:
        ``(width_mm, height_mm)`` of the CDS design area, used to set
        the SVG's physical-size attributes.
    embed_image:
        If True, base64-embed the PNG bytes into the SVG.  If False,
        emit a relative ``xlink:href`` link to ``rectified_png_path``
        as a sibling file.
    """
    H, W = image_size
    width_mm, height_mm = design_area_mm

    parts: list[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width_mm}mm" height="{height_mm}mm" '
        f'viewBox="0 0 {W} {H}" '
        f'preserveAspectRatio="xMidYMid meet">'
    )

    # --- Layer 1: reference image -----------------------------------------
    if rectified_png_path is not None:
        if embed_image:
            with open(rectified_png_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("ascii")
            href = f"data:image/png;base64,{img_b64}"
        else:
            href = rectified_png_path.name
        parts.append('  <g id="reference_image" inkscape:label="Reference Image">')
        parts.append(
            f'    <image xlink:href="{href}" '
            f'x="0" y="0" width="{W}" height="{H}" />'
        )
        parts.append("  </g>")

    # --- Layer 2: outer border --------------------------------------------
    # A thin rectangle at the image bounds.  Selecting it together with
    # the tool vectors and scaling the whole group to design_area_mm
    # snaps every contour to its true real-world size.
    parts.append(
        '  <g id="outer_border" inkscape:label="Outer Border" '
        f'stroke="#000000" stroke-width="{border_stroke_px}" fill="none">'
    )
    parts.append(
        f'    <rect x="0" y="0" width="{W}" height="{H}" />'
    )
    parts.append("  </g>")

    # --- Layer 3: tool vectors --------------------------------------------
    parts.append(
        '  <g id="tool_vectors" inkscape:label="Tool Vectors" '
        f'stroke="#D02828" stroke-width="{vector_stroke_px}" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round">'
    )
    for segments in segments_per_loop:
        if not segments:
            continue
        d = _path_d_from_segments(segments)
        if d:
            parts.append(f'    <path d="{d}" />')
    parts.append("  </g>")

    parts.append("</svg>")
    parts.append("")

    output_path.write_text("\n".join(parts), encoding="utf-8")


def _path_d_from_segments(segments: Iterable) -> str:
    """Convert a list of cubic Bezier segments into an SVG path ``d``."""
    pieces: list[str] = []
    started = False
    for seg in segments:
        p1, cp1, cp2, p2 = seg
        p1 = np.asarray(p1, dtype=float)
        cp1 = np.asarray(cp1, dtype=float)
        cp2 = np.asarray(cp2, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        if not started:
            pieces.append(f"M {p1[0]:.2f} {p1[1]:.2f}")
            started = True
        pieces.append(
            f"C {cp1[0]:.2f} {cp1[1]:.2f} "
            f"{cp2[0]:.2f} {cp2[1]:.2f} "
            f"{p2[0]:.2f} {p2[1]:.2f}"
        )
    if started:
        pieces.append("Z")
    return " ".join(pieces)
