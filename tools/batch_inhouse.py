"""Batch process all 6 in-house test templates and build a comparison page.

V1 (major+minor grid lines), V2 (minor only), V3 (dots) x Pen + Pencil.

Output:
    data/outputs/InhouseProduction/<timestamp>_BATCH/
        Vx_Pen/
            diagnostic.png
            vectors.svg
            ...
        Vx_Pencil/
            ...
        SUMMARY.png       <-- 2x3 grid: rows = ink type, cols = template version
                              each cell shows the 6-panel diagnostic in miniature
        summary.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
import fitz   # PyMuPDF
import numpy as np

# Reuse the pipeline from vectorise_cli
from vectorise_cli import run_pipeline


VERSIONS = [
    ("V1", "InHouseTestPenV1", "major + minor grid"),
    ("V2", "InHouseTestPenV2", "minor only"),
    ("V3", "InHouseTestPenV3", "dots only"),
]

PENCIL_VERSIONS = [
    ("V1", "InHouseTestPencilV1", "major + minor grid"),
    ("V2", "InHouseTestPencilV2", "minor only"),
    ("V3", "InHouseTestPencilV3", "dots only"),
]


def pdf_to_png(pdf_path: Path, out_path: Path, dpi: int = 200) -> Path:
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    pix.save(str(out_path))
    return out_path


def _fit(img, max_w, max_h):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_root = REPO / "data/outputs/InhouseProduction" / f"{ts}_BATCH"
    batch_root.mkdir(parents=True, exist_ok=True)
    pdf_dir = REPO / "data/inputs/raw_pdfs"
    png_dir = batch_root / "_pngs"

    results = []
    for ink_type, version_list in [("Pen", VERSIONS),
                                   ("Pencil", PENCIL_VERSIONS)]:
        for vname, stem, desc in version_list:
            pdf_path = pdf_dir / f"{stem}.pdf"
            if not pdf_path.exists():
                print(f"!! missing {pdf_path}")
                continue
            png_path = pdf_to_png(pdf_path, png_dir / f"{stem}.png", dpi=200)
            tag = f"{vname}_{ink_type}"
            print(f"\n>>> {tag} ({desc})")
            out_root = batch_root
            out_dir, verdict = run_pipeline(
                png_path, out_root,
                forced_route=("pencil" if ink_type == "Pencil" else None),
                rectifier="scan",
                ts_prefix=tag,
            )
            results.append({
                "ink": ink_type, "version": vname, "desc": desc,
                "out_folder": str(out_dir),
                "verdict": verdict,
            })

    # Build SUMMARY: 2 rows (pen, pencil) x 3 cols (V1, V2, V3)
    # Each cell shows the diagnostic.png from that run (downsampled).
    cell_w, cell_h = 1400, 1050
    header_h = 60
    grid = np.full((header_h + 2 * cell_h, 3 * cell_w, 3), 250, dtype=np.uint8)
    cv2.putText(grid,
               f"InhouseProduction batch  {ts}    rows = ink type, cols = template version",
               (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
               (30, 30, 30), 2, cv2.LINE_AA)

    for ri, ink in enumerate(["Pen", "Pencil"]):
        for ci, (vname, _, _) in enumerate(VERSIONS):
            r = next((r for r in results
                     if r["ink"] == ink and r["version"] == vname), None)
            if r is None:
                continue
            diag = cv2.imread(str(Path(r["out_folder"]) / "diagnostic.png"))
            if diag is None:
                continue
            fit = _fit(diag, cell_w - 20, cell_h - 80)
            fh, fw = fit.shape[:2]
            y0 = header_h + ri * cell_h
            # Cell header
            cv2.rectangle(grid, (ci * cell_w, y0),
                         ((ci + 1) * cell_w, y0 + 50),
                         (35, 35, 35), -1)
            label = f"{ink}  {vname}    [{r['verdict']}]"
            cv2.putText(grid, label, (ci * cell_w + 14, y0 + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (255, 255, 255), 2, cv2.LINE_AA)
            # Place the diagnostic
            x0 = ci * cell_w + (cell_w - fw) // 2
            yp = y0 + 60 + (cell_h - 70 - fh) // 2
            grid[yp:yp + fh, x0:x0 + fw] = fit
    cv2.imwrite(str(batch_root / "SUMMARY.png"), grid)
    (batch_root / "summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n\n=== BATCH SUMMARY ===")
    print(f"{'ink':<8s} {'version':<6s} {'desc':<24s} {'verdict':<12s}")
    for r in results:
        print(f"{r['ink']:<8s} {r['version']:<6s} {r['desc']:<24s} "
              f"{r['verdict']:<12s}")
    print(f"\nFolder: {batch_root}")
    print(f"Grid:   {batch_root / 'SUMMARY.png'}")


if __name__ == "__main__":
    main()
