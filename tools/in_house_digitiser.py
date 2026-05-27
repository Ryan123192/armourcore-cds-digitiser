"""In-House Digitiser - GUI for the in-house CDS tracing workflow.

(Distinct from the broader 'CDS Digitiser' for arbitrary photos -
this one is purpose-built for the controlled scan + printed-template
workflow.)

Workflow:
    1. Open scan or PDF (PDF is auto-converted to PNG at 300dpi)
    2. Preview shows the loaded image
    3. Use Rotate/Flip buttons to orient correctly (markers should be
       in their proper corners after rotation)
    4. Pick paper size + ink type
    5. Click VECTORISE
    6. Save SVG or open output folder
"""
from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path
from tkinter import (Tk, Frame, Button, Label, StringVar, Canvas,
                    OptionMenu, filedialog, messagebox, PhotoImage)
from tkinter import ttk

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
import numpy as np
from PIL import Image, ImageTk

from vectorise_cli import run_pipeline
import armourcore_cds.phase1.marker_rectify_fast_v4 as _p1

PAPER_SIZES = {
    "A3 (420 x 297)":   (420.0, 297.0),
    "A2 (594 x 420)":   (594.0, 420.0),
    "A1 (841 x 594)":   (841.0, 594.0),
    "A0 (1189 x 841)":  (1189.0, 841.0),
    "Custom 350 x 260": (350.0, 260.0),
}

PREVIEW_W, PREVIEW_H = 520, 380


class InHouseDigitiser:
    def __init__(self, root):
        self.root = root
        self.root.title("In-House Digitiser")
        self.root.geometry("1100x700")
        self.input_path: Path | None = None
        self.current_image_bgr: np.ndarray | None = None  # working image
        self.original_image_bgr: np.ndarray | None = None
        self.last_out_dir: Path | None = None
        self._photo = None  # holds reference so Tk doesn't GC
        # default output root, overridable in GUI
        self.output_root: Path = REPO / "data/outputs/InhouseProduction"

        # Left column: controls
        left = Frame(root, padx=18, pady=18, width=480)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        Label(left, text="In-House Digitiser",
             font=("Arial", 16, "bold")).pack(anchor="w")
        Label(left, text="(in-house controlled scan workflow)",
             fg="#666666").pack(anchor="w", pady=(0, 10))

        Label(left, text="1. Pick scan / PDF",
             font=("Arial", 11, "bold")).pack(anchor="w")
        Button(left, text="Open scan or PDF...",
              command=self.pick_input,
              height=2, width=24).pack(anchor="w", pady=(0, 6))
        self.input_label = Label(left, text="No file selected",
                                fg="#666666", wraplength=440, justify="left")
        self.input_label.pack(anchor="w")

        Label(left, text="\n2. Orient (markers should be in corners)",
             font=("Arial", 11, "bold")).pack(anchor="w")
        rot_frame = Frame(left)
        rot_frame.pack(anchor="w", pady=4)
        Button(rot_frame, text="↻ 90 CW",
              command=lambda: self.transform("cw"),
              width=10).pack(side="left", padx=2)
        Button(rot_frame, text="↺ 90 CCW",
              command=lambda: self.transform("ccw"),
              width=10).pack(side="left", padx=2)
        Button(rot_frame, text="180°",
              command=lambda: self.transform("180"),
              width=8).pack(side="left", padx=2)
        flip_frame = Frame(left)
        flip_frame.pack(anchor="w", pady=2)
        Button(flip_frame, text="Flip H",
              command=lambda: self.transform("flipH"),
              width=10).pack(side="left", padx=2)
        Button(flip_frame, text="Flip V",
              command=lambda: self.transform("flipV"),
              width=10).pack(side="left", padx=2)
        Button(flip_frame, text="Reset",
              command=lambda: self.transform("reset"),
              width=8).pack(side="left", padx=2)

        Label(left, text="\n3. Paper size",
             font=("Arial", 11, "bold")).pack(anchor="w")
        self.paper_var = StringVar(value="A3 (420 x 297)")
        OptionMenu(left, self.paper_var, *PAPER_SIZES.keys()).pack(anchor="w")

        Label(left, text="\n4. Ink / mode",
             font=("Arial", 11, "bold")).pack(anchor="w")
        self.ink_var = StringVar(value="auto-detect")
        OptionMenu(left, self.ink_var,
                  "auto-detect",
                  "pen (clean tracings)",
                  "pen (with notes/scribbles)",
                  "pencil").pack(anchor="w")

        Label(left, text="\n5. Output folder",
             font=("Arial", 11, "bold")).pack(anchor="w")
        out_pick_frame = Frame(left)
        out_pick_frame.pack(anchor="w")
        Button(out_pick_frame, text="Choose...",
              command=self.pick_output_folder, width=10).pack(side="left")
        self.output_label = Label(out_pick_frame, text=str(self.output_root),
                                 fg="#666666", wraplength=320,
                                 justify="left", anchor="w")
        self.output_label.pack(side="left", padx=(8, 0))

        self.run_btn = Button(left, text="VECTORISE",
                             command=self.run_pipeline_threaded,
                             height=2, width=22,
                             bg="#2D7D40", fg="white",
                             font=("Arial", 11, "bold"))
        self.run_btn.pack(anchor="w", pady=(16, 6))

        self.status = Label(left, text="Ready - open a scan to start.",
                           fg="#444444", font=("Arial", 10),
                           wraplength=440, justify="left")
        self.status.pack(anchor="w", pady=4)

        self.progress = ttk.Progressbar(left, mode="indeterminate", length=300)
        self.progress.pack(anchor="w", pady=4)

        out_frame = Frame(left)
        out_frame.pack(anchor="w", pady=8)
        self.save_btn = Button(out_frame, text="Save SVG...",
                              command=self.save_svg,
                              state="disabled", width=14)
        self.save_btn.pack(side="left", padx=2)
        self.open_folder_btn = Button(out_frame, text="Open folder",
                                     command=self.open_folder,
                                     state="disabled", width=14)
        self.open_folder_btn.pack(side="left", padx=2)

        # Right column: preview canvas
        right = Frame(root, padx=18, pady=18)
        right.pack(side="left", fill="both", expand=True)
        Label(right, text="Preview",
             font=("Arial", 11, "bold")).pack(anchor="w")
        self.canvas = Canvas(right, width=PREVIEW_W, height=PREVIEW_H,
                            bg="#DDDDDD", highlightthickness=1,
                            highlightbackground="#999999")
        self.canvas.pack(anchor="w", pady=4)
        self.size_label = Label(right, text="", fg="#666666")
        self.size_label.pack(anchor="w")

    def pick_output_folder(self):
        d = filedialog.askdirectory(
            title="Choose output folder",
            initialdir=str(self.output_root),
        )
        if d:
            self.output_root = Path(d)
            self.output_label.config(text=str(self.output_root))
            self.status.config(text=f"Output -> {d}")

    # ------------------------- file handling -------------------------

    def pick_input(self):
        fp = filedialog.askopenfilename(
            title="Pick a scanned CDS sheet or PDF",
            filetypes=[("All supported",
                       "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.pdf"),
                      ("Image files",
                       "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                      ("PDF files", "*.pdf"),
                      ("All files", "*.*")],
        )
        if not fp:
            return
        path = Path(fp)
        # PDF: convert to PNG via PyMuPDF
        if path.suffix.lower() == ".pdf":
            try:
                import fitz
                self.status.config(text=f"Converting PDF: {path.name}...")
                self.root.update()
                doc = fitz.open(str(path))
                pix = doc[0].get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                png_path = REPO / "data/outputs/InhouseProduction/_gui_tmp.png"
                png_path.parent.mkdir(parents=True, exist_ok=True)
                pix.save(str(png_path))
                img = cv2.imread(str(png_path))
            except Exception as exc:
                messagebox.showerror("PDF conversion failed", str(exc))
                return
        else:
            img = cv2.imread(str(path))
            if img is None:
                messagebox.showerror("Cannot read file",
                                    f"Could not load {path}")
                return

        self.input_path = path
        self.original_image_bgr = img
        self.current_image_bgr = img.copy()
        self.input_label.config(text=path.name)
        self._refresh_preview()
        self.status.config(text="Loaded - check orientation, then Vectorise.")

    def transform(self, kind: str):
        if self.current_image_bgr is None:
            return
        if kind == "cw":
            self.current_image_bgr = cv2.rotate(
                self.current_image_bgr, cv2.ROTATE_90_CLOCKWISE)
        elif kind == "ccw":
            self.current_image_bgr = cv2.rotate(
                self.current_image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif kind == "180":
            self.current_image_bgr = cv2.rotate(
                self.current_image_bgr, cv2.ROTATE_180)
        elif kind == "flipH":
            self.current_image_bgr = cv2.flip(self.current_image_bgr, 1)
        elif kind == "flipV":
            self.current_image_bgr = cv2.flip(self.current_image_bgr, 0)
        elif kind == "reset":
            self.current_image_bgr = self.original_image_bgr.copy()
        self._refresh_preview()

    def _refresh_preview(self):
        if self.current_image_bgr is None:
            return
        img = self.current_image_bgr
        h, w = img.shape[:2]
        s = min(PREVIEW_W / w, PREVIEW_H / h)
        nw, nh = int(w * s), int(h * s)
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        # centre image
        self.canvas.create_image(PREVIEW_W // 2, PREVIEW_H // 2,
                                anchor="center", image=self._photo)
        self.size_label.config(text=f"{w} x {h}  (showing {nw} x {nh})")

    # ------------------------- pipeline run --------------------------

    def run_pipeline_threaded(self):
        if self.current_image_bgr is None:
            messagebox.showwarning("No file", "Open a scan first.")
            return
        self.run_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.open_folder_btn.config(state="disabled")
        self.progress.start(10)
        self.status.config(text="Running pipeline...")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            # Save the user-oriented image to a temp file
            tmp_dir = REPO / "data/outputs/InhouseProduction/_gui_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_png = tmp_dir / f"{self.input_path.stem}.png"
            cv2.imwrite(str(tmp_png), self.current_image_bgr)

            # Configure paper size globals
            paper_label = self.paper_var.get()
            paper_w, paper_h = PAPER_SIZES[paper_label]
            _p1.PAPER_W_MM = paper_w
            _p1.PAPER_H_MM = paper_h

            ink = self.ink_var.get()
            if ink.startswith("pen (clean"):
                forced_route = "pen"
            elif ink.startswith("pen (with"):
                forced_route = "pen_dense"
            elif ink == "pencil":
                forced_route = "pencil"
            else:
                forced_route = None

            out_dir, verdict = run_pipeline(
                tmp_png, self.output_root,
                forced_route=forced_route,
                rectifier="scan",
                ts_prefix="gui",
            )
            self.last_out_dir = out_dir
            self.root.after(0, lambda: self._done_ok(out_dir, verdict))
        except Exception as exc:
            err = str(exc)
            self.root.after(0, lambda: self._done_err(err))

    def _done_ok(self, out_dir, verdict):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.open_folder_btn.config(state="normal")
        self.status.config(
            text=f"Done.  Verdict: {verdict}\nFolder: {out_dir.name}")
        # Show diagnostic in preview pane
        diag = out_dir / "diagnostic.png"
        if diag.exists():
            img = cv2.imread(str(diag))
            self.current_image_bgr = img   # so preview shows the diagnostic
            self._refresh_preview()

    def _done_err(self, err):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.config(text="ERROR - see popup")
        messagebox.showerror("Pipeline failed", err)

    def save_svg(self):
        if self.last_out_dir is None:
            return
        src = self.last_out_dir / "vectors.svg"
        if not src.exists():
            messagebox.showerror("Missing SVG", f"{src}\nnot found")
            return
        dst = filedialog.asksaveasfilename(
            defaultextension=".svg",
            initialfile=f"{self.input_path.stem}.svg",
            filetypes=[("SVG files", "*.svg")],
        )
        if dst:
            shutil.copy(src, dst)
            self.status.config(text=f"Saved -> {Path(dst).name}")

    def open_folder(self):
        if self.last_out_dir is None:
            return
        import os
        os.startfile(str(self.last_out_dir))


def main():
    root = Tk()
    InHouseDigitiser(root)
    root.mainloop()


if __name__ == "__main__":
    main()
