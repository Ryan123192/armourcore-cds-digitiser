"""Simple Tkinter GUI wrapping the vectorise CLI.

Workflow:
    1. Click "Open Scan..." -> pick scan PNG/JPG
    2. Choose paper size (A3 / A2 / A1 / A0)
    3. Choose ink type (auto / pen / pencil)
    4. Click "Vectorise" -> pipeline runs, diagnostic shown
    5. Click "Save SVG" -> writes to chosen location

Designed for office staff: minimal options, clear output.
"""
from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path
from tkinter import Tk, Frame, Button, Label, StringVar, OptionMenu, filedialog, messagebox
from tkinter import ttk

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from vectorise_cli import run_pipeline
import armourcore_cds.phase1.marker_rectify_fast_v4 as _p1

# Paper-size presets (landscape).  Width x Height in mm.
PAPER_SIZES = {
    "A3 (420 x 297)":   (420.0, 297.0),
    "A2 (594 x 420)":   (594.0, 420.0),
    "A1 (841 x 594)":   (841.0, 594.0),
    "A0 (1189 x 841)":  (1189.0, 841.0),
    "Custom 350 x 260": (350.0, 260.0),   # original test template
}


class VectoriseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ArmourCore CDS Vectoriser")
        self.root.geometry("700x520")
        self.input_path: Path | None = None
        self.last_out_dir: Path | None = None

        # Controls
        frame = Frame(root, padx=18, pady=18)
        frame.pack(fill="both", expand=True)

        Label(frame, text="1. Pick scan", font=("Arial", 12, "bold")).pack(anchor="w")
        Button(frame, text="Open scan...", command=self.pick_input,
              height=2, width=20).pack(anchor="w", pady=(0, 12))
        self.input_label = Label(frame, text="No file selected",
                                fg="#666666")
        self.input_label.pack(anchor="w")

        Label(frame, text="\n2. Paper size",
             font=("Arial", 12, "bold")).pack(anchor="w")
        self.paper_var = StringVar(value="A3 (420 x 297)")
        OptionMenu(frame, self.paper_var, *PAPER_SIZES.keys()).pack(anchor="w")

        Label(frame, text="\n3. Ink type",
             font=("Arial", 12, "bold")).pack(anchor="w")
        self.ink_var = StringVar(value="auto-detect")
        OptionMenu(frame, self.ink_var,
                  "auto-detect", "pen", "pencil").pack(anchor="w")

        Label(frame, text="").pack()  # spacer
        self.run_btn = Button(frame, text="VECTORISE",
                             command=self.run_pipeline_threaded,
                             height=2, width=20,
                             bg="#2D7D40", fg="white",
                             font=("Arial", 11, "bold"))
        self.run_btn.pack(anchor="w", pady=8)

        self.status = Label(frame, text="Ready",
                           fg="#444444", font=("Arial", 11))
        self.status.pack(anchor="w", pady=(4, 4))

        self.progress = ttk.Progressbar(frame, mode="indeterminate",
                                       length=300)
        self.progress.pack(anchor="w", pady=4)

        self.save_btn = Button(frame, text="Save SVG to...",
                              command=self.save_svg,
                              state="disabled", height=2, width=20)
        self.save_btn.pack(anchor="w", pady=8)

        self.open_folder_btn = Button(frame, text="Open output folder",
                                     command=self.open_folder,
                                     state="disabled", height=1, width=20)
        self.open_folder_btn.pack(anchor="w")

    def pick_input(self):
        fp = filedialog.askopenfilename(
            title="Pick a scanned CDS sheet",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                      ("All files", "*.*")],
        )
        if fp:
            self.input_path = Path(fp)
            self.input_label.config(text=self.input_path.name)
            self.status.config(text="Ready to vectorise.")

    def run_pipeline_threaded(self):
        if self.input_path is None:
            messagebox.showwarning("No file", "Pick a scan first.")
            return
        # Disable controls, start progress
        self.run_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.open_folder_btn.config(state="disabled")
        self.progress.start(10)
        self.status.config(text="Running pipeline...")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            paper_label = self.paper_var.get()
            paper_w, paper_h = PAPER_SIZES[paper_label]

            # Patch paper-size globals so the pipeline uses them.
            _p1.PAPER_W_MM = paper_w
            _p1.PAPER_H_MM = paper_h

            ink = self.ink_var.get()
            forced_route = None
            if ink == "pen":
                forced_route = "pen"
            elif ink == "pencil":
                forced_route = "pencil"

            out_root = REPO / "data/outputs/InhouseProduction"
            out_dir, verdict = run_pipeline(
                self.input_path, out_root,
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
        self.status.config(text=f"Done.  Verdict: {verdict}  ->  {out_dir.name}")

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
    VectoriseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
