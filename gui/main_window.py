"""ArmourCore CDS Vectoriser — main window.

Single-window PyQt6 GUI.  Supports both single-file and folder-batch
modes from the same UI:

    * "Browse File..." adds one row to the input table.
    * "Browse Folder..." adds every supported image/PDF in that folder.
    * Each row has Include checkbox, file name, editable insert name,
      template-type dropdown, and live status column.
    * "Start Vectorising" iterates over the included rows, running the
      full pipeline on each and reporting progress.

Outputs per file (under ``data/outputs/end_to_end/<stem>/<timestamp>/``):

    <stem>_rectified.png        -- rectified CDS to scale
    <stem>_vectors.svg          -- vectors + outer border (for scaling)
    <stem>_debug_summary.png    -- 5-panel diagnostic page
"""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QSettings, QSize, QUrl
from PyQt6.QtGui import (
    QPixmap, QFont, QIcon, QImage, QDragEnterEvent, QDropEvent,
)
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from gui.pipeline_runner import (
    TEMPLATES, DEFAULT_TEMPLATE_ID, auto_detect_template,
    RunResult,
)
from gui.worker import make_worker_thread, PipelineWorker


REPO = Path(__file__).parent.parent
APP_TITLE = "ArmourCore CDS Vectoriser"

# Formats we will scan for in folder-batch mode.
SUPPORTED_INPUT_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".tif", ".tiff",
    ".heic", ".heif",
)

# Table column indices
COL_INCLUDE = 0
COL_FILENAME = 1
COL_TEMPLATE = 2
COL_STATUS = 3

# Preview thumbnail size cap.  Prevents Qt from rejecting huge production
# rasters (QImageIOHandler default is a 256 MB allocation limit and our
# rectified PNGs comfortably blow past that).  See ISSUE-010.
PREVIEW_MAX_DIM = 1400

# Default output root if the user hasn't picked one yet.
DEFAULT_OUTPUT_ROOT = REPO / "data" / "outputs" / "end_to_end"


def _load_thumbnail_pixmap(path: Path, max_dim: int = PREVIEW_MAX_DIM) -> QPixmap | None:
    """Load an image as a downscaled QPixmap safe for GUI display.

    Bypasses ``QPixmap(path)`` because PyQt enforces a 256 MB image-byte
    allocation limit by default and our production rectified PNGs can
    exceed it (4094 x 5512 x 3 = ~64 MB raw, ~110 MB decoded with
    alpha).  Loading via cv2 and downscaling to a sensible thumbnail
    keeps memory bounded and never hits the Qt limit.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    qimg = QImage(
        rgb.data, rgb.shape[1], rgb.shape[0],
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    )
    # Copy because the underlying numpy buffer can be reclaimed.
    return QPixmap.fromImage(qimg.copy())


def _open_in_explorer(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    if platform.system() == "Windows":
        if path.is_dir():
            subprocess.Popen(["explorer", str(path)])
        else:
            subprocess.Popen(["explorer", "/select,", str(path)])
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class MainWindow(QMainWindow):
    """The single application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 880)

        # Persistent user settings (last folders, preview prefs, etc.)
        # Stored per-user via QSettings; nothing leaks into the project.
        self._settings = QSettings("ArmourCore", "CDS Vectoriser")

        # Drag-and-drop input support (ISSUE-007).
        self.setAcceptDrops(True)

        # Runtime state
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._last_result: RunResult | None = None
        self._last_rectified_path: Path | None = None

        # Batch queue
        self._batch_queue: list[dict] = []      # list of row-info dicts
        self._batch_idx: int = -1               # current row in flight
        self._batch_results: list[RunResult] = []
        self._stop_requested: bool = False

        self._build_ui()
        self._set_processing(False)

    # ------------------------------------------------------------------
    # Drag-and-drop (ISSUE-007) — accept image / PDF / HEIC files
    # dropped onto the window from File Explorer.
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        added = 0
        skipped = 0
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if not p.exists():
                continue
            if p.is_dir():
                # Treat folder-drop the same as Browse Folder...
                for c in sorted(p.iterdir()):
                    if (c.is_file()
                            and c.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
                            and self._add_input_row(c)):
                        added += 1
                continue
            if p.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
                skipped += 1
                continue
            if self._add_input_row(p):
                added += 1
            else:
                skipped += 1
        if added:
            self.footer.setText(
                f"Drag-and-drop: added {added} file(s)"
                + (f", skipped {skipped}" if skipped else "")
                + "."
            )
        elif skipped:
            self.footer.setText(
                f"Drag-and-drop: skipped {skipped} unsupported file(s)."
            )
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 12)
        root_layout.setSpacing(10)

        # ---- Title strip -----------------------------------------------
        title_label = QLabel(APP_TITLE)
        title_label.setObjectName("AppTitle")
        title_label.setStyleSheet(
            "font-size:22pt; font-weight:700; letter-spacing:2px; "
            "color:#FFFFFF; padding:2px 0;"
        )
        subtitle = QLabel(
            "Convert customer CDS photos and PDFs into closed-contour "
            "vector outlines."
        )
        subtitle.setProperty("muted", True)
        header = QVBoxLayout()
        header.setSpacing(0)
        header.addWidget(title_label)
        header.addWidget(subtitle)
        root_layout.addLayout(header)

        # ---- Main split: controls (left) | output (right) --------------
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        root_layout.addWidget(split, stretch=1)

        # ====== LEFT panel: input selection + table + actions =========
        left_panel = QFrame()
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(14, 14, 14, 14)
        left.setSpacing(8)

        # Add-input row (file or folder)
        left.addWidget(self._heading("Inputs"))
        btn_row = QHBoxLayout()
        self.browse_file_btn = QPushButton("Browse File...")
        self.browse_file_btn.clicked.connect(self._on_browse_file)
        self.browse_folder_btn = QPushButton("Browse Folder...")
        self.browse_folder_btn.clicked.connect(self._on_browse_folder)
        btn_row.addWidget(self.browse_file_btn)
        btn_row.addWidget(self.browse_folder_btn)
        left.addLayout(btn_row)

        clear_row = QHBoxLayout()
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self._on_remove_selected)
        self.clear_btn = QPushButton("Clear List")
        self.clear_btn.clicked.connect(self._on_clear)
        clear_row.addWidget(self.remove_btn)
        clear_row.addWidget(self.clear_btn)
        left.addLayout(clear_row)

        # Inputs table
        self.input_table = QTableWidget(0, 4)
        self.input_table.setHorizontalHeaderLabels(
            ["Include", "File", "Template", "Status"]
        )
        self.input_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.input_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.input_table.setAlternatingRowColors(True)
        hdr = self.input_table.horizontalHeader()
        hdr.setSectionResizeMode(COL_INCLUDE,  QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(COL_FILENAME, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(COL_TEMPLATE, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(COL_STATUS,   QHeaderView.ResizeMode.ResizeToContents)
        self.input_table.verticalHeader().setVisible(False)
        self.input_table.setMinimumHeight(220)
        left.addWidget(self.input_table, stretch=1)

        # Action buttons
        left.addSpacing(4)
        self.start_btn = QPushButton("Start Vectorising")
        self.start_btn.setObjectName("StartButton")
        self.start_btn.setMinimumHeight(42)
        self.start_btn.clicked.connect(self._on_start)
        left.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop and Cancel")
        self.stop_btn.setObjectName("StopButton")
        self.stop_btn.setMinimumHeight(32)
        self.stop_btn.clicked.connect(self._on_stop)
        left.addWidget(self.stop_btn)

        # Debug-output toggle (D) — choose between lean production output
        # (Off: 3 files + run_report.json) and Full (adds per-phase
        # debug copies, per-tool crops, and a complete log.txt).
        left.addSpacing(6)
        left.addWidget(self._heading("Debug output"))
        self.debug_combo = QComboBox()
        self.debug_combo.addItem(
            "Off  —  rectified + vectors + summary + run_report.json",
            userData="off",
        )
        self.debug_combo.addItem(
            "Full  —  + phase 1/2 intermediates, per-tool crops, log.txt",
            userData="full",
        )
        last_debug = self._settings.value("last_debug_mode", "off", type=str)
        for i in range(self.debug_combo.count()):
            if self.debug_combo.itemData(i) == last_debug:
                self.debug_combo.setCurrentIndex(i)
                break
        self.debug_combo.currentIndexChanged.connect(self._on_debug_mode_changed)
        left.addWidget(self.debug_combo)

        # Output-folder selector (ISSUE-008) — user picks where outputs go.
        # Defaults to the last-used folder (ISSUE-009) or the project's
        # default outputs directory on first launch.
        left.addSpacing(6)
        left.addWidget(self._heading("Output folder"))
        out_pick_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        last_output = self._settings.value("last_output_dir", "", type=str)
        initial_output = Path(last_output) if last_output else DEFAULT_OUTPUT_ROOT
        self.output_edit.setText(str(initial_output))
        self.output_edit.setToolTip(str(initial_output))
        out_pick_row.addWidget(self.output_edit, stretch=1)
        self.output_browse_btn = QPushButton("Change...")
        self.output_browse_btn.clicked.connect(self._on_browse_output)
        out_pick_row.addWidget(self.output_browse_btn)
        left.addLayout(out_pick_row)

        # Open-output convenience button (the only output-side button
        # we keep — ISSUE-014 removed "Open SVG" which was redundant
        # since the output folder already contains the SVG).
        self.open_output_btn = QPushButton("Open Output Folder")
        self.open_output_btn.clicked.connect(self._on_open_output)
        left.addWidget(self.open_output_btn)

        split.addWidget(left_panel)

        # ====== RIGHT panel: progress + preview + log =================
        right_panel = QFrame()
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(14, 14, 14, 14)
        right.setSpacing(8)

        right.addWidget(self._heading("Progress"))
        self.stage_label = QLabel("Idle")
        self.stage_label.setStyleSheet("color:#FFFFFF; font-size:11pt;")
        right.addWidget(self.stage_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        right.addWidget(self.progress_bar)

        self.batch_label = QLabel("")
        self.batch_label.setProperty("muted", True)
        right.addWidget(self.batch_label)

        right.addSpacing(4)
        right.addWidget(self._heading("Latest preview"))
        self.preview_label = QLabel()
        self.preview_label.setObjectName("PreviewArea")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(300)
        self.preview_label.setText("(no preview yet)")
        right.addWidget(self.preview_label, stretch=1)

        right.addWidget(self._heading("Log"))
        self.log_view = QTextEdit()
        self.log_view.setObjectName("LogView")
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(150)
        right.addWidget(self.log_view, stretch=1)

        split.addWidget(right_panel)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([460, 940])

        # Footer
        self.footer = QLabel("Ready.")
        self.footer.setProperty("muted", True)
        root_layout.addWidget(self.footer)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _heading(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("heading", True)
        return lbl

    def _append_log(self, line: str) -> None:
        self.log_view.append(line)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_processing(self, busy: bool) -> None:
        self.browse_file_btn.setEnabled(not busy)
        self.browse_folder_btn.setEnabled(not busy)
        self.remove_btn.setEnabled(not busy)
        self.clear_btn.setEnabled(not busy)
        self.input_table.setEnabled(not busy)
        self.start_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)

    # ------------------------------------------------------------------
    # Input table management
    # ------------------------------------------------------------------

    def _existing_paths(self) -> set[str]:
        out: set[str] = set()
        for row in range(self.input_table.rowCount()):
            item = self.input_table.item(row, COL_FILENAME)
            if item is not None:
                out.add(item.data(Qt.ItemDataRole.UserRole))
        return out

    def _add_input_row(self, path: Path) -> bool:
        """Append one row for *path*.  Returns True if added (False if dup)."""
        path = Path(path).resolve()
        if str(path) in self._existing_paths():
            return False
        if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            return False

        row = self.input_table.rowCount()
        self.input_table.insertRow(row)

        # Include checkbox column
        include_widget = QWidget()
        include_layout = QHBoxLayout(include_widget)
        include_layout.setContentsMargins(0, 0, 0, 0)
        include_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cb = QCheckBox()
        cb.setChecked(True)
        include_layout.addWidget(cb)
        self.input_table.setCellWidget(row, COL_INCLUDE, include_widget)
        # Stash widget for later retrieval
        include_widget.setProperty("checkbox", cb)

        # Filename column (display name; full path in UserRole)
        name_item = QTableWidgetItem(path.name)
        name_item.setData(Qt.ItemDataRole.UserRole, str(path))
        name_item.setToolTip(str(path))
        self.input_table.setItem(row, COL_FILENAME, name_item)

        # Template-type dropdown
        tcombo = QComboBox()
        for t in TEMPLATES:
            tcombo.addItem(t.display_name, userData=t.template_id)
        # Auto-detect default
        tid = auto_detect_template(path)
        for i in range(tcombo.count()):
            if tcombo.itemData(i) == tid:
                tcombo.setCurrentIndex(i)
                break
        self.input_table.setCellWidget(row, COL_TEMPLATE, tcombo)

        # Status column
        status_item = QTableWidgetItem("Ready")
        status_item.setForeground(Qt.GlobalColor.lightGray)
        self.input_table.setItem(row, COL_STATUS, status_item)

        return True

    def _row_checkbox(self, row: int) -> QCheckBox | None:
        widget = self.input_table.cellWidget(row, COL_INCLUDE)
        if widget is None:
            return None
        return widget.property("checkbox")

    def _row_path(self, row: int) -> Path | None:
        item = self.input_table.item(row, COL_FILENAME)
        if item is None:
            return None
        return Path(item.data(Qt.ItemDataRole.UserRole))

    def _row_template_id(self, row: int) -> str:
        combo = self.input_table.cellWidget(row, COL_TEMPLATE)
        if combo is None:
            return DEFAULT_TEMPLATE_ID
        return combo.currentData() or DEFAULT_TEMPLATE_ID

    def _set_row_status(self, row: int, text: str) -> None:
        item = self.input_table.item(row, COL_STATUS)
        if item is not None:
            item.setText(text)

    # ------------------------------------------------------------------
    # UI event handlers
    # ------------------------------------------------------------------

    def _last_input_dir(self) -> Path:
        """Return the remembered last-used input directory, falling back
        to the project's raw_images folder, then the repo root."""
        v = self._settings.value("last_input_dir", "", type=str)
        if v and Path(v).exists():
            return Path(v)
        proj_default = REPO / "data" / "inputs" / "raw_images"
        return proj_default if proj_default.exists() else REPO

    def _remember_input_dir(self, path: Path) -> None:
        self._settings.setValue("last_input_dir", str(path))

    def _remember_output_dir(self, path: Path) -> None:
        self._settings.setValue("last_output_dir", str(path))

    def _on_browse_file(self) -> None:
        start_dir = self._last_input_dir()
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select one or more CDS input files",
            str(start_dir),
            "Images / PDFs (*.png *.jpg *.jpeg *.pdf *.tif *.tiff *.bmp *.heic *.heif);;"
            "All files (*.*)",
        )
        added = 0
        for p in paths:
            if self._add_input_row(Path(p)):
                added += 1
        if paths:
            # Remember the folder of the first selected file (ISSUE-009)
            self._remember_input_dir(Path(paths[0]).parent)
        if added:
            self.footer.setText(f"Added {added} file(s).")

    def _on_browse_folder(self) -> None:
        start_dir = self._last_input_dir()
        folder = QFileDialog.getExistingDirectory(
            self, "Select an order folder", str(start_dir)
        )
        if not folder:
            return
        folder_path = Path(folder)
        self._remember_input_dir(folder_path)
        candidates = sorted(
            p for p in folder_path.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
        )
        added = 0
        for p in candidates:
            if self._add_input_row(p):
                added += 1
        if added == 0:
            QMessageBox.information(
                self, "No new files",
                f"No supported input files were found in:\n{folder_path}",
            )
        else:
            self.footer.setText(
                f"Added {added} file(s) from {folder_path.name}."
            )

    def _on_browse_output(self) -> None:
        """User chooses where the output folders should be created (ISSUE-008)."""
        current = self.output_edit.text().strip()
        start_dir = (Path(current) if current and Path(current).exists()
                     else self._last_input_dir())
        folder = QFileDialog.getExistingDirectory(
            self, "Choose output folder",
            str(start_dir),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not folder:
            return
        chosen = Path(folder)
        self.output_edit.setText(str(chosen))
        self.output_edit.setToolTip(str(chosen))
        self._remember_output_dir(chosen)

    def _current_output_root(self) -> Path:
        """Return the user's chosen output root, falling back to the default."""
        v = self.output_edit.text().strip()
        return Path(v) if v else DEFAULT_OUTPUT_ROOT

    def _current_debug_mode(self) -> str:
        """Return 'off' or 'full' from the dropdown."""
        return self.debug_combo.currentData() or "off"

    def _on_debug_mode_changed(self, _idx: int) -> None:
        """Remember the user's last debug-mode choice across launches."""
        self._settings.setValue("last_debug_mode", self._current_debug_mode())

    def _on_remove_selected(self) -> None:
        rows = sorted(
            {idx.row() for idx in self.input_table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            self.input_table.removeRow(row)

    def _on_clear(self) -> None:
        self.input_table.setRowCount(0)

    def _on_start(self) -> None:
        # Build queue from rows the user has checked Include on
        queue: list[dict] = []
        for row in range(self.input_table.rowCount()):
            cb = self._row_checkbox(row)
            if cb is None or not cb.isChecked():
                continue
            path = self._row_path(row)
            if path is None or not path.exists():
                self._set_row_status(row, "missing")
                continue
            queue.append({
                "row": row,
                "path": path,
                "template_id": self._row_template_id(row),
            })

        if not queue:
            QMessageBox.warning(
                self, "Nothing to process",
                "Add at least one input and tick the Include box.",
            )
            return

        # Reset processing state
        self._batch_queue = queue
        self._batch_idx = -1
        self._batch_results = []
        self._stop_requested = False
        self.log_view.clear()
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("(no preview yet)")
        self._set_processing(True)

        # Mark all queued rows as "queued"
        for q in queue:
            self._set_row_status(q["row"], "queued")

        self._run_next_in_batch()

    def _run_next_in_batch(self) -> None:
        """Spin up a worker for the next queued row.  Empty = batch done."""
        self._batch_idx += 1
        if self._stop_requested or self._batch_idx >= len(self._batch_queue):
            self._finalise_batch()
            return

        q = self._batch_queue[self._batch_idx]
        path = q["path"]
        template_id = q["template_id"]

        self.batch_label.setText(
            f"File {self._batch_idx + 1} of {len(self._batch_queue)}:  "
            f"{path.name}"
        )
        self.stage_label.setText("Starting...")
        self.progress_bar.setValue(0)
        self._set_row_status(q["row"], "running")
        self._append_log(
            f"\n=========================================================="
            f"\n[{self._batch_idx + 1}/{len(self._batch_queue)}]  {path.name}"
            f"\n=========================================================="
        )

        self._last_rectified_path = None
        self._thread, self._worker = make_worker_thread(
            path,
            template_id,
            output_root=self._current_output_root(),
            debug_mode=self._current_debug_mode(),
        )
        self._worker.stage_changed.connect(self._on_stage_changed)
        self._worker.log_line.connect(self._append_log)
        self._worker.preview_ready.connect(self._on_preview_ready)
        self._worker.rectified_saved.connect(self._on_rectified_saved)
        self._worker.finished_signal.connect(self._on_one_file_done)
        self._thread.start()

    def _on_one_file_done(self, result: RunResult) -> None:
        """Called after each pipeline run; advance to next file or finish."""
        row = self._batch_queue[self._batch_idx]["row"]
        self._set_row_status(
            row,
            f"OK ({result.n_loops} loops)" if result.success else "FAILED",
        )
        self._batch_results.append(result)
        self._last_result = result
        if self._thread:
            self._thread.wait()
        self._thread = None
        self._worker = None
        # Move on (or stop early if requested)
        self._run_next_in_batch()

    def _finalise_batch(self) -> None:
        self._set_processing(False)
        n = len(self._batch_results)
        n_ok = sum(1 for r in self._batch_results if r.success)
        n_loops_total = sum(r.n_loops for r in self._batch_results if r.success)
        cancelled = self._stop_requested
        if cancelled:
            self.stage_label.setText("Cancelled")
            self.footer.setText(
                f"Cancelled. Completed {n_ok}/{len(self._batch_queue)} "
                f"before stop."
            )
        else:
            self.stage_label.setText("Done")
            self.progress_bar.setValue(100)
            self.footer.setText(
                f"Batch complete: {n_ok}/{n} succeeded, "
                f"{n_loops_total} total loops."
            )
            QMessageBox.information(
                self, "Batch complete",
                f"Processed {n} file(s):\n"
                f"  OK:       {n_ok}\n"
                f"  Failed:   {n - n_ok}\n"
                f"  Loops:    {n_loops_total}",
            )

    def _on_stop(self) -> None:
        self._stop_requested = True
        if self._worker is not None:
            self._worker.request_cancel()
            self.footer.setText("Cancel requested... waiting for safe stop.")
            self.stop_btn.setEnabled(False)

    def _on_stage_changed(self, label: str, pct: int) -> None:
        self.stage_label.setText(label)
        self.progress_bar.setValue(max(0, min(100, int(pct))))

    def _on_preview_ready(self, path: str) -> None:
        """Show a thumbnail preview safely (ISSUE-010).

        Previously called ``QPixmap(path)`` directly which would silently
        reject anything > Qt's 256 MB allocation limit (production
        rectified PNGs blow past that).  We now load via cv2 + downscale
        to bounded memory before handing to Qt.
        """
        pm = _load_thumbnail_pixmap(Path(path))
        if pm is None or pm.isNull():
            return
        scaled = pm.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def _on_rectified_saved(self, path: str) -> None:
        """The pipeline saved the rectified PNG before Phase 2 started
        (ISSUE-006).  Update footer status so the user knows their
        rectified output is safe even if later stages hang or fail."""
        self._last_rectified_path = Path(path)
        self.footer.setText(
            f"Rectified saved: {Path(path).name}  "
            f"(safe even if vectorisation fails)"
        )

    def _on_open_output(self) -> None:
        # Priority order for "open output folder":
        # 1) the per-file output folder of the last completed run
        # 2) the per-stem folder above that, so all timestamps are visible
        # 3) the user-chosen output root
        target: Path | None = None
        if self._last_result and self._last_result.output_dir:
            target = self._last_result.output_dir
        if target is None and self._last_rectified_path is not None:
            target = self._last_rectified_path.parent
        if target is None:
            target = self._current_output_root()
            target.mkdir(parents=True, exist_ok=True)
        _open_in_explorer(target)
