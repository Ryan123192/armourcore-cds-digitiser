"""Qt worker thread that runs ``pipeline_runner.run_pipeline`` off the UI.

The pipeline emits progress / log / preview callbacks; the worker
forwards them as Qt signals so the main window can update widgets
safely from the GUI thread.
"""
from __future__ import annotations

from pathlib import Path
from threading import Event

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from gui.pipeline_runner import RunCallbacks, RunResult, run_pipeline


class PipelineWorker(QObject):
    """Worker that runs one pipeline invocation and emits Qt signals."""

    # Signals are how the worker tells the UI thread what to do.  Qt
    # marshals them across threads automatically, so the slots run on
    # the UI thread even though the worker is on a background thread.
    stage_changed   = pyqtSignal(str, int)        # (label, percent)
    log_line        = pyqtSignal(str)             # (one line of log)
    preview_ready   = pyqtSignal(str)             # (path to preview PNG)
    finished_signal = pyqtSignal(object)          # (RunResult)

    def __init__(self, input_path: Path, template_id: str) -> None:
        super().__init__()
        self._input_path = Path(input_path)
        self._template_id = template_id
        self._cancel_event = Event()

    def request_cancel(self) -> None:
        """Set the cancellation flag; the worker stops at the next checkpoint."""
        self._cancel_event.set()

    def run(self) -> None:
        """Entry point invoked when the QThread starts."""
        callbacks = RunCallbacks(
            on_stage=lambda label, pct: self.stage_changed.emit(label, pct),
            on_log=lambda line: self.log_line.emit(line),
            on_preview=lambda p: self.preview_ready.emit(str(p)),
            is_cancelled=self._cancel_event.is_set,
        )
        result: RunResult = run_pipeline(
            input_path=self._input_path,
            template_id=self._template_id,
            callbacks=callbacks,
        )
        self.finished_signal.emit(result)


def make_worker_thread(
    input_path: Path,
    template_id: str,
) -> tuple[QThread, PipelineWorker]:
    """Create a started-and-ready worker + thread pair.

    Returns ``(thread, worker)``.  Caller is responsible for connecting
    signals and calling ``thread.start()``.
    """
    thread = QThread()
    worker = PipelineWorker(input_path=input_path, template_id=template_id)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished_signal.connect(thread.quit)
    return thread, worker
