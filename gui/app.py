"""ArmourCore CDS Vectoriser — Qt application entry point.

Run via the project-root launcher::

    python Launch_ArmourCore_Vectoriser.py

or on Windows by double-clicking ``Launch_ArmourCore_Vectoriser.bat``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'gui.*' and 'armourcore_cds.*' imports
# resolve regardless of how the launcher invokes us.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def _load_stylesheet() -> str:
    qss_path = Path(__file__).parent / "armourcore.qss"
    if qss_path.exists():
        try:
            return qss_path.read_text(encoding="utf-8")
        except OSError:
            pass
    return ""


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ArmourCore CDS Vectoriser")
    app.setOrganizationName("ArmourCore")

    # Apply the ArmourCore-branded dark theme.  Falling back to default
    # styling if the QSS file is missing keeps the GUI launchable in a
    # stripped-down install.
    qss = _load_stylesheet()
    if qss:
        app.setStyleSheet(qss)

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
