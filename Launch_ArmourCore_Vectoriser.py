"""Root launcher for the ArmourCore CDS Vectoriser GUI.

Cross-platform Python entry point.  Windows users typically use the
double-click ``Launch_ArmourCore_Vectoriser.bat`` which invokes this
file with the system Python.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from gui.app import main


if __name__ == "__main__":
    sys.exit(main())
