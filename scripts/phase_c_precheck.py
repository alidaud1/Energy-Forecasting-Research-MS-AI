from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script from repo root on Windows/Unix.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_c_precheck import main


if __name__ == "__main__":
    main()
