from __future__ import annotations

import sys
from pathlib import Path

try:
    from .cli import main
except ImportError:
    # Support execution as a direct script path (e.g. python benchbro/__main__.py).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchbro.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
