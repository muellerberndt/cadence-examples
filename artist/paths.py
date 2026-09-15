"""Put the examples repository (``agent`` and ``arm``) on the import path."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def examples_root() -> Path:
    """The cadence-examples checkout: ``CADENCE_EXAMPLES`` or the sibling of cadence-paper."""
    named = os.environ.get("CADENCE_EXAMPLES")
    candidates = [Path(named)] if named else []
    candidates.append(HERE.parent)  # this stage lives inside the examples repository
    for path in candidates:
        if (path / "agent" / "brain.py").is_file() and (path / "arm" / "env.py").is_file():
            return path.resolve()
    raise ModuleNotFoundError("cadence-examples not found; set CADENCE_EXAMPLES to the accepted checkout")


def ensure() -> Path:
    root = examples_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
