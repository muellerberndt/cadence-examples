"""alpha-zero-general's Connect Four in PyTorch: the net (their Othello architecture, as their
Keras Connect4 net copies it), its wrapper, a trainer around their Coach and an opponent."""

from __future__ import annotations

import sys
from pathlib import Path

AZG = Path(__file__).resolve().parents[2] / "external" / "azg"


def import_azg() -> None:
    """Put alpha-zero-general on the path (its modules import each other by bare name)."""
    if not AZG.exists():
        raise FileNotFoundError(f"{AZG}: run connect_four/bench/setup_external.sh")
    if str(AZG) not in sys.path:
        sys.path.insert(0, str(AZG))
