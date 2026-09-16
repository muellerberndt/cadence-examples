"""An AlphaZero checkpoint as an opponent ``(cells, legal) -> column``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import import_azg
from .net import ARGS, NNetWrapper

import_azg()
from connect4.Connect4Game import Connect4Game  # noqa: E402
from MCTS import MCTS  # noqa: E402
from utils import dotdict  # noqa: E402


def canonical(cells: np.ndarray) -> np.ndarray:
    """alpha-zero-general's board (6 by 7, row 0 the top, 1 the side to move, -1 the other)
    from a board as the mover sees it (42 cells row-major from the bottom-left, 1 own, 2 other)."""
    grid = np.asarray(cells).reshape(6, 7)[::-1]
    return np.where(grid == 1, 1, np.where(grid == 2, -1, 0)).astype(np.float64)


class AlphaZeroOpponent:
    """The checkpoint's MCTS policy at temperature 0 with ``sims`` simulations per move; a
    fresh tree per game is not needed (the tree is keyed by board). Ties break by numpy's
    argmax, so the opponent is deterministic given the net; ``seed`` only names it."""

    def __init__(self, checkpoint: str | Path, sims: int = 50, seed: int = 0, name: str | None = None, cuda: bool | None = None) -> None:
        self.game = Connect4Game()
        args = dotdict({**ARGS, "cuda": ARGS.cuda if cuda is None else cuda})
        self.net = NNetWrapper(self.game, args)
        path = Path(checkpoint)
        self.net.load_checkpoint(str(path.parent), path.name)
        self.sims = int(sims)
        self.mcts = MCTS(self.game, self.net, dotdict({"numMCTSSims": self.sims, "cpuct": 1.0}))
        self.name = name or f"alphazero_{path.stem.split('.')[0]}_{self.sims}"
        self.calls = 0

    def __call__(self, cells: np.ndarray, legal: np.ndarray) -> int:
        probs = np.asarray(self.mcts.getActionProb(canonical(cells), temp=0))
        probs = np.where(np.asarray(legal, bool), probs, -1.0)
        self.calls += 1
        return int(np.argmax(probs))
