"""An AlphaZero checkpoint as an opponent ``(cells, legal) -> column``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import import_azg
from .net import ARGS, NNetWrapper

import_azg()
from connect4.Connect4Game import Connect4Game
from MCTS import MCTS
from utils import dotdict


def canonical(cells: np.ndarray) -> np.ndarray:
    """alpha-zero-general's board (6 by 7, row 0 the top, 1 the side to move, -1 the other)
    from a board as the mover sees it (42 cells row-major from the bottom-left, 1 own, 2 other)."""
    grid = np.asarray(cells).reshape(6, 7)[::-1]
    return np.where(grid == 1, 1, np.where(grid == 2, -1, 0)).astype(np.float64)


class AlphaZeroOpponent:
    """The checkpoint's MCTS policy with ``sims`` simulations per move: the visit distribution
    is sampled at temperature 1 while the board holds fewer than ``temp_plies`` stones (the
    opening variety of alpha-zero-general's own self-play, which ``tempThreshold`` sets to 15)
    and its mode is played after that. The tree is rebuilt at the start of every game (a board
    with at most one stone), as their Coach's arena does. ``seed`` drives the sampling, so the
    two games of a pair see the same openings."""

    def __init__(self, checkpoint: str | Path, sims: int = 50, seed: int = 0, name: str | None = None, cuda: bool | None = None, temp_plies: int = 4) -> None:
        self.game = Connect4Game()
        args = dotdict({**ARGS, "cuda": ARGS.cuda if cuda is None else cuda})
        self.net = NNetWrapper(self.game, args)
        path = Path(checkpoint)
        self.net.load_checkpoint(str(path.parent), path.name)
        self.sims = int(sims)
        self.temp_plies = int(temp_plies)
        self.args = dotdict({"numMCTSSims": self.sims, "cpuct": 1.0})
        self.mcts = MCTS(self.game, self.net, self.args)
        self.name = name or f"alphazero_{path.stem.split('.')[0]}_{self.sims}"
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, cells: np.ndarray, legal: np.ndarray) -> int:
        stones = int((np.asarray(cells) != 0).sum())
        if stones <= 1:
            self.mcts = MCTS(self.game, self.net, self.args)
        temp = 1 if stones < self.temp_plies else 0
        probs = np.asarray(self.mcts.getActionProb(canonical(cells), temp=temp), dtype=float)
        probs = np.where(np.asarray(legal, bool), probs, 0.0)
        self.calls += 1
        if temp == 0 or probs.sum() <= 0:
            return int(np.argmax(np.where(np.asarray(legal, bool), probs, -1.0)))
        return int(self.rng.choice(len(probs), p=probs / probs.sum()))
