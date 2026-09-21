"""Train alpha-zero-general's Connect Four (their Coach, self-play with MCTS) in PyTorch.

    python connect4/bench/alphazero/train.py --out runs/connect4/alphazero --iters 50

Writes ``best.pth.tar`` and ``checkpoint_<i>.pth.tar`` under ``--out`` and a ``train.json``
with the arguments, the wall clock per iteration and the Arena acceptance record.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from connect4.bench.alphazero import import_azg
from connect4.bench.alphazero.net import ARGS, NNetWrapper

import_azg()
import logging

import coloredlogs
from Coach import Coach
from connect4.Connect4Game import Connect4Game
from utils import dotdict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--episodes", type=int, default=100, help="self-play games per iteration")
    p.add_argument("--sims", type=int, default=25, help="MCTS simulations per move")
    p.add_argument("--arena", type=int, default=40, help="games of the acceptance arena")
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--history", type=int, default=20, help="iterations of examples kept")
    p.add_argument("--resume", action="store_true", help="continue from best.pth.tar in --out")
    a = p.parse_args()
    coloredlogs.install(level="INFO")
    a.out.mkdir(parents=True, exist_ok=True)
    args = dotdict({
        "numIters": a.iters, "numEps": a.episodes, "tempThreshold": 15, "updateThreshold": 0.6, "maxlenOfQueue": 200000,
        "numMCTSSims": a.sims, "arenaCompare": a.arena, "cpuct": 1, "checkpoint": str(a.out) + "/", "load_model": a.resume,
        "load_folder_file": (str(a.out), "best.pth.tar"), "numItersForTrainExamplesHistory": a.history,
    })
    net_args = dotdict({**ARGS, "num_channels": a.channels, "epochs": a.epochs})
    NNetWrapper.default_args = net_args
    game = Connect4Game()
    net = NNetWrapper(game, net_args)
    if a.resume:
        net.load_checkpoint(str(a.out), "best.pth.tar")
    coach = Coach(game, net, args)
    if a.resume:
        coach.loadTrainExamples()
    started = time.time()
    coach.learn()
    (a.out / "train.json").write_text(json.dumps({"args": dict(args), "net": dict(net_args), "seconds": time.time() - started}, indent=1))
    logging.getLogger(__name__).info("done in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
