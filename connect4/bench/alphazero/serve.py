"""An AlphaZero checkpoint behind a line protocol, for the page engine's arena.

    python connect4/bench/alphazero/serve.py <checkpoint> <simulations>

One request per line: ``<seed> <42 cells>`` with the cells row-major from the bottom-left as
the side to move sees them (``.`` empty, ``x`` its own stone, ``o`` the other side's). The
answer is the column. A seed that differs from the last one reseeds the opening sampling, so
the two games of a pair see the same openings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# alpha-zero-general's own game package is named ``connect4`` as well, so this example's
# package must not be imported here: the wrapper is loaded under a name of its own.
import importlib.util  # noqa: E402

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("c4_alphazero", _HERE / "__init__.py", submodule_search_locations=[str(_HERE)])
_package = importlib.util.module_from_spec(_spec)
sys.modules["c4_alphazero"] = _package
_spec.loader.exec_module(_package)
from c4_alphazero.opponent import AlphaZeroOpponent  # noqa: E402


def main() -> None:
    opponent = AlphaZeroOpponent(sys.argv[1], sims=int(sys.argv[2]), seed=0, cuda=False)
    last = None
    print("ready", flush=True)
    for line in sys.stdin:
        seed, board = line.split()
        if seed != last:
            opponent.reseed(int(seed))
            last = seed
        cells = np.array([".xo".index(ch) for ch in board])
        legal = cells.reshape(6, 7)[5] == 0
        print(int(opponent(cells, legal)), flush=True)


if __name__ == "__main__":
    main()
