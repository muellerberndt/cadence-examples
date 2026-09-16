#!/usr/bin/env bash
# Fetches and builds the external opponents under connect_four/external/ (git-ignored):
# Pascal Pons' perfect solver with its opening book, behind the board-protocol bridge, and
# alpha-zero-general for the AlphaZero baseline. Run from the repository root.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ext="$here/../external"
mkdir -p "$ext"
if [ ! -d "$ext/pons" ]; then
  git clone -q --depth 1 https://github.com/PascalPons/connect4.git "$ext/pons"
fi
cp "$here/bridge.cpp" "$ext/pons/bridge.cpp"
( cd "$ext/pons" && make -s c4solver && g++ --std=c++11 -O3 -DNDEBUG -o c4bridge bridge.cpp Solver.o )
if [ ! -f "$ext/pons/7x6.book" ]; then
  ( cd "$ext/pons" && ( gh release download book --repo PascalPons/connect4 --pattern '7x6.book' \
      || curl -sSL -o 7x6.book https://github.com/PascalPons/connect4/releases/download/book/7x6.book ) )
fi
if [ ! -d "$ext/azg" ]; then
  git clone -q --depth 1 https://github.com/suragnair/alpha-zero-general.git "$ext/azg"
fi
printf '..........................................\n' | "$ext/pons/c4bridge" "$ext/pons/7x6.book"
# numpy 2 removed np.int and ndarray.tostring
sed -i.bak 's/dtype=np\.int)/dtype=int)/; s/board\.tostring()/board.tobytes()/' "$ext/azg/connect4/Connect4Logic.py" "$ext/azg/connect4/Connect4Game.py"
