// A line protocol over Pascal Pons' solver: one board per line, the scores of its seven columns
// per line. The board is 42 characters, row-major from the bottom-left cell ('.' empty, 'x' the
// side to move, 'o' the other side), the encoding of connect_four/env.py. A score is the solver's
// (positive: the side to move wins, the larger the sooner; negative: it loses; 0: a draw) or
// -1000 for an unplayable column. The transposition table persists between lines.
#define private public
#include "Solver.hpp"
#undef private
#include <iostream>
#include <string>

using namespace GameSolver::Connect4;

int main(int argc, char **argv) {
  Solver solver;
  if (argc > 1) solver.loadBook(argv[1]);
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.size() != 42) { std::cout << "error length " << line.size() << std::endl; continue; }
    Position P;
    for (int r = 0; r < 6; r++) for (int c = 0; c < 7; c++) {
      char ch = line[r * 7 + c];
      if (ch == '.') continue;
      Position::position_t bit = Position::position_t(1) << (c * 7 + r);
      P.mask |= bit;
      if (ch == 'x') P.current_position |= bit;
      P.moves++;
    }
    std::vector<int> scores = solver.analyze(P, false);
    for (int c = 0; c < 7; c++) std::cout << scores[c] << (c < 6 ? " " : "");
    std::cout << " " << solver.getNodeCount() << std::endl;
  }
  return 0;
}
