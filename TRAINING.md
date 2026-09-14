# Training a player

The game examples follow a learning life inspired by development:

1. **Design the brain.** Choose the observation ports, carried state, connections
   and enough capacity. Use validation examples to compare candidate sizes.
2. **Imitate.** A teacher provides actions on training states. Learn them with
   the same free/nudged local update used by the digits example.
3. **Experiment.** Play games and collect the observations, actions and outcomes.
   Pong uses advantage-weighted nudges; Connect Four consults the game rules and
   its teacher on encountered positions.
4. **Revisit weaknesses.** Ask the teacher about errors, and mix those examples
   with earlier lessons so correction does not erase established skills.
5. **Keep learning.** Each later game supplies a reusable episode. Save the
   learned parameters and rehearse earlier experiences alongside new ones.

## Learn while playing in the browser

After installing `requirements.txt`, run either command from the repository root:

```bash
python serve.py pong --learn
python serve.py connect-four --learn
```

Learning runs locally in Python. The browser sends completed episodes to the local
server; no cloud service receives the game data. Each completed game is saved under
`runs/learning/`, together with the learner checkpoint and an update log. Later
sessions resume it. Paths include the shipped checkpoint's digest, so a new public
model starts a separate learning history.

The server applies small updates and teacher corrections, rehearses both initial
examples and recent episodes, and checks retention on the initial rehearsal set.
An update that reduces that check's accuracy is rolled back, while the episode is
kept for further rehearsal. This protects a measured subset of prior skills; it
does not guarantee every game's update improves the player's overall win rate.
The page reports whether the update was retained.

Without `--learn`, the pages are frozen demos. The teacher-only Pong toggle is
also frozen. Opening an HTML file alone cannot run the Python learner.

## Measure progress

Test with learning disabled, across different seeds and opponents. Report wins,
losses, draws and the exact opponent settings. Keep validation games separate from
final test games. Pong's high return rate can hide draws; winning requires making
the other paddle miss. Connect Four's lookahead strength includes the supplied
game rules, so report the raw neural policy and search-only control as well.

Save slow learned parameters across games, but reset the temporal trace between
independent episodes. Replay the trace available when an action was taken; never
advance live memory with shuffled teacher targets. The full game state, teacher
privileges, data budgets and checkpoint selection are recorded with the examples.

Digits uses teacher-labelled images. Recall and updating memory learn directly
from each observed key/value association. Circuit interventions use a supplied
rule rather than a game-training curriculum. These simpler tasks retain the
smallest mechanism that serves their purpose.

## Think before acting

Connect Four compares candidate futures using its game rules. This is a composition
of a transition model, a state evaluator and a bounded search controller. Branches
keep separate temporary state; imagined rewards do not become observed evidence.
After acting, an outcome that is better or worse than predicted supplies a signed
teaching signal. See Cadence’s tested [deliberation example](https://github.com/muellerberndt/cadence/blob/main/docs/deliberation.md)
for a learned evaluator, branch-isolation tests and a wrong-model control.
