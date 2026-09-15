# The Connect Four page

A board a visitor plays on, with the whole brain beside it. The visitor drops a stone by
clicking a column; the brain answers from the records it learned in its life, and writes the
game it is playing into those same records. Nothing is scripted and no engine is called: the
page runs the brain, not a recording of it.

    python connect_four/web/build_page.py --run runs/connect_four/acceptance --seed 10
    python connect_four/tools/parity_fixture.py --run runs/connect_four/acceptance --seed 10 \
        --games 32 --out runs/connect_four/parity
    node connect_four/web/parity.mjs runs/connect_four/parity
    python connect_four/web/check_page.py connect_four/index.html

## The files

- `brain.js`: the port of `connect_four/brain.py`. The two record cortices (the port of
  `cadence.Records`: habituation, the fixed expansion, the winner code, the delta rule on the
  active cells), the imagination with its validator, negamax with alpha-beta within the budget,
  and the writes after every observed move and every finished game. The S00 part of the brain is
  the shared engine in `web/engine.js`, instantiated from the agent snapshot with this planner
  as its controller. Also here: `PCG64`, the port of numpy's generator and of
  `Generator.integers`, continued from the state the checkpoint saved, so a tie among equal
  moves falls the way it falls in Python.
- `game.js`: the port of `connect_four/env.py`, the rules and the world that emits moments.
- `page.js`, `index.html`, `style.css`: the page. The board is drawn on a canvas, the brain
  through `web/brain_scan.js`, and the two cortices through `web/records_view.js`: the drop
  cortex shows the cells of the chosen column's reading, the line cortex the cells of the
  windows through the new stone, and both flash orange where the outcome was written.
- `build_page.py`: exports one file per checkpoint of a run and writes the page.
- `check_page.py`: the headless check.
- `parity.mjs`: the harness that holds this brain to the Python one.

## What the page carries

`build_page.py --run <dir>` writes, beside `index.html`:

- `checkpoints/<id>.json`, one per checkpoint of the life: the agent snapshot of `agent/web.py`
  and a brain block with the record tables, the habituation means, the planner's statistics and
  generator state, the validity record and the counters.
- `cortices.json`: the projection and the offsets both cortices were drawn with. Every
  checkpoint of one life shares them, and the page carries them inlined.
- `checkpoints.json`: the manifest and the receipt's numbers, also inlined.

The page holds the last checkpoint inlined and fetches another when the visitor picks it, which
a page opened from `file://` cannot do.

## The fixed cells travel with the life

`cadence.Records` draws its expansion from `Mulberry32`, whose uniform draws are integer
arithmetic, and turns them into normals with `log`, `cos` and `sin`. Those round differently on
different platforms: the acceptance run drew its projection on Linux x86_64, and a rebuild from
the same seed on macOS arm64 differs by up to 2.2e-16 in 1,313 of 38,000 entries, which is enough
for `Brain.load` to refuse the checkpoint. V8 differs from both by the same order. So the
expansion the life used travels in `cortices.json` and the page loads it, while the seed is still
used to regenerate the expansion and the largest difference is kept in `brain.expansion` (the
page's `window.__page.stats.expansion`). `build_page.py` loads such a checkpoint through
`load_brain`, which tolerates the difference and then installs the recorded arrays.

## Parity

`tools/parity_fixture.py` records the Python brain playing 32 games against a fixed random
opponent with learning on: every call of `Brain.step`, the column it chose, the landings the drop
records predicted, the value the line records read for every legal column, the imagined next
board, the counters and the moments of every record table. `parity.mjs` replays the same games
through `brain.js` from the same checkpoint and compares. On the acceptance life of seed 10, 458
steps and 229 decisions: every column, every landing, every imagined board and every counter
match exactly, the reads to 6.7e-16 and the record tables to 1.3e-15 after the last game.

## The bound the check holds

One exchange is the brain reading the visitor's move, searching, moving and reading its own move.
`check_page.py` plays a game at every checkpoint and fails when one exchange takes longer than
100 ms. On the acceptance page the worst exchange measures 21 ms at the last checkpoint and under
15 ms at the earlier ones, of which the search is 8 to 18 ms.
