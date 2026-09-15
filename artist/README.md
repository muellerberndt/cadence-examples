# S04: the artist draws what it is shown

The arm of S01 holds a pen over a 32 by 32 canvas. One life scribbles, learns what its
strokes leave on the canvas, and then draws requested figures: single segments, two
connected segments, polygons, curves and compositions it has never seen, replanning from
what the canvas shows after every stroke. A longer upper link and an offset canvas change
the body and the world during a drawing; the artist keeps drawing and learns the change.
Every decision and its outcome go through the S00 experience contract.

## Supplied and learned

Supplied: the arm body of S01 with its own goal switched off, the canvas with stroke
rasterisation at one pixel width, the pen state as one more action bit (nine torques by
pen up or down), the two aligned views of the target and the canvas, the discrepancy
measure the reward and the evaluation use (symmetric Chamfer distance normalised by the
canvas diagonal, with foreground F1 within one pixel reported beside it), the reward as
the change in measured discrepancy less an action cost, the target families and their
splits, and the two-level search: a slow level that chooses a stroke intention (direction,
distance, pen state) by imagining its marks, and a fast level that holds each candidate
torque for a few decisions and follows the intention through the learned body.

Learned: the records of the world head (a records cortex over the body senses, the 8 by 8
ink window around the pen, the target window, the intention and the action, with fields
for the displacement change, the velocity change, the angle change and the marks the pen
leaves), and the synapses of the settled regions, from empty records and random synapses.
There is no replay ring: each outcome is written once into the records its reading touched.

Controls: the same architecture frozen from birth, random scribbling, the supplied path
oracle, an online MLP with a replay ring behind the same search, corrupted dynamics,
shuffled action pairing, replanning removed, and the canvas readback removed.

## Files

| file | contents |
|---|---|
| `env.py` | the canvas world over the S01 arm: rasterisation, pen, windows, discrepancy and F1, families and splits, reward |
| `brain.py` | the graph over the S00 agent with its records head, the intention search and the torque search, `ArtistLife` |
| `controls.py` | the random scribbler, the path oracle and the online MLP behind the same search |
| `run.py` | the progression (scribble, segments, strokes with replanning, compositions, perturbations), held-out suites, controls, receipt, checkpoints |
| `verify.py` | recomputes every gate from the event log, checks digests, source hashes and the seed schedule, fails closed |
| `config.json` | budgets, gates and the recorded assumptions |
| `tests/` | rasterisation, pen-up travel, window alignment, discrepancy and F1 on known canvases, splits, reward, the graph contract, imagination writing nothing, copy isolation |

## Commands

```bash
PY=../cadence/.venv/bin/python
$PY -m pytest artist/tests -q
$PY artist/run.py --seeds 0 --pilot --out runs/artist/pilot        # a tenth of every budget, about ten minutes
$PY artist/run.py --seeds 10 11 12 13 14 --workers 5 --out runs/artist/acceptance   # about two hours, 9 GB
$PY artist/verify.py runs/artist/acceptance
```

## Assumptions recorded in config.json

The canvas spans 0.7 arm lengths centred half an arm length above the shoulder, so every
target pixel lies between 0.15 and 0.92 of the reach and off the arm's fold. The ink head
predicts the 8 by 8 window around the pen at canvas resolution, since one decision moves
the pen about 1.4 pixels and a coarser window cannot resolve a one-pixel stroke. The
intention travels in the goal port, so the executed reading and the imagined readings
carry the same fields. The action's neurons are driven at three times the input gain, which
separates the pen states in the code. The planning objective caps each target pixel's
reach at 1.5 pixels and a drawing finishes when the discrepancy falls under 0.010.

## Evaluation copies

Held-out drawings run on an adapting copy of the checkpoint with exploration off: the copy
keeps writing its records while it draws, which is the ongoing mode. The lesions (corrupted
dynamics, shuffled pairing) are read on the first ten drawings, since an adapting copy
heals a lesion within a few dozen drawings; the readback control removes the canvas
observation after the perturbation and must lose.

## Receipt

`receipt.json` is the acceptance receipt of seeds 10 to 14: every predicate with its value
and threshold, the source hashes of this directory and of the library, the sha256 of every
event log and checkpoint, the learning curve, and the digest of the receipt itself.
`verify.py` recomputes the gates from the event logs and fails closed on an incomplete run.
