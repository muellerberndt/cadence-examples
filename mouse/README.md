# Teachable mouse

From the repository root: `python serve.py mouse`. The launcher opens
`/mouse/`; use `/` to browse the other examples.

Try it:

1. Pick a task (**Find cheese**, **Go home**, **Get water**). The mouse recalls the
   destination from memory and walks there at once.
2. Pick **New task**. The mouse stops, because the task has no lesson.
3. Choose a destination under **Teach this task to go to** and press **Teach**.
   One memory write stores the lesson and the mouse walks there.
4. Teach the same task a different destination and it changes course. The other
   tasks keep their lessons, and **New maze** carries them into a new layout.

Click walls to change the maze, **Move goal**, or switch off **Motor neurons**.
Lessons persist in this browser.

## Control path

Task cues recall destinations from associative memory. A supplied visual map defines a recurrent spatial field. Position errors drive four directional motor units, which move the body.

265 allocated owners, 132 initially unmasked, 285 seams and 32 learned entries in seed 13. Map and readout rules are supplied.

The circuit at the top shows actual state, repairs and retained information.
Supplied readout and body rules are documented rather than shown as extra neurons.
See the [shared viewer guide](../METHODS.md#read-the-brain-view).

## A shared equilibrium

The labeled regions participate in one connected solve for the current input.
Local readback and repair change the same joint state; the action readout uses
that state. The top circuit panel shows the global equation error and can replay
the actual cross-region cascade. Lessons change records between phases.
[Task wiring, boundaries and tests](../COUPLED_BRAINS.md) explain the
connections. Self-consistency is not a guarantee of the globally best behavior.

## Reproduce

Run `node tools/nervous_system_benchmark.mjs` from the repository root.
[evidence.json](evidence.json) contains all scheduled task trials and source hashes.

See the [advantage contract](../ADVANTAGES.md) for navigation and motor-ablation
results, with BFS and dictionary lookup as strong conventional controls.
