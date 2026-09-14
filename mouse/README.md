# Teachable mouse

From the repository root: `python serve.py mouse`. The launcher opens
`/mouse/`; use `/` to browse the other examples.

Try it:

1. Pick a task (**Find cheese**, **Go home**, **Get water**). The mouse recalls the
   destination from memory and walks there at once. The three buttons under the
   maze issue these commands directly, resume a paused body and end held inspection. After arrival,
   choose another task: the same mouse continues from its current position.
2. Pick **New task**. The mouse stops, because the task has no lesson.
3. Choose a destination under **Teach this task to go to** and press **Teach**.
   One memory write stores the lesson and the mouse walks there.
4. Teach the same task a different destination and it changes course. The other
   tasks keep their lessons, and **New maze** carries them into a new layout.

Click walls to change the maze, **Move goal**, or switch off **Motor neurons**.
Actual transient and persistent synaptic strengths survive browser reloads. Repeat a lesson to strengthen its persistent trace; reloading does not rehearse it.

## Control path

Task cues recall destinations from associative memory. A supplied visual map defines a recurrent spatial field. Position errors drive four directional motor units, which move the body.

265 allocated neurons, 132 initially unmasked, 285 synapses and 32 persistent memory weights plus 32 transient residuals in seed 13. Map and readout rules are supplied.

The circuit at the top shows actual state, activity changes and retained information.
Supplied readout and body rules are documented rather than shown as extra neurons.
See the [shared viewer guide](../METHODS.md#read-the-brain-view).

## A shared equilibrium

The labeled regions participate in one connected solve for the current input.
Changing tasks keeps the prior neural potentials and repairs the new goal field
in that solve; no separately pre-solved spatial field hides the replanning cascade.
Local readback and settling change the same joint state; the action readout uses
that state. The top circuit panel shows the global equation error and can replay
the actual cross-region cascade. Observed lessons update synaptic strengths within the ongoing interaction loop. Repetition strengthens their persistent component. Neuronal settling and plasticity have different timescales.
[Task connectomes, boundaries and tests](../COUPLED_BRAINS.md) explain the
connections. Self-consistency is not a guarantee of the globally best behavior.

## Reproduce

Run `node tools/nervous_system_benchmark.mjs` from the repository root.
`node --test mouse/brain.test.mjs` checks successive cheese/home/water/cheese
arrivals in three layouts, variable frame timing, waiting between tasks and
replanning from retained state. The browser suite tests actual task controls.
[evidence.json](evidence.json) contains all scheduled task trials and source hashes.

See the [advantage contract](../ADVANTAGES.md) for navigation and motor-ablation
results, with BFS and dictionary lookup as strong conventional controls.
