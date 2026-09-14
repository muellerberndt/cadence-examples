# What each example demonstrates

Cadence exposes bounded state, ports, local readback, retained records and repair
as explicit software components. Each example tests a particular consequence of
that design. A frozen feedforward mapping does not update its weights between
observations; conventional systems can add memory, feedback and search too.
These experiments compare specified implementations, not everything an MLP or
transformer could implement. None establishes universal architectural superiority.

| Example | Intervention and supported benefit | Comparison and limit |
|---|---|---|
| [Changing memory](memory/) | Replace distinct-key records in one residual write while retaining earlier keys. The recorded 128-write stream gives 996/996 correct queries. | Same observations and queries for an online MLP at 1, 10 and 100 updates. About 3×, 7× and 52× lower local processing time respectively; explicit keys, one recorded runtime stream. A dictionary is also exact. Correlated keys can favor the MLP. |
| [Teachable mouse](mouse/) | Teach or revise a cue/destination pair during use; retained lessons work in a new maze. Spatial feedback and motor activity adapt the route to the goal. | 24/24 current intact/moved-goal trials succeed; 12 motor ablations cannot move. Earlier field trials also compare BFS, which succeeds. The full map and symbolic task cues are supplied; this is not natural-language task learning or a speed win over BFS. |
| [Eye & arm](eye-arm/) | Pixel readback, joint coordination and opposing motor populations form a continuous correction loop. Pencil motors control contact. | All targets covered on three fixtures with intact or disturbed feedback; 32–42% after the same disturbance without pose feedback. Silenced joints cannot move; silenced raised-pencil motors cannot draw. These are causal controls, not a trained neural baseline or artistic-fidelity test. |
| [C. elegans habitat](worm/) | Local food signals drive the chemical graph and directional motor circuit. Changed walls alter the supplied odor field and behavior. | Two food patches consumed in the fixed intact trial; smell, chemical-motor, directional-motor and sealed-wall controls acquire none. The separate 297-cell circuit comparison tests exact dynamics under interventions; its MLP surrogate is faster but approximate. No whole-worm simulation or overall speed claim. |
| [Fly-inspired forager](fly/) | Actual flower contact revises nectar memory; sensory/motor activity moves the body. Changing nectar creates new observations during use. | Fixed actuator trial: 25 contacts intact, zero with motors disabled. Live MLP and Cadence agents share motor design but see different trajectories; nectar totals are illustrative, not a matched learning benchmark. Use the memory stream for the controlled learner comparison. |
| [Connect Four](connect-four/) | Compare hypothetical replies without changing the live board; monitor own option-value changes and ambiguity to allocate further search. | 8/8 wins against one-ply evaluation, 5/8 against four-ply search over four openings and both sides. Same evaluator, different search budgets. Conventional minimax can match this; recurrence does not automatically provide a world model, and the monitor is not evidence of consciousness. |

## A precise limit of frozen inference without history

The [history-required experiment](memory/history_evidence.json) presents eight
distinct cues over 16 rounds. Each round first teaches their new meanings, then
queries them. Each cue takes each of four meanings equally often. The query
contains only that cue, with no history, clock, round index or previous answer.

| Controller | Correct queries | Information retained between lessons and queries |
|---|---:|---|
| Cadence residual memory | 128/128 (100%) | Updated associative weights |
| Conventional last-value lookup | 128/128 (100%) | Stored latest lesson for each cue |
| Frozen example MLP | 32/128 (25%) | None |
| Best possible fixed deterministic function of the current cue | At most 32/128 (25%) | None; bound allows optimal answers chosen in hindsight |

The bound follows directly: a fixed function must return one answer for each
identical input. Each answer is correct on only four of that cue's sixteen
queries. More layers or parameters cannot resolve this missing information.
For a history-independent randomized predictor the same bound holds in
expectation; it is not a bound on every random finite-sample realization.

This is an impossibility result for the **specified information restriction**.
It is not an impossibility result for transformers with lesson history in their
context, recurrent networks, online weight updates or external storage. The
lookup control demonstrates that conventional memory also removes the obstacle.
The bundled MLP's initialization cannot weaken this conclusion: even an optimally
trained fixed query-only predictor faces the same bound.

This test isolates the retained-association mechanism used by the teachable mouse
and forager. It does not prove whole-body performance on new tasks. Run
`node memory/history_benchmark.mjs`; the receipt keeps every query, its actual
cue vector, predictions, source hashes and the exact comparison boundary.

## Choosing an efficiency or capability claim

An efficiency comparison should fix task quality and information access, count
training and adaptation as well as inference, and report memory, hardware and
latency. Two orders of magnitude means 100× on the named metric. The recorded memory result is about 52× against one specified update
budget, rather than a demonstrated 100× advantage. The other examples currently
establish the feedback, intervention or planning effects described in the table.

A capability claim must specify what the comparator can observe, retain, update
and compute. Compare frozen inference, an adaptive neural controller, and the
appropriate strong conventional method separately. In these tasks those include
dictionary memory, BFS, classical feedback and minimax. Motor ablations locate
causal mechanisms; they cannot substitute for those performance comparisons.

## Measured processing time

The [runtime receipt](memory/evidence.json) records Node 25.8.1 on an Apple M4,
three warmups and eleven measured repetitions of one deterministic distinct-key
stream. Time includes 128 writes and all 996 queries, with allocations; it
excludes process startup and initial model construction. The MLP's update count
is a chosen adaptation budget, not a claim that it always needs 100 steps.

| Learner | Correct queries | Median stream time | MLP time / Cadence time |
|---|---:|---:|---:|
| Cadence residual memory | 996/996 (100%) | 0.435 ms | 1.0× |
| MLP, 1 update per observation | 565/996 (56.7%) | 1.338 ms | 3.1× |
| MLP, 10 updates | 963/996 (96.7%) | 3.170 ms | 7.3× |
| MLP, 100 updates | 939/996 (94.3%) | 22.430 ms | 51.5× |

Sub-millisecond measurements vary by machine, runtime and scheduling. This is
neither a GPU benchmark nor an energy measurement. Run
`node memory/benchmark.mjs` to record this machine's result. CI checks its source
bindings and numerical contracts; it does not demand identical wall-clock time.
The separate [matched multi-seed benchmark](showcase/evidence.json) includes
correlated keys and preserves all scheduled outcomes.

## Evidence, not animation

The body and circuit panels expose executed state. Slowed cascades replay a
captured settlement from its real starting state; input-release probes use an
isolated copy. Learned seam writes are displayed only where learning occurs.
Most controller weights are supplied. The examples do not imply that every
movement learns or that colored activity measures happiness or neurotransmitters.

Current actuator trials: `node tools/nervous_system_benchmark.mjs`.
Strategy trials: `node connect-four/benchmark.mjs`.
Full numerical and browser checks: see the [reproduction guide](showcase/README.md#reproduce).
All scheduled trials are retained in source-bound receipts. These small suites
establish behavior on those fixtures; they are not broad generalization estimates.
