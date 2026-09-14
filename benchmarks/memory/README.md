# Memory benchmarks for embodied learning

These reproducible kernel comparisons support the mouse and forager demos. They
have no standalone website. The live bodies use the same `SynapticMemory` rule:
observed experiences change transient and persistent weights; predictions alone
do not write new memories.

From the repository root:

```bash
node benchmarks/memory/benchmark.mjs
node benchmarks/memory/history_benchmark.mjs
node benchmarks/memory/consolidation_benchmark.mjs
```

`evidence.json` records distinct-key stream timing against online MLP update
budgets. It excludes graded readout, body simulation and visualization. See the
[comparison contracts](../../ADVANTAGES.md#measured-processing-time) for all costs
and controls. Similar keys can interfere with associative recall.

## History-required recall

The same current cue can require different answers after different lessons.
`history_evidence.json` keeps all 128 queries and predictions. Cadence's fast
reference and ordinary last-value lookup score 100%; a frozen query-only MLP
scores 25%, the best possible fixed predictor on this balanced schedule.
A transformer given lesson history is outside that restriction.

## Lasting synaptic memory

`consolidation_evidence.json` clears all transient weights before measuring recall:

| Experience | Remaining unit-cue response |
|---|---:|
| One ordinary lesson | 0.0500 |
| 40 repeated lessons | 0.8715 |
| One salience-19 lesson | 1.0000 |
| 40 lessons without consolidation | 0.0000 |

An association survives 200 orthogonal distractors and changes after 60 corrective
observations (response 0.9539). These are numerical model responses, not biological
retention rates. The tests also check partial feedback and checkpoint recovery.
