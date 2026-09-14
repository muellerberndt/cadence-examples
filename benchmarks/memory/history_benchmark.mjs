// Identical current cues, different previously taught meanings.
// This isolates access to history; it is not a claim against networks with context.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { FastMemory, MLP, argmax, keys, zeros } from "../../shared/engine.js";

export function runHistoryBenchmark() {
  const cues = keys(),
    memory = new FastMemory(),
    lookup = new Map(),
    initial = JSON.parse(
      readFileSync(new URL("../../evidence/evidence.json", import.meta.url)),
    ).browser.online_mlp,
    frozen = new MLP(initial),
    rows = [];
  for (let round = 0; round < 16; round++) {
    // Both stateful controllers receive every lesson before the query phase.
    // The frozen/query-only control has no retained lesson, context, clock or index.
    for (let key = 0; key < cues.length; key++) {
      const value = (round + key) % 4,
        target = zeros(4);
      target[value] = 1;
      memory.observe(cues[key], target);
      lookup.set(JSON.stringify(cues[key]), value);
    }
    for (let key = 0; key < cues.length; key++)
      rows.push([
        round,
        key,
        (round + key) % 4,
        argmax(memory.predict(cues[key])),
        argmax(frozen.predict(cues[key])),
        lookup.get(JSON.stringify(cues[key])),
      ]);
  }
  // Best possible fixed deterministic f(current cue), even with hindsight.
  // Group by the actual feature vector, not by the desired answer or history.
  const counts = new Map();
  for (const [, key, target] of rows) {
    const input = JSON.stringify(cues[key]);
    if (!counts.has(input)) counts.set(input, [0, 0, 0, 0]);
    counts.get(input)[target]++;
  }
  const bound = [...counts.values()].reduce((n, c) => n + Math.max(...c), 0),
    accuracy = (column) =>
      rows.filter((r) => r[column] === r[2]).length / rows.length;
  const scores = {
    cadence: accuracy(3),
    frozen_mlp: accuracy(4),
    history_lookup: accuracy(5),
    best_fixed_query_only: bound / rows.length,
  };
  assert.equal(scores.cadence, 1);
  assert.equal(scores.history_lookup, 1);
  assert.equal(scores.best_fixed_query_only, 0.25);
  assert.ok(scores.frozen_mlp <= scores.best_fixed_query_only);
  const sources = Object.fromEntries(
    [
      "benchmarks/memory/history_benchmark.mjs",
      "shared/engine.js",
      "evidence/evidence.json",
    ].map((path) => [
      path,
      createHash("sha256")
        .update(readFileSync(new URL("../../" + path, import.meta.url)))
        .digest("hex"),
    ]),
  );
  return {
    schema: "cadence.history-required/v1",
    rounds: 16,
    writes: rows.length,
    queries: rows.length,
    columns: [
      "round",
      "key",
      "target",
      "cadence",
      "frozen_mlp",
      "history_lookup",
    ],
    cues,
    rows,
    scores,
    sources,
    boundary:
      "A fixed deterministic function of the current cue alone cannot distinguish histories with different correct labels. This restriction excludes retained activations, changed weights, past-context tokens, a clock and an external store. A history lookup also solves the task. No general impossibility claim about MLPs or transformers with memory/context.",
  };
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  const report = runHistoryBenchmark();
  writeFileSync(
    new URL("./history_evidence.json", import.meta.url),
    JSON.stringify(report, null, 2) + "\n",
  );
  console.log(
    JSON.stringify({ queries: report.queries, scores: report.scores }),
  );
}
