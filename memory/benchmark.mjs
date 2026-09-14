import { FastMemory, MLP, keys, argmax, zeros } from "../showcase/engine.js";
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { cpus } from "node:os";
const bank = keys(),
  initial = JSON.parse(
    readFileSync(new URL("../showcase/evidence.json", import.meta.url)),
  ).browser.online_mlp;
function stream(kind) {
  const model = kind === "cadence" ? new FastMemory() : new MLP(initial),
    truth = Array(8).fill(-1);
  let correct = 0,
    queries = 0;
  const start = performance.now();
  for (let k = 0; k < 128; k++) {
    const key = (k * 5 + 3) % 8,
      value = (Math.floor(k / 8) + k) % 4,
      target = zeros(4);
    target[value] = 1;
    model.observe(bank[key], target, kind === "cadence" ? 1 : +kind);
    truth[key] = value;
    for (let i = 0; i < 8; i++)
      if (truth[i] >= 0) {
        correct += +(argmax(model.predict(bank[i])) === truth[i]);
        queries++;
      }
  }
  return {
    correct,
    queries,
    accuracy: correct / queries,
    ms: performance.now() - start,
  };
}
const rows = [];
for (const kind of ["cadence", "1", "10", "100"]) {
  for (let i = 0; i < 3; i++) stream(kind);
  const runs = Array.from({ length: 11 }, () => stream(kind));
  const sorted = runs.map((r) => r.ms).sort((a, b) => a - b);
  rows.push({
    kind,
    accuracy: runs[0].accuracy,
    correct: runs[0].correct,
    queries: runs[0].queries,
    median_ms: sorted[5],
    min_ms: sorted[0],
    max_ms: sorted.at(-1),
    updates_per_observation: kind === "cadence" ? 1 : +kind,
  });
}
const sources = Object.fromEntries(
  ["memory/benchmark.mjs", "showcase/engine.js", "showcase/evidence.json"].map(
    (path) => [
      path,
      createHash("sha256")
        .update(readFileSync(new URL("../" + path, import.meta.url)))
        .digest("hex"),
    ],
  ),
);
const report = {
  schema: "cadence.memory-runtime/v1",
  runtime: process.version,
  cpu: cpus()[0].model,
  platform: process.platform,
  architecture: process.arch,
  writes: 128,
  warmups: 3,
  repeats: 11,
  rows,
  sources,
  boundary:
    "Local JavaScript training plus queries on a distinct-key overwrite stream. Supplied explicit keys; no general language or energy-efficiency claim. Timings include memory allocation and differ across machines.",
};
writeFileSync(
  new URL("./evidence.json", import.meta.url),
  JSON.stringify(report, null, 2) + "\n",
);
console.log(JSON.stringify(rows));
