// Reproducible joint-equation and cross-region ablations for the six live brains.
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import { DrawingArm } from "../eye-arm/brain.js";
import { imageFixture } from "../eye-arm/fixtures.js";
import { Mouse } from "../mouse/brain.js";
import { WormArena } from "../worm/brain.js";
import { Forager } from "../fly/brain.js";
import { MemoryBrain } from "../memory/brain.js";
import { FastMemory, flowers, keys } from "../shared/engine.js";
import { reason, drop, brainSnapshot } from "../connect-four/brain.js";
import { Circuit } from "../shared/nervous_system.js";
import { repairTrace } from "../shared/telemetry.js";
const root = new URL("../", import.meta.url);
export function samples() {
  const arm = new DrawingArm(imageFixture("square"));
  for (let i = 0; i < 20; i++) arm.step();
  const mouse = new Mouse(13);
  mouse.step();
  const worm = new WormArena(
    JSON.parse(readFileSync(new URL("worm/worm.json", root))),
  );
  worm.step();
  const fly = new Forager(new FastMemory()),
    field = flowers();
  fly.memory.observe(keys()[field[0].kind], [0, 0, 0, 1]);
  fly.step(field);
  const memory = new MemoryBrain();
  memory.observe(keys()[0], [0, 0, 1, 0]);
  let board = Array(42).fill(0);
  [3, 2, 3, 4].forEach((c, i) => (board = drop(board, c, i % 2 ? -1 : 1)));
  return {
    arm: arm.brain.snapshot(),
    mouse: mouse.joint,
    worm: worm.brain(),
    fly: fly.brain(field),
    memory: memory.brain(keys()[0]),
    game: brainSnapshot(
      board,
      1,
      reason(board, 1, { depth: 4, maxNodes: 20000 }),
    ),
  };
}
export function benchmark() {
  const results = Object.entries(samples()).map(([name, s]) => {
    assert.ok(!s.blocks, `${name}: needs one actual graph`);
    const incoming = Array(s.state.length).fill(0);
    for (const [a, b, w] of s.edges) incoming[b] += w * s.state[a];
    const residual = Math.max(
      ...s.potential.map((v, i) =>
        Math.abs(s.mask[i] ? incoming[i] + s.drive[i] - v : v),
      ),
    );
    assert.ok(
      residual <= s.equilibrium.tolerance,
      `${name}: equation residual ${residual}`,
    );
    const replayError = Math.max(
      ...repairTrace(s)
        .frames.at(-1)
        .map((v, i) => Math.abs(v - s.state[i])),
    );
    assert.ok(replayError < 1e-12, `${name}: replay error`);
    const cut = new Circuit(
      s.names,
      s.groups,
      s.edges.filter(([a, b]) => s.groups[a] === s.groups[b]),
      s.dt,
    );
    cut.state = s.state.slice();
    cut.potential = s.potential.slice();
    cut.mask = s.mask;
    cut.settle(s.drive, 3000, 1e-13);
    const regions = Object.keys(s.equilibrium.regions);
    const changed = regions.filter((g) =>
      s.state.some(
        (v, i) => s.groups[i] === g && Math.abs(v - cut.state[i]) > 1e-12,
      ),
    );
    assert.ok(
      changed.length,
      `${name}: inter-region links must have a causal effect`,
    );
    return {
      name,
      owners: s.state.length,
      seams: s.edges.length,
      steps: s.steps,
      tolerance: s.equilibrium.tolerance,
      residual,
      regions: s.equilibrium.regions,
      active_region_links: s.equilibrium.links,
      replay_error: replayError,
      changed_regions_when_links_cut: changed,
    };
  });
  const files = [
    "shared/nervous_system.js",
    "shared/telemetry.js",
    "shared/engine.js",
    "shared/embodied.js",
    "worm/worm_arena.js",
    "worm/worm.json",
    "eye-arm/brain.js",
    "eye-arm/fixtures.js",
    "mouse/brain.js",
    "worm/brain.js",
    "fly/brain.js",
    "memory/brain.js",
    "connect-four/brain.js",
    "tools/coupled_brain_benchmark.mjs",
  ];
  const sources = Object.fromEntries(
    files.map((f) => [
      f,
      createHash("sha256")
        .update(readFileSync(new URL(f, root)))
        .digest("hex"),
    ]),
  );
  // Numerical certificates use conservative envelopes, stable across JS runtimes.
  const envelope = (v) => (v === 0 ? 0 : 10 ** Math.ceil(Math.log10(v)));
  for (const row of results) {
    row.residual = envelope(row.residual);
    row.replay_error = envelope(row.replay_error);
    row.regions = Object.fromEntries(
      Object.entries(row.regions).map(([g, v]) => [g, envelope(v)]),
    );
  }
  writeFileSync(
    new URL("evidence/coupled_evidence.json", root),
    JSON.stringify(
      {
        schema: "cadence.joint-equilibrium/v1",
        residual_reporting: "upper power-of-ten envelope",
        sources,
        results,
      },
      null,
      2,
    ) + "\n",
  );
  console.log(JSON.stringify(results));
}
if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url)
  benchmark();
