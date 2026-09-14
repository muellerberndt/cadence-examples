import test from "node:test";
import assert from "node:assert/strict";
import { Worker } from "node:worker_threads";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { Reasoner, reason, drop, brainSnapshot } from "./brain.js";

function finish(planner, chunk = 17) {
  while (planner.active) {
    const before = planner.nodes;
    planner.tick(chunk);
    assert.ok(planner.nodes - before <= chunk);
  }
  return planner.result;
}
const scores = r => r.candidates.map(c => [c.column, c.score]);

test("bounded slices snapshot input, cancel old work and preserve exact choices", () => {
  const p = new Reasoner(), board = Array(42).fill(0);
  p.start(board, 1);
  board[0] = -1; // Caller mutation after start must not enter the imagined board.
  p.tick(1);
  assert.equal(p.nodes, 1);
  assert.equal(p.result, null);
  p.cancel();
  p.tick();
  assert.equal(p.nodes, 1);
  assert.equal(p.active, false);
  p.start(Array(42).fill(0), 1, { depth: 3 });
  const warm = finish(p), cold = reason(Array(42).fill(0), 1, { depth: 3 });
  assert.deepEqual(scores(warm), scores(cold));
  assert.equal(warm.nodes, cold.nodes);
  const count = p.totalNodes;
  p.tick();
  assert.equal(p.totalNodes, count);
});

test("opponent pondering reuses compatible work and accounts for its cost", () => {
  const p = new Reasoner(), board = Array(42).fill(0);
  p.start(board, 1);
  const pondered = finish(p);
  assert.equal(pondered.depth, 6);
  const next = drop(board, 3, 1);
  p.start(next, -1);
  const warm = finish(p), cold = reason(next, -1);
  assert.equal(warm.depth, cold.depth);
  assert.deepEqual(scores(warm), scores(cold));
  assert.equal(warm.column, cold.column);
  assert.ok(warm.cacheHits > 0 && warm.nodes < cold.nodes);
  assert.equal(p.totalNodes, pondered.nodes + warm.nodes);
  assert.ok(p.totalNodes > cold.nodes); // Thinking ahead is not free work.
  assert.deepEqual(board, Array(42).fill(0));
  assert.equal(next.filter(Boolean).length, 1);
  p.reset();
  assert.equal(p.cache.size, 0);
  assert.equal(p.result, null);
  assert.equal(p.totalNodes, 0);
});

test("cache bound and hard budget preserve only completed depths", () => {
  const p = new Reasoner();
  p.start(Array(42).fill(0), 1, { depth: 4 });
  finish(p);
  assert.ok(p.cache.size > 3);
  p.start(Array(42).fill(0), 1, { cacheLimit: 3, depth: 4 });
  finish(p);
  assert.ok(p.cache.size <= 3);
  p.start(Array(42).fill(0), -1, { maxNodes: 1 });
  const r = finish(p);
  assert.equal(r.nodes, 1);
  assert.equal(r.depth, 0);
  assert.equal(r.budgetExhausted, true);
  assert.throws(() => p.tick(0));
});

test("worker reports the replacement observation with matching candidate scores", async () => {
  const url = new URL("./worker.js", import.meta.url).href;
  const worker = new Worker(`
    const {parentPort} = require('node:worker_threads');
    globalThis.self = {postMessage: value => parentPort.postMessage(value)};
    import(${JSON.stringify(url)}).then(() => {
      parentPort.on('message', data => self.onmessage({data}));
      parentPort.postMessage({kind:'ready'});
    });
  `, { eval: true });
  const board = Array(42).fill(0), next = drop(board, 3, 1);
  try {
    const result = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(Error("worker replacement timed out")), 10000);
      let replaced = false;
      worker.on("error", reject);
      worker.on("message", message => {
        if (message.kind === "ready") worker.postMessage({id: 1, board, player: 1, depth: 6});
        if (message.id === 1 && message.kind === "progress" && !replaced) {
          replaced = true;
          worker.postMessage({id: 2, board: next, player: -1, depth: 4});
        }
        if (message.kind === "error") { clearTimeout(timeout); reject(Error(message.message)); }
        if (message.id === 2 && message.kind === "done") {
          clearTimeout(timeout); resolve(message.result);
        }
      });
    });
    assert.deepEqual(scores(result), scores(reason(next, -1, {depth: 4})));
    assert.equal(result.depth, 4);
  } finally {
    await worker.terminate();
  }
});


test("new input cancels scheduled work and rejects a stale worker callback", () => {
  const messages = [], scheduled = new Map();
  let clock = 0;
  const self = {postMessage: message => messages.push(message)};
  const code = readFileSync(new URL("./worker.js", import.meta.url), "utf8")
    .replace('import { Reasoner } from "./brain.js";', '');
  runInNewContext(code, {
    Reasoner, self,
    setTimeout: callback => { scheduled.set(++clock, callback); return clock; },
    clearTimeout: id => scheduled.delete(id),
  });
  const slice = () => {
    const [id, callback] = scheduled.entries().next().value;
    scheduled.delete(id);
    callback();
  };
  const board = Array(42).fill(0);
  self.onmessage({data: {id: 1, board, player: 1, depth: 6}});
  slice();
  assert.ok(messages.some(m => m.kind === "progress"));
  assert.ok(!messages.some(m => m.kind === "done"));
  const stale = scheduled.values().next().value;
  self.onmessage({data: {id: 2, board: drop(board, 3, 1), player: -1, depth: 4}});
  const marker = messages.length;
  stale(); // Even a callback already dequeued by a scheduler cannot advance the old job.
  while (scheduled.size) slice();
  assert.ok(messages.slice(marker).every(m => m.id === 2));
  assert.equal(messages.at(-1).kind, "done");
  assert.equal(messages.at(-1).result.depth, 4);
  self.onmessage({data: {kind: "cancel", id: 3, reset: true}});
  assert.equal(scheduled.size, 0);
});


test("rendering brain readback never mutates the stored decision", () => {
  const board = Array(42).fill(0), result = reason(board, 1, {depth: 2});
  const before = JSON.stringify(result);
  brainSnapshot(board, 1, result);
  assert.equal(JSON.stringify(result), before);
});
