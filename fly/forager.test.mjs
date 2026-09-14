import test from "node:test";
import assert from "node:assert/strict";
import { Forager } from "./brain.js";
import { SynapticMemory, flowers, keys } from "../shared/engine.js";

test("unobserved nectar cannot affect selection, motion or synaptic weights", () => {
  const a = new Forager(new SynapticMemory());
  const b = new Forager(new SynapticMemory());
  const field = flowers();
  const secret = field.map(f => ({ ...f, get value() {
    throw new Error("nectar read before contact");
  } }));
  for (let t = 0; t < 20; t++) { a.step(field); b.step(secret); }
  assert.equal(a.encounters, 0);
  assert.deepEqual([a.x, a.y, a.target], [b.x, b.y, b.target]);
  assert.deepEqual(a.memory, b.memory);
});

test("old observations can outweigh a known reward without a new lesson", () => {
  const agent = new Forager(new SynapticMemory());
  const field = flowers().map(f => ({ ...f, x: 0.8, y: 0.5 }));
  field.forEach((f, i) => agent.memory.observe(keys()[f.kind], i ? [1,0,0,0] : [0,0,0,1]));
  agent.visits.fill(1);
  agent.time = 100;
  agent.lastVisit.fill(100);
  agent.lastVisit[4] = 0;
  agent.choose(field);
  assert.equal(agent.target, 4);
  // Fresh observations remove the reason for checking the poorer flower.
  agent.lastVisit[4] = 100;
  agent.choose(field);
  assert.equal(agent.target, 0);
});

test("only committed flower contact updates the visit record", () => {
  const agent = new Forager(new SynapticMemory());
  const field = flowers();
  agent.x = field[2].x; agent.y = field[2].y; agent.target = 2;
  const commit = agent.step(field, 0.025, true);
  assert.equal(agent.encounters, 0);
  assert.equal(agent.lastVisit[2], -Infinity);
  commit();
  assert.equal(agent.encounters, 1);
  assert.equal(agent.lastVisit[2], agent.time);
  assert.equal(agent.last.value, field[2].value);
});
