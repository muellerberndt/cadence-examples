import test from "node:test";
import assert from "node:assert/strict";
import { Mouse } from "./brain.js";
import { TaskLessons, stations } from "../shared/embodied.js";
import { settlingTrace } from "../shared/telemetry.js";

test("one mouse follows successive learned tasks after waiting at each destination", () => {
  for (const seed of [13, 23, 33]) {
    const mouse = new Mouse(seed), lessons = new TaskLessons();
    for (const cue of [0, 1, 2, 0]) {
      const destination = lessons.recall(cue);
      mouse.world.goal = stations(mouse.world)[destination];
      mouse.bindTask(lessons.memory, cue, destination);
      let ticks = 0;
      while (mouse.cell !== mouse.world.goal && ticks++ < 5000) {
        const dt = [0.008, 0.016, 0.05, 0.024][ticks % 4];
        mouse.step(dt);
        assert.equal(mouse.world.grid[Math.floor(mouse.y) * 19 + Math.floor(mouse.x)], 0);
      }
      assert.equal(mouse.cell, mouse.world.goal, `seed ${seed}, task ${cue}`);
      const moves = mouse.moves;
      for (let t = 0; t < 30; t++) mouse.step(.025);
      assert.equal(mouse.moves, moves, "arrival rests until another command");
    }
    assert.equal(mouse.arrivals, 4);
  }
});

test("a changed goal repairs the retained spatial state in the joint solve", () => {
  const mouse = new Mouse(), lessons = new TaskLessons();
  mouse.step();
  const retained = mouse.place.potential.slice();
  mouse.world.goal = stations(mouse.world)[1];
  mouse.bindTask(lessons.memory, 1, 1);
  assert.deepEqual(mouse.joint.initialPotential.slice(0, retained.length), retained);
  const trace = settlingTrace(mouse.joint);
  assert.ok(Math.max(...trace.mismatches[0].map(Math.abs)) > .01);
  assert.ok(Math.max(...trace.mismatches.at(-1).map(Math.abs)) <= 1e-13);
  assert.ok(trace.frames.some(row => row.slice(0, retained.length).some((v, i) =>
    Math.abs(v - trace.frames[0][i]) > .001)));
});
