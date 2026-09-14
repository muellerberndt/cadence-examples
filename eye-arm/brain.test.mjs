import assert from "node:assert/strict";
import test from "node:test";
import { DrawingArm, RETINA } from "./brain.js";
import { PAPER_SIZE, pixelPoint } from "./paper.js";

const line = () =>
  Array.from(
    { length: RETINA ** 2 },
    (_, i) =>
      +(Math.floor(i / RETINA) === 20 && i % RETINA >= 12 && i % RETINA <= 36),
  );
function finish(arm, budget = 4000) {
  for (let i = 0; i < budget; i++) {
    arm.step();
    if (arm.target < 0 && arm.lifted) return;
  }
  assert.fail(`Unfinished: ${arm.phase}, coverage ${arm.coverage}`);
}
// Check the actual bitmap's connected components, independently of target
// coverage, controller bookkeeping and its continuous-stroke counter.
function components(paper) {
  const unseen = new Set();
  paper.pixels.forEach((v, i) => {
    if (v) unseen.add(i);
  });
  let count = 0;
  while (unseen.size) {
    count++;
    const queue = [unseen.values().next().value];
    unseen.delete(queue[0]);
    for (let k = 0; k < queue.length; k++) {
      const i = queue[k],
        x = i % PAPER_SIZE,
        y = Math.floor(i / PAPER_SIZE);
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          if (
            x + dx < 0 ||
            x + dx >= PAPER_SIZE ||
            y + dy < 0 ||
            y + dy >= PAPER_SIZE
          )
            continue;
          const j = (y + dy) * PAPER_SIZE + x + dx;
          if (unseen.delete(j)) queue.push(j);
        }
    }
  }
  return count;
}
test("a line has no gaps between retinal samples and needs one stroke", () => {
  const arm = new DrawingArm(line());
  finish(arm);
  assert.equal(arm.coverage, 1);
  assert.equal(arm.strokes, 1);
  assert.equal(components(arm.paper), 1);
  const a = pixelPoint(20 * RETINA + 12),
    b = pixelPoint(20 * RETINA + 36);
  for (let i = 0; i <= 192; i++)
    assert.equal(arm.paper.at([a[0] + ((b[0] - a[0]) * i) / 192, a[1]]), 1);
});
test("disconnected marks stay disconnected while each stroke is continuous", () => {
  const pixels = Array.from(
    { length: RETINA ** 2 },
    (_, i) =>
      +(
        [12, 35].includes(i % RETINA) &&
        Math.floor(i / RETINA) >= 14 &&
        Math.floor(i / RETINA) <= 32
      ),
  );
  const arm = new DrawingArm(pixels);
  finish(arm);
  assert.equal(arm.coverage, 1);
  assert.equal(arm.strokes, 2);
  assert.equal(components(arm.paper), 2);
});
test("erased ink reopens completed work through missing-mark neurons", () => {
  const arm = new DrawingArm(line());
  finish(arm);
  arm.paper.erase(pixelPoint(20 * RETINA + 24), 0.025);
  assert.ok(arm.coverage < 0.9);
  assert.equal(components(arm.paper), 2);
  arm.step();
  assert.ok(arm.brain.missing.state.some((v) => v > 0.1));
  assert.ok(arm.target >= 0);
  finish(arm);
  assert.equal(arm.coverage, 1);
  assert.equal(components(arm.paper), 1);
});
test("ablating missing-mark neurons suppresses repair; restoring them repairs", () => {
  const arm = new DrawingArm(line());
  finish(arm);
  arm.paper.erase(pixelPoint(20 * RETINA + 24), 0.025);
  const before = arm.paper.pixels.slice();
  arm.brain.missing.mask.fill(0);
  for (let i = 0; i < 80; i++) arm.step();
  assert.deepEqual(arm.paper.pixels, before);
  assert.equal(arm.target, -1);
  arm.brain.missing.mask.fill(1);
  finish(arm);
  assert.equal(arm.coverage, 1);
});
test("missing ink alone cannot draw when pencil motors are silenced", () => {
  const arm = new DrawingArm(line());
  arm.brain.motor.mask.fill(0, 15, 17);
  for (let i = 0; i < 400; i++) arm.step();
  assert.equal(arm.coverage, 0);
  assert.ok(arm.paper.pixels.every((v) => v === 0));
  assert.ok(arm.brain.missing.state.every((v) => v > 0.1));
});
