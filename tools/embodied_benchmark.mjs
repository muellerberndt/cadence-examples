import { Arm, drawingTargets, forward, motorSettling } from "../shared/embodied.js";
const arm = [];
for (const image of ["flower", "leaf", "spiral"])
  for (const disturbance_tick of [60, 120, 240])
    for (const closed of [true, false]) {
      const a = new Arm(drawingTargets(image));
      a.closed = closed;
      for (let t = 0; t < 4000; t++) {
        if (t === disturbance_tick) a.disturb();
        a.step();
      }
      arm.push({
        image,
        disturbance_tick,
        feedback: closed,
        coverage: a.coverage,
        covered: a.covered.size,
        targets: a.targets.length,
        attempted: a.attempted.size,
        ink_samples: a.ink.length,
        final_tip: forward(a.q),
      });
    }
const q = [-1.8, 1.1],
  target = [0.45, 0.35];
console.log(
  JSON.stringify({
    arm,
    parity: {
      q,
      target,
      motor_state: motorSettling(q, target).state,
    },
    boundaries: [
      "Eye-arm uses an image-to-edge adapter and supplied arm geometry/Jacobian. Visual and motor patches settle jointly.",
      "Coverage is target points within 0.022 normalized units of deposited ink, not perceptual drawing quality.",
      "The comparison removes visual/proprioceptive feedback from the same controller. It is not a trained MLP comparison.",
    ],
  }),
);
