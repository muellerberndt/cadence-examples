import {
  Mouse,
  Arm,
  drawingTargets,
  neighbors,
  forward,
  motorSettling,
  TaskLessons,
  stations,
} from "../shared/embodied.js";
function bfs(world, start) {
  const q = [start],
    prev = new Map([[start, -1]]);
  for (let k = 0; k < q.length; k++) {
    const i = q[k];
    if (i === world.goal) {
      const route = [];
      for (let j = i; j !== -1; j = prev.get(j)) route.unshift(j);
      return route;
    }
    for (const j of neighbors(world, i))
      if (!prev.has(j)) {
        prev.set(j, i);
        q.push(j);
      }
  }
  return [];
}
function legal(m) {
  return !m.world.grid[Math.floor(m.y) * m.world.cols + Math.floor(m.x)];
}
const mouse = [];
for (const seed of [13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123]) {
  for (const condition of ["new_maze", "changed_corridor", "moved_goal"]) {
    const m = new Mouse(seed),
      original = bfs(m.world, m.cell);
    let removed = -1;
    if (condition === "changed_corridor")
      for (const cell of original.slice(2, -1)) {
        m.world.grid[cell] = 1;
        if (bfs(m.world, m.cell).length) {
          removed = cell;
          break;
        }
        m.world.grid[cell] = 0;
      }
    if (condition === "moved_goal") {
      const floors = m.world.grid
        .map((v, i) => (v ? -1 : i))
        .filter((i) => i >= 0 && i !== m.cell);
      m.world.goal = floors[(seed * 17) % floors.length];
    }
    m.repair();
    let collision = false,
      ticks = 0;
    for (; ticks < 12000 && m.cell !== m.world.goal; ticks++) {
      m.step(0.05);
      if (!legal(m)) collision = true;
    }
    mouse.push({
      seed,
      condition,
      removed,
      success: m.cell === m.world.goal,
      collision,
      moves: m.moves,
      shortest_path_moves: bfs(m.world, 1 + m.world.cols).length - 1,
      residual: m.circuit.residual,
      frozen_route_valid:
        original.at(-1) === m.world.goal &&
        original.every((i) => !m.world.grid[i]),
      bfs_replanner_success: !!bfs(m.world, 1 + m.world.cols).length,
    });
  }
}
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
const m = new Mouse(13),
  q = [-1.8, 1.1],
  target = [0.45, 0.35];
const lessons = new TaskLessons(),
  before = lessons.recall(3);
lessons.teach(3, 3);
const taught = lessons.recall(3);
lessons.teach(3, 2);
const tasks = {
  unfamiliar_before: before === null,
  new_task_after_one_write: taught === 3,
  revised_task: lessons.recall(3) === 2,
  earlier_tasks_retained: [0, 1, 2].every((i) => lessons.recall(i) === i),
  transfer: [],
};
for (const seed of [13, 23, 33]) {
  const mouse = new Mouse(seed);
  mouse.world.goal = stations(mouse.world)[lessons.recall(3)];
  mouse.repair();
  for (let t = 0; t < 12000 && mouse.cell !== mouse.world.goal; t++)
    mouse.step(0.05);
  tasks.transfer.push({ seed, success: mouse.cell === mouse.world.goal });
}
console.log(
  JSON.stringify({
    mouse,
    arm,
    tasks,
    parity: {
      world: m.world,
      maze_state: m.circuit.state,
      q,
      target,
      motor_state: motorSettling(q, target).state,
    },
    boundaries: [
      "Mouse receives the full maze as a visual occupancy map; it does not learn unknown walls.",
      "The spatial field uses supplied neighbor connections and a contractive tanh rule; BFS also solves the maze.",
      "Motor commands move a simplified body with explicit collision checks. No validated mouse biomechanics.",
      "Eye-arm uses an image-to-edge adapter and supplied arm geometry/Jacobian. Visual and motor patches settle jointly.",
      "Coverage is target points within 0.022 normalized units of deposited ink, not perceptual drawing quality.",
      "The comparison removes visual/proprioceptive feedback from the same controller. It is not a trained MLP comparison.",
    ],
  }),
);
