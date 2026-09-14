import { WormArena } from "./brain.js";

let retainedHabitat;

export function mountWorm({ data, $, ctx, metrics, explain, act }) {
  const arena = (retainedHabitat ??= new WormArena(data));
  let tool = "food",
    paused = false,
    speed = 2,
    showOdor = true;
  let accumulator = 0,
    painting = false,
    last = null,
    cursor = arena.cell,
    clock = 0,
    drawnMoves = arena.moves,
    movement = 1;
  let box = { x: 0, y: 0, size: 1 },
    dirty = false;
  $("headline").innerHTML = "Build its world.<br>Follow the food cue.";
  $("intro").textContent =
    "Place food. Draw a barrier. Watch a worm-shaped body follow local chemical cues through a public C. elegans circuit. Open Circuit to inspect the network.";
  $("stage-label").textContent = "C. ELEGANS · INTERACTIVE HABITAT";
  $("stage-detail").textContent = "Local odor → circuit → supplied body";
  $("stage-hint").textContent =
    "Choose a tool, then click or drag in the habitat. Space paints at the keyboard cursor.";
  $("scene").setAttribute(
    "aria-label",
    "Worm habitat. Select food, wall or eraser, then draw. Focus this canvas and use arrow keys and Space to edit with the keyboard.",
  );
  $("controls").innerHTML =
    `<div><h2>Make a world to explore.</h2><p>Food emits a local cue. Walls block diffusion and movement. The body consumes food on contact and rests when there is no usable cue.</p></div><div><label>Paint the habitat</label><div class="buttons" role="group" aria-label="Habitat tools"><button data-worm-tool="food" aria-pressed="true">＋ Food</button><button data-worm-tool="wall" aria-pressed="false">▥ Wall</button><button data-worm-tool="erase" aria-pressed="false">Eraser</button></div></div><div class="buttons"><button id="worm-pause">Pause</button><button id="worm-smell" aria-pressed="true">Smell on</button><button id="worm-odor" aria-pressed="true">Food cue on</button></div><div><label for="worm-speed">Playback speed</label><select id="worm-speed"><option value="1">1× · observe</option><option value="2" selected>2× · explore</option><option value="4">4× · fast</option></select></div><div class="buttons"><button id="worm-reset">Reset maze</button><button id="worm-empty">Open field</button><button id="worm-clear-food">Clear food</button></div>${metrics(
      [
        ["Food patches eaten", "0", "worm-eaten"],
        ["Food remaining", "2", "worm-remaining"],
        ["Motor activity", "—", "worm-activity"],
      ],
    )}<p id="worm-status" role="status">Following local food cues</p>`;
  document.querySelectorAll("[data-worm-tool]").forEach(
    (button) =>
      (button.onclick = () => {
        tool = button.dataset.wormTool;
        document
          .querySelectorAll("[data-worm-tool]")
          .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      }),
  );
  $("worm-pause").onclick = () => {
    paused = !paused;
    $("worm-pause").textContent = paused ? "Resume" : "Pause";
  };
  $("worm-smell").textContent = arena.smell ? "Smell on" : "Smell off";
  $("worm-smell").setAttribute("aria-pressed", String(arena.smell));
  $("worm-smell").onclick = () => {
    arena.smell = !arena.smell;
    $("worm-smell").textContent = arena.smell ? "Smell on" : "Smell off";
    $("worm-smell").setAttribute("aria-pressed", String(arena.smell));
  };
  $("worm-odor").onclick = () => {
    showOdor = !showOdor;
    $("worm-odor").textContent = showOdor ? "Food cue on" : "Food cue off";
    $("worm-odor").setAttribute("aria-pressed", String(showOdor));
  };
  $("worm-speed").onchange = () => (speed = +$("worm-speed").value);
  const reset = (layout) => {
    arena.reset(layout);
    drawnMoves = arena.moves;
    movement = 1;
    accumulator = 0;
    last = null;
    dirty = false;
    painting = false;
  };
  $("worm-reset").onclick = () => reset("maze");
  $("worm-empty").onclick = () => reset("open");
  $("worm-clear-food").onclick = () => {
    arena.food.clear();
    arena.repairOdor();
  };
  explain([
    [
      "A supplied environment",
      "Food sources hold a leaky diffusion field fixed at their cells. Walls prevent flux and body movement. Only adjacent food-cue samples enter the movement adapter.",
    ],
    [
      "A circuit in the loop",
      "The strongest neighboring cue drives the AWA/AWC ports. Mean motor activity gates movement; heading follows the local gradient. Turn smell off to interrupt this path.",
    ],
    [
      "A visible modeling boundary",
      "Consumption removes a food patch. Body steps, heading and feeding are engineered adapters, not learned locomotion or a model of pharyngeal pumping. The circuit comparison below measures neural responses, not foraging skill.",
    ],
  ]);
  function point(i) {
    return [
      box.x + ((i % arena.cols) + 0.5) * box.size,
      box.y + (Math.floor(i / arena.cols) + 0.5) * box.size,
    ];
  }
  function cell(x, y) {
    const a = Math.floor((x - box.x) / box.size),
      b = Math.floor((y - box.y) / box.size);
    return a >= 0 && a < arena.cols && b >= 0 && b < arena.rows
      ? b * arena.cols + a
      : null;
  }
  function paint(i) {
    if (i === null) return;
    const from = last ?? i,
      ax = from % arena.cols,
      ay = Math.floor(from / arena.cols),
      bx = i % arena.cols,
      by = Math.floor(i / arena.cols);
    const n = Math.max(Math.abs(bx - ax), Math.abs(by - ay), 1);
    for (let k = 0; k <= n; k++) {
      const x = Math.round(ax + ((bx - ax) * k) / n),
        y = Math.round(ay + ((by - ay) * k) / n);
      dirty = arena.edit(y * arena.cols + x, tool) || dirty;
    }
    last = i;
    cursor = i;
  }
  function pointer(x, y) {
    painting = true;
    last = null;
    paint(cell(x, y));
  }
  function pointerMove(x, y) {
    if (painting) paint(cell(x, y));
  }
  function pointerEnd() {
    painting = false;
    last = null;
    if (dirty) {
      arena.repairOdor();
      dirty = false;
    }
  }
  function key(event) {
    const x = cursor % arena.cols,
      y = Math.floor(cursor / arena.cols);
    const moves = {
      ArrowLeft: [Math.max(0, x - 1), y],
      ArrowRight: [Math.min(arena.cols - 1, x + 1), y],
      ArrowUp: [x, Math.max(0, y - 1)],
      ArrowDown: [x, Math.min(arena.rows - 1, y + 1)],
    };
    if (moves[event.key]) {
      event.preventDefault();
      const [a, b] = moves[event.key];
      cursor = b * arena.cols + a;
    } else if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      last = null;
      paint(cursor);
      pointerEnd();
    }
  }
  function draw(width, height, dt) {
    clock += dt;
    box.size = Math.min((width - 32) / arena.cols, (height - 42) / arena.rows);
    box.x = (width - arena.cols * box.size) / 2;
    box.y = (height - arena.rows * box.size) / 2;
    if (!paused && !painting) {
      accumulator += dt * speed;
      if (accumulator >= 0.3) {
        act(() => arena.step(true));
        accumulator %= 0.3;
      }
    }
    ctx.fillStyle = "#0e211d";
    ctx.fillRect(box.x, box.y, arena.cols * box.size, arena.rows * box.size);
    for (let i = 0; i < arena.walls.length; i++) {
      const [x, y] = point(i),
        s = box.size;
      if (arena.walls[i]) {
        ctx.fillStyle = "#55706c";
        ctx.fillRect(x - s / 2 + 0.5, y - s / 2 + 0.5, s - 1, s - 1);
      } else if (showOdor && arena.odor[i] > 0.002) {
        ctx.fillStyle = `rgba(160,223,118,${Math.sqrt(arena.odor[i]) * 0.24})`;
        ctx.fillRect(x - s / 2, y - s / 2, s, s);
      }
    }
    for (const i of arena.food) {
      const [x, y] = point(i);
      ctx.fillStyle = "#d9ec83";
      for (let k = 0; k < 7; k++) {
        const a = k * 2.4;
        ctx.beginPath();
        ctx.arc(
          x + Math.cos(a) * box.size * 0.23,
          y + Math.sin(a) * box.size * 0.23,
          Math.max(1, box.size * 0.085),
          0,
          Math.PI * 2,
        );
        ctx.fill();
      }
      ctx.strokeStyle = "#d9ec8360";
      ctx.beginPath();
      ctx.arc(
        x,
        y,
        box.size * (0.38 + 0.03 * Math.sin(clock * 3)),
        0,
        Math.PI * 2,
      );
      ctx.stroke();
    }
    const path = arena.path;
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#87ecc225";
    ctx.beginPath();
    path.forEach((i, k) => {
      const [x, y] = point(i);
      k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    // Interpolation follows actual committed movement, not the controller tick.
    // A tick without a new step must never replay the previous displacement.
    if (arena.moves !== drawnMoves) {
      drawnMoves = arena.moves;
      movement = 0;
    }
    if (!paused && !painting) movement = Math.min(1, movement + dt * speed / 0.3);
    const t = movement,
      tail = path.slice(-7);
    // Interpolate along the traversed centers: the displayed body stays within open cells.
    const centers = tail.map((i, k) => {
      const p = point(i),
        prev = point(tail[Math.max(0, k - 1)]);
      return [prev[0] + (p[0] - prev[0]) * t, prev[1] + (p[1] - prev[1]) * t];
    });
    centers.forEach(([x, y], k) => {
      ctx.fillStyle = k === centers.length - 1 ? "#d9fff0" : "#87ecc2";
      ctx.beginPath();
      ctx.arc(
        x,
        y,
        box.size * (0.14 + (0.16 * k) / centers.length),
        0,
        Math.PI * 2,
      );
      ctx.fill();
      if (k) {
        ctx.strokeStyle = "#87ecc2";
        ctx.lineWidth = box.size * (0.22 + (0.28 * k) / centers.length);
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(...centers[k - 1]);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    });
    if (document.activeElement === $("scene")) {
      const [x, y] = point(cursor);
      ctx.strokeStyle = "white";
      ctx.lineWidth = 2;
      ctx.strokeRect(x - box.size / 2, y - box.size / 2, box.size, box.size);
    }
    $("worm-eaten").textContent = arena.eaten;
    $("worm-remaining").textContent = arena.food.size;
    $("worm-activity").textContent = arena.activity.toFixed(4);
    if ($("worm-status").textContent !== arena.status)
      $("worm-status").textContent = arena.status;
    $("stage-readout").textContent = paused
      ? "Paused · you can still edit"
      : painting
        ? "Editing · release to update the food cue"
        : `${arena.moves} body steps · ${speed}×`;
  }
  return {
    draw,
    pointer,
    pointerMove,
    pointerEnd,
    key,
    brain: () => ({
      adapters: "Supplied: odor field · heading · body mechanics",
      regionLabels: {
        sensory: "Sensory input",
        interneuron: "Interneurons",
        motor: "Motor output",
      },
      behavior: paused
        ? { label: "Paused", tone: "neutral" }
        : arena.status.includes("consumed")
          ? { label: "Food consumed", tone: "positive" }
          : arena.status.includes("Following")
            ? { label: "Seeking food", tone: "seeking" }
            : { label: "No usable food cue", tone: "neutral" },
      ...arena.brain(),
    }),
    snapshot: () => ({ ...arena.snapshot(), paused }),
  };
}
