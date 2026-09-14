import { Mouse } from "./brain.js";
import { TaskLessons, stations } from "../shared/embodied.js";
export function mountEmbodied(mode, api) {
  const {
    $,
    metrics,
    explain,
    table,
    pct,
    mean,
    ctx,
    circle,
    line,
    text,
    evidence,
  } = api;
  let lessons,
    cue = 0,
    waiting = false;
  let mouse,
    seed = 13,
    paused = false,
    heat = true,
    refreshCells = () => {};
  const title =
    mode === "mouse"
      ? "Teach a new task.<br>Watch it find a way."
      : "An eye sees the error.<br>An arm makes the correction.";
  $("headline").innerHTML = title;
  $("intro").textContent =
    mode === "mouse"
      ? "Pick a task and the mouse recalls where to go. Teach a task by choosing its destination: one memory write, and the mouse walks there. Earlier lessons are kept."
      : "Present an outline. Watch visual and motor regions settle together, move the joints, and compare the drawing with the image again.";
  const worldXY = (w, h) => {
    const cell = Math.min((w - 42) / 19, (h - 35) / 13);
    return { cell, ox: (w - cell * 19) / 2, oy: (h - cell * 13) / 2 };
  };
  if (mode === "mouse") {
    try {
      lessons = new TaskLessons(
        JSON.parse(
          localStorage.getItem("cadence.mouse.lessons.v1") || "null",
        ) ?? undefined,
      );
    } catch {
      lessons = new TaskLessons();
    }
    mouse = new Mouse(seed);
    $("stage-label").textContent =
      "LIVE EMBODIMENT · VISUAL MAP + SPATIAL FIELD + MOTOR";
    $("stage-detail").textContent = "Simplified mouse · supplied maze rules";
    $("stage-hint").textContent =
      "Click a wall or corridor to change it. Green shows the settled goal field.";
    $("scene").setAttribute(
      "aria-label",
      "A mouse navigates a labyrinth toward cheese. Brightness shows the spatial goal field. Controls can alter the maze and goal.",
    );
    $("controls").innerHTML =
      `<div><h2>Several parts. One closed loop.</h2><p>Vision updates the map. Neighboring place neurons settle. Motor output moves the body; position readback selects the next action.</p></div><div><label for="task-cue">Task · the mouse reacts at once</label><select id="task-cue"><option value="0">Find cheese</option><option value="1">Go home</option><option value="2">Get water</option><option value="3">New task · untaught</option></select><label for="teach-goal" style="margin-top:9px">Teach this task to go to</label><select id="teach-goal"><option value="0">Cheese</option><option value="1">Home</option><option value="2">Water</option><option value="3">Flag</option></select><div class="buttons" style="margin-top:8px"><button id="teach-task">Teach</button><button id="perform-task">Go again</button></div><p id="lesson-status" aria-live="polite">Three tasks are taught. Choose New task, pick a destination and press Teach.</p></div><div class="buttons"><button id="new-maze">New maze</button><button id="move-goal">Move goal</button><button id="pause">Pause</button></div><details><summary>Edit the maze with the keyboard</summary><label for="edit-cell">Edit corridor / wall</label><select id="edit-cell"></select><button id="toggle-wall" style="margin-top:8px">Toggle selected cell</button><p>Unreachable goals stop the mouse; it does not walk through walls.</p></details><div class="buttons"><button id="heat" aria-pressed="true">Show goal field</button></div>${metrics(
        [
          ["Body moves", "0", "mouse-moves"],
          ["Spatial residual", "0", "spatial-residual"],
          ["Goal", "Navigating", "mouse-state"],
        ],
      )}`;
    function choices() {
      $("edit-cell").innerHTML = mouse.world.grid
        .map((v, i) => {
          const x = i % 19,
            y = Math.floor(i / 19);
          return x > 0 && x < 18 && y > 0 && y < 12
            ? `<option value="${i}">Column ${x}, row ${y} · ${v ? "wall" : "corridor"}</option>`
            : "";
        })
        .join("");
    }
    function perform() {
      const destination = lessons.recall(cue);
      if (destination === null) {
        waiting = true;
        $("lesson-status").textContent =
          "No lesson for this task. Pick where it should go and press Teach.";
        return;
      }
      waiting = false;
      mouse.world.goal = stations(mouse.world)[destination];
      mouse.bindTask(lessons.memory, cue, destination);
      $("lesson-status").textContent =
        `Task ${cue + 1}: memory recalls ${["cheese", "home", "water", "flag"][destination]}, and the mouse heads there.`;
    }
    $("task-cue").onchange = () => {
      cue = +$("task-cue").value;
      perform();
    };
    $("perform-task").onclick = perform;
    $("teach-task").onclick = () => {
      lessons.teach(cue, +$("teach-goal").value);
      try {
        localStorage.setItem(
          "cadence.mouse.lessons.v1",
          JSON.stringify(lessons),
        );
      } catch {}
      if (cue === 3)
        $("task-cue").options[3].textContent = "New task · learned";
      perform();
      $("lesson-status").textContent =
        `Taught in one memory write: task ${cue + 1} goes to ${["cheese", "home", "water", "flag"][lessons.recall(cue)]}. Other tasks are kept, saved in this browser.`;
    };
    $("new-maze").onclick = () => {
      seed += 10;
      mouse = new Mouse(seed);
      motorButton.textContent = "Motor neurons on";
      motorButton.setAttribute("aria-pressed", "true");
      perform();
      choices();
    };
    $("move-goal").onclick = () => {
      const floors = mouse.world.grid
        .map((v, i) => (v ? -1 : i))
        .filter((i) => i >= 0 && i !== mouse.cell);
      mouse.world.goal = floors[(seed * 17 + mouse.moves * 3) % floors.length];
      mouse.repair();
    };
    // Rebuild the cell list after any edit, keeping the selected cell, so each
    // option names the cell's current state (wall or corridor).
    refreshCells = () => {
      const selected = $("edit-cell").value;
      choices();
      $("edit-cell").value = selected;
    };
    $("toggle-wall").onclick = () => {
      if (mouse.toggle(+$("edit-cell").value)) refreshCells();
    };
    $("pause").onclick = () => {
      paused = !paused;
      $("pause").textContent = paused ? "Resume" : "Pause";
    };
    $("heat").onclick = () => {
      heat = !heat;
      $("heat").setAttribute("aria-pressed", String(heat));
    };
    const motorButton = document.createElement("button");
    motorButton.id = "mouse-motors";
    motorButton.textContent = "Motor neurons on";
    motorButton.setAttribute("aria-pressed", "true");
    motorButton.onclick = () => {
      const enabled = !mouse.nerves.mask[2];
      mouse.nerves.mask.fill(+enabled, 2);
      motorButton.setAttribute("aria-pressed", String(enabled));
      motorButton.textContent = enabled
        ? "Motor neurons on"
        : "Motor neurons off";
    };
    $("controls").append(motorButton);
    choices();
    perform();
    if (lessons.known.has(3)) {
      $("task-cue").options[3].textContent = "New task · learned";
      $("lesson-status").textContent =
        "Saved lessons restored from this browser.";
    }
    explain([
      [
        "Task memory + visual map",
        "Three initial demonstrations teach named task cues. Teach or revise a fourth in one residual write. A supplied visual map locates the recalled destination in each new maze.",
      ],
      [
        "Spatial equilibrium",
        "A goal drive spreads through a contractive network of place neurons. Each neuron reads neighboring values; a local gradient guides the next step.",
      ],
      [
        "Motor / body feedback",
        "A moving body reports its actual cell. Editing the map or moving the cheese makes the spatial field settle again and changes the motor choices.",
      ],
    ]);
    $("evidence-title").textContent = "A changed world gets a fresh route";
    $("evidence-note").textContent =
      "Reference spatial-field/body benchmark (before the expanded motor circuit): twelve generated mazes, three conditions each: initial navigation, a changed corridor where an alternate path exists, and a moved goal. Every body position is checked against walls.";
    $("evidence-table").innerHTML = table(
      [
        "Condition",
        "Cadence field + body",
        "Frozen original route still valid",
        "BFS finds a route",
      ],
      ["new_maze", "changed_corridor", "moved_goal"].map((c) => {
        const rows = evidence.mouse.filter((r) => r.condition === c);
        return [
          {
            new_maze: "New maze",
            changed_corridor: "Changed corridor",
            moved_goal: "Moved goal",
          }[c],
          `${rows.filter((r) => r.success).length}/${rows.length}`,
          `${rows.filter((r) => r.frozen_route_valid).length}/${rows.length}`,
          `${rows.filter((r) => r.bfs_replanner_success).length}/${rows.length}`,
        ];
      }),
    );
    $("boundary").textContent =
      "Task cues are explicit symbolic keys, not natural-language instructions. Lessons persist only in this browser; a dictionary can also retain these mappings. The graph and full occupancy map are supplied; the mouse is a simplified planar body. Cadence solves a tanh spatial field, and a local readout chooses an ascending neighbor. BFS is a strong conventional control and also succeeds. The frozen-route control demonstrates why stale plans need feedback; it is not an MLP trained to navigate. Changing a corridor may be impossible without disconnecting a perfect maze; each scheduled edit is recorded in the evidence.";
  }
  const evidenceLink = document.querySelector(".evidence details a");
  evidenceLink.href = "evidence/composite_evidence.json";
  return {
    draw(w, h, dt) {
      if (mode === "mouse") {
        if (!paused && !waiting) api.act(() => mouse.step(dt, true));
        const { cell, ox, oy } = worldXY(w, h),
          world = mouse.world,
          state = mouse.circuit.state,
          max = Math.max(...state),
          xy = (i) => [
            ox + ((i % 19) + 0.5) * cell,
            oy + (Math.floor(i / 19) + 0.5) * cell,
          ];
        world.grid.forEach((wall, i) => {
          const x = ox + (i % 19) * cell,
            y = oy + Math.floor(i / 19) * cell;
          ctx.fillStyle = wall
            ? "#304740"
            : heat
              ? `rgba(135,236,194,${0.025 + 0.25 * Math.sqrt(state[i] / Math.max(max, 1e-12))})`
              : "#10221e";
          ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        });
        for (let i = 1; i < mouse.path.length; i++)
          line(xy(mouse.path[i - 1]), xy(mouse.path[i]), "#87ecc255", 2);
        stations(world).forEach((i, k) => {
          const [sx, sy] = xy(i);
          if (i !== world.goal) {
            circle(sx, sy, 5, ["#ffd580", "#bfacfa", "#8acfff", "#ffb17a"][k]);
            text(["C", "H", "W", "F"][k], sx, sy + 3, "#0a1715", 8, "center");
          }
        });
        const [gx, gy] = xy(world.goal);
        const goalKind = stations(world).indexOf(world.goal);
        if (goalKind === 3) {
          line([gx - 5, gy - 10], [gx - 5, gy + 10], "#ffb17a", 2);
          ctx.fillStyle = "#ffb17a";
          ctx.beginPath();
          ctx.moveTo(gx - 4, gy - 10);
          ctx.lineTo(gx + 9, gy - 5);
          ctx.lineTo(gx - 4, gy);
          ctx.fill();
        } else if (goalKind === 1) {
          ctx.fillStyle = "#bfacfa";
          ctx.fillRect(gx - 7, gy - 2, 14, 11);
          ctx.beginPath();
          ctx.moveTo(gx - 10, gy - 2);
          ctx.lineTo(gx, gy - 11);
          ctx.lineTo(gx + 10, gy - 2);
          ctx.fill();
        } else if (goalKind === 2) {
          circle(gx, gy, 9, "#8acfff");
          text("W", gx, gy + 4, "#092028", 11, "center");
        } else {
          ctx.fillStyle = "#ffd580";
          ctx.beginPath();
          ctx.moveTo(gx - 8, gy + 7);
          ctx.lineTo(gx + 9, gy + 7);
          ctx.lineTo(gx + 5, gy - 8);
          ctx.closePath();
          ctx.fill();
          circle(gx, gy + 3, 2, "#ba8f46");
          circle(gx + 4, gy - 2, 1.5, "#ba8f46");
        }
        const x = ox + mouse.x * cell,
          y = oy + mouse.y * cell;
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(mouse.heading);
        ctx.strokeStyle = "#e5c3b0";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(-8, 0);
        ctx.quadraticCurveTo(-17, 8, -21, 3);
        ctx.stroke();
        ctx.fillStyle = "#d5c8b8";
        ctx.beginPath();
        ctx.ellipse(0, 0, 10, 6, 0, 0, Math.PI * 2);
        ctx.fill();
        circle(6, -5, 3, "#edcfc4");
        circle(6, 5, 3, "#edcfc4");
        circle(10, 0, 2, "#f3bfb4");
        circle(7, -2, 1, "#17221f");
        line([9, -2], [15, -5], "#e7ddd0", 0.7);
        line([9, 2], [15, 5], "#e7ddd0", 0.7);
        ctx.restore();
        $("mouse-moves").textContent = mouse.moves;
        $("spatial-residual").textContent =
          mouse.circuit.residual.toExponential(1);
        $("mouse-state").textContent = waiting
          ? "Needs a lesson"
          : mouse.cell === world.goal
            ? "Reached"
            : mouse.next() < 0
              ? "No route"
              : "Navigating";
        $("stage-readout").textContent =
          `${world.grid.filter((v) => !v).length} place neurons · ${mouse.circuit.steps} settling steps`;
      }
    },
    pointer(x, y, w, h) {
      if (mode !== "mouse") return;
      const { cell, ox, oy } = worldXY(w, h),
        a = Math.floor((x - ox) / cell),
        b = Math.floor((y - oy) / cell);
      if (a >= 0 && a < 19 && b >= 0 && b < 13 && mouse.toggle(b * 19 + a))
        refreshCells();
    },
    brain() {
      if (mode === "mouse") {
        return Object.assign(mouse.joint, {
          behavior: paused
            ? { label: "Paused", tone: "neutral" }
            : waiting
              ? { label: "Waiting for a lesson", tone: "neutral" }
              : mouse.cell === mouse.world.goal
              ? { label: "Goal reached", tone: "positive" }
              : mouse.next() < 0
                ? { label: "Route blocked", tone: "correcting" }
                : { label: "Navigating", tone: "seeking" },
        });
      }
      return null;
    },
    snapshot() {
      return {
        cell: mouse.cell,
        x: mouse.x,
        y: mouse.y,
        motors: mouse.nerves.state.slice(),
        goal: mouse.world.goal,
        lessons: lessons.records,
        moves: mouse.moves,
        residual: mouse.circuit.residual,
      };
    },
  };
}
